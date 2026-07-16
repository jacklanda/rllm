"""Data-parallel SGLang rollout engine for agentic eval."""

from __future__ import annotations

import asyncio
import dataclasses
import multiprocessing as mp
import os
import queue
import traceback
import uuid
from dataclasses import dataclass
from typing import Any

from rllm.engine.rollout.rollout_engine import ModelOutput, RolloutEngine


@dataclass
class _WorkerHandle:
    worker_id: int
    devices: str
    request_queue: Any
    response_queue: Any
    process: mp.Process
    lock: asyncio.Lock


def _parse_device_list(devices: str | None, required: int) -> tuple[list[str], bool]:
    explicit = bool(devices)
    if devices:
        parsed = [d.strip() for d in devices.split(",") if d.strip()]
    else:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        parsed = [d.strip() for d in visible.split(",") if d.strip()]

    if not parsed:
        parsed = [str(i) for i in range(required)]
    if len(parsed) < required:
        raise ValueError(
            f"Need {required} visible devices for data-parallel SGLang, got {parsed}."
        )
    return parsed[:required], explicit


def _worker_main(
    worker_id: int,
    request_queue: Any,
    response_queue: Any,
    ready_queue: Any,
    engine_config: dict[str, Any],
) -> None:
    os.environ.setdefault("SGLANG_APPLY_CONFIG_BACKUP", "none")
    try:
        print(f"[evals-sglang] worker {worker_id}: importing engine", flush=True)
        from evals.sglang_engine import SGLangEngine

        print(f"[evals-sglang] worker {worker_id}: constructing engine", flush=True)
        engine = SGLangEngine(**engine_config)
        print(f"[evals-sglang] worker {worker_id}: warming up engine", flush=True)
        asyncio.run(
            engine.get_model_response(
                [{"role": "user", "content": "ping"}],
                application_id=f"sglang-warmup-{worker_id}",
                max_tokens=1,
            )
        )
        print(f"[evals-sglang] worker {worker_id}: ready", flush=True)
        ready_queue.put((worker_id, None))
    except BaseException:
        ready_queue.put((worker_id, traceback.format_exc()))
        return

    try:
        while True:
            item = request_queue.get()
            if item is None:
                break

            request_id, messages, kwargs = item
            try:
                output = asyncio.run(engine.get_model_response(messages, **kwargs))
                response_queue.put((request_id, None, dataclasses.asdict(output)))
            except BaseException:
                response_queue.put((request_id, traceback.format_exc(), None))
    finally:
        try:
            engine.shutdown()
        except BaseException:
            pass


class SGLangDataParallelEngine(RolloutEngine):
    """Route rollout requests across multiple single-node SGLang workers."""

    def __init__(
        self,
        model: str,
        data_parallel_size: int = 1,
        data_parallel_devices: str = "",
        max_prompt_length: int = 16384,
        max_response_length: int = 8192,
        max_model_length: int | None = None,
        sampling_params: dict | None = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        trust_remote_code: bool = True,
        disable_thinking: bool = False,
        no_system_prompt: bool = False,
        engine_kwargs: dict | None = None,
        **_: Any,
    ) -> None:
        if data_parallel_size < 1:
            raise ValueError("data_parallel_size must be >= 1")
        if tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be >= 1")

        self.model = model
        self.data_parallel_size = data_parallel_size
        self.tensor_parallel_size = tensor_parallel_size
        self._next_worker = 0
        self._workers: list[_WorkerHandle] = []
        self._closed = False

        required_devices = data_parallel_size * tensor_parallel_size
        devices, explicit_devices = _parse_device_list(
            data_parallel_devices or None, required_devices
        )
        groups = [
            ",".join(devices[i : i + tensor_parallel_size])
            for i in range(0, required_devices, tensor_parallel_size)
        ]

        ctx = mp.get_context("spawn")
        ready_queue = ctx.Queue()
        base_engine_config = {
            "model": model,
            "max_prompt_length": max_prompt_length,
            "max_response_length": max_response_length,
            "max_model_length": max_model_length,
            "sampling_params": dict(sampling_params or {}),
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "trust_remote_code": trust_remote_code,
            "disable_thinking": disable_thinking,
            "no_system_prompt": no_system_prompt,
            "engine_kwargs": dict(engine_kwargs or {}),
        }

        pending_workers: set[int] = set()
        try:
            for worker_id, worker_devices in enumerate(groups):
                worker_engine_config = dict(base_engine_config)
                worker_engine_kwargs = dict(base_engine_config["engine_kwargs"])
                if explicit_devices:
                    sglang_gpu_ids = [int(d) for d in worker_devices.split(",")]
                else:
                    first = worker_id * tensor_parallel_size
                    sglang_gpu_ids = list(range(first, first + tensor_parallel_size))
                if len(sglang_gpu_ids) > 1:
                    gpu_id_step = sglang_gpu_ids[1] - sglang_gpu_ids[0]
                    if any(
                        b - a != gpu_id_step
                        for a, b in zip(sglang_gpu_ids, sglang_gpu_ids[1:])
                    ):
                        raise ValueError(
                            "SGLang data-parallel device groups must have a uniform "
                            f"GPU id step, got {worker_devices}."
                        )
                    worker_engine_kwargs.setdefault("gpu_id_step", gpu_id_step)
                worker_engine_kwargs.setdefault("base_gpu_id", sglang_gpu_ids[0])
                worker_engine_config["engine_kwargs"] = worker_engine_kwargs

                request_queue = ctx.Queue()
                response_queue = ctx.Queue()
                process = ctx.Process(
                    target=_worker_main,
                    args=(
                        worker_id,
                        request_queue,
                        response_queue,
                        ready_queue,
                        worker_engine_config,
                    ),
                    name=f"EvalsSGLangDP-{worker_id}",
                )
                process.start()
                self._workers.append(
                    _WorkerHandle(
                        worker_id=worker_id,
                        devices=worker_devices,
                        request_queue=request_queue,
                        response_queue=response_queue,
                        process=process,
                        lock=asyncio.Lock(),
                    )
                )
                pending_workers.add(worker_id)
        except BaseException:
            self.shutdown()
            raise

        # Match RayVLLMEngine's startup shape: launch every replica first so
        # model imports and weight loading happen concurrently across devices,
        # then block until all workers have reported readiness.
        self._wait_until_ready(ready_queue, pending=pending_workers)

        self.validate = False

    def _wait_until_ready(self, ready_queue: Any, pending: set[int]) -> None:
        errors: list[str] = []
        while pending:
            try:
                worker_id, error = ready_queue.get(timeout=5)
            except queue.Empty:
                dead = [w for w in self._workers if not w.process.is_alive()]
                if dead:
                    errors.extend(
                        f"worker {w.worker_id} on devices {w.devices} exited with "
                        f"code {w.process.exitcode}"
                        for w in dead
                    )
                    break
                continue

            if worker_id not in pending:
                if error:
                    errors.append(
                        f"worker {worker_id} failed during startup after readiness "
                        f"was already observed:\n{error}"
                    )
                continue

            pending.discard(worker_id)
            if error:
                errors.append(f"worker {worker_id} failed during startup:\n{error}")
                break

        if errors:
            self.shutdown()
            raise RuntimeError("\n".join(errors))

    def _pick_worker(self) -> _WorkerHandle:
        worker = self._workers[self._next_worker % len(self._workers)]
        self._next_worker += 1
        return worker

    @staticmethod
    def _wait_for_response(worker: _WorkerHandle) -> tuple[str, str | None, Any]:
        while True:
            try:
                return worker.response_queue.get(timeout=1)
            except queue.Empty:
                if not worker.process.is_alive():
                    raise RuntimeError(
                        f"SGLang DP worker {worker.worker_id} on devices "
                        f"{worker.devices} exited with code {worker.process.exitcode}"
                    )

    async def get_model_response(self, messages: list[dict], **kwargs) -> ModelOutput:
        if self._closed:
            raise RuntimeError("SGLangDataParallelEngine has been shut down")

        worker = self._pick_worker()
        async with worker.lock:
            request_id = str(uuid.uuid4())
            worker.request_queue.put((request_id, messages, kwargs))
            response_id, error, payload = await asyncio.to_thread(
                self._wait_for_response, worker
            )

        if response_id != request_id:
            raise RuntimeError(
                f"SGLang DP worker {worker.worker_id} returned response "
                f"{response_id}, expected {request_id}"
            )
        if error:
            raise RuntimeError(
                f"SGLang DP worker {worker.worker_id} failed:\n{error}"
            )
        return ModelOutput(**payload)

    async def wake_up(self) -> None:
        return None

    async def sleep(self) -> None:
        return None

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True

        for worker in self._workers:
            try:
                worker.request_queue.put(None)
            except BaseException:
                pass
        for worker in self._workers:
            worker.process.join(timeout=30)
            if worker.process.is_alive():
                worker.process.terminate()
                worker.process.join(timeout=10)
            if worker.process.is_alive():
                worker.process.kill()


__all__ = ["SGLangDataParallelEngine"]
