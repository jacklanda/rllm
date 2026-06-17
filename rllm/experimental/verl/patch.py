"""Monkey-patches for Verl and vLLM within the rLLM unified trainer.

All patches are applied lazily (on first call) and are idempotent — calling
them multiple times is safe.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_VERL_DYNAMIC_BATCH_PATCHED = False
_VERL_QWEN3_VL_DUMMY_INPLACE_PATCHED = False
_VERL_TENSORDICT_JAGGED_PATCHED = False
_VERL_VLLM_TYPED_TOKEN_PROMPT_PATCHED = False
_VERL_RAY_LOCAL_RANK_ENV_PATCHED = False
_VERL_VLLM_SERVER_VISIBLE_DEVICES_PATCHED = False
_VERL_VLLM_ROLLOUT_LOCAL_RANK_PATCHED = False
_VERL_VLLM_ASYNC_SERVER_RAY_GET_PATCHED = False
_VLLM_WORKER_SHARED_LOCK_WARNING_PATCHED = False

_VERL_WORKER_PROCESS_SETUP_HOOK = "rllm.experimental.verl.patch.apply_all_verl_patches"


def _split_visible_devices(value: str | None) -> list[str]:
    return [device.strip() for device in str(value or "").split(",") if device.strip()]


def _node_visible_devices(local_world_size: int) -> list[str]:
    """Return the physical CUDA ids rLLM should map node-local ranks onto."""
    import os

    devices = _split_visible_devices(os.environ.get("RLLM_CUDA_VISIBLE_DEVICES"))
    if devices:
        return devices

    devices = _split_visible_devices(os.environ.get("CUDA_VISIBLE_DEVICES"))
    if devices:
        return devices

    return [str(rank) for rank in range(local_world_size)]


def _bind_current_worker_cuda_device(worker=None) -> int | None:
    """Select the CUDA device for the current Ray worker process.

    Do not mutate CUDA_VISIBLE_DEVICES here. Ray and torch may already have
    initialized the process-local CUDA device table, so changing the env var at
    this point can leave torch still seeing the old table while our code thinks
    it narrowed visibility to one device. The reliable operation before NCCL
    init is setting torch's current device to the node-local rank.
    """
    import os

    from verl.utils.device import get_torch_device

    selected_device = os.environ.get("RLLM_WORKER_CUDA_VISIBLE_DEVICES")
    visible_devices = _split_visible_devices(os.environ.get("CUDA_VISIBLE_DEVICES"))
    if not visible_devices:
        visible_devices = _split_visible_devices(os.environ.get("RLLM_CUDA_VISIBLE_DEVICES"))

    if selected_device and visible_devices:
        try:
            cuda_rank = visible_devices.index(selected_device)
        except ValueError:
            cuda_rank = int(os.environ.get("RLLM_NODE_LOCAL_RANK", os.environ.get("LOCAL_RANK", "0")))
    elif len(visible_devices) == 1:
        cuda_rank = 0
    else:
        cuda_rank = int(os.environ.get("RLLM_NODE_LOCAL_RANK", os.environ.get("LOCAL_RANK", "0")))

    device_count = get_torch_device().device_count()
    if device_count > 0:
        cuda_rank %= device_count
    get_torch_device().set_device(cuda_rank)

    os.environ["LOCAL_RANK"] = str(cuda_rank)
    if worker is not None:
        worker.__dict__["_local_rank"] = cuda_rank
        if selected_device:
            worker.__dict__["_cuda_visible_devices"] = selected_device
    return cuda_rank


def _vllm_server_runtime_env(env_vars: dict[str, str]) -> dict[str, object]:
    """Build a Ray runtime_env for vLLM server actors.

    The vLLM server actor is launched with a per-node CUDA visibility override.
    Supplying an actor-specific runtime_env can replace the job-level worker
    setup hook, so carry the hook here as well; otherwise server-only patches
    such as the typed-token prompt fix never run in vLLMHttpServer processes.
    """
    return {
        "env_vars": env_vars,
        "worker_process_setup_hook": _VERL_WORKER_PROCESS_SETUP_HOOK,
    }


# ---------------------------------------------------------------------------
# Verl Ray worker env: provide LOCAL_RANK / LOCAL_WORLD_SIZE
# ---------------------------------------------------------------------------


def patch_verl_ray_worker_local_rank_env() -> None:
    """Patch Verl Ray worker creation to export standard local-rank env vars.

    Some Verl versions only set ``RAY_LOCAL_WORLD_SIZE`` when creating
    RayWorkerGroup actors.  ``verl.single_controller.base.worker.Worker`` reads
    ``LOCAL_RANK`` and ``LOCAL_WORLD_SIZE`` instead, defaulting both to a
    single-rank process when they are absent.  With FSDP2 this makes every
    local rank select cuda:0 and NCCL aborts with "Duplicate GPU detected".

    rLLM/Verl colocates multiple roles through fractional Ray GPU requests.
    Ray may then give multiple FSDP ranks the same accelerator id, which makes
    NCCL report duplicate GPUs.  We cannot put our chosen
    ``CUDA_VISIBLE_DEVICES`` into the actor ``runtime_env`` because Ray reads
    that variable while translating placement-group accelerator ids; with a
    single-device runtime value and a non-zero bundle id, Ray itself can crash
    with ``IndexError`` before the actor starts.  Instead, pass the intended
    physical device through rLLM-private env vars and bind CUDA after Ray has
    finished launching the worker process.
    """
    global _VERL_RAY_LOCAL_RANK_ENV_PATCHED
    if _VERL_RAY_LOCAL_RANK_ENV_PATCHED:
        return

    from verl.single_controller.base import Worker
    from verl.single_controller.ray import RayWorkerGroup
    from verl.workers import fsdp_workers

    _original_create_worker = RayWorkerGroup._create_worker
    _original_worker_init = Worker.__init__
    _original_actor_rollout_ref_worker_init = fsdp_workers.ActorRolloutRefWorker.__init__

    def _patched_create_worker(
        self,
        rank,
        pg_idx,
        pg,
        local_rank,
        resource_pool,
        ray_cls_with_init,
        worker_env,
        detached,
    ):
        local_world_size = resource_pool.store[0]
        node_visible_devices = _node_visible_devices(local_world_size)
        visible_device = node_visible_devices[local_rank % len(node_visible_devices)]
        worker_env = dict(worker_env or {})
        worker_env.setdefault("LOCAL_RANK", str(local_rank))
        worker_env.setdefault("LOCAL_WORLD_SIZE", str(local_world_size))
        worker_env.setdefault("RLLM_NODE_LOCAL_RANK", str(local_rank))
        if getattr(self, "device_name", "cuda") == "cuda":
            worker_env.setdefault("RLLM_WORKER_CUDA_VISIBLE_DEVICES", visible_device)
        return _original_create_worker(
            self,
            rank,
            pg_idx,
            pg,
            local_rank,
            resource_pool,
            ray_cls_with_init,
            worker_env,
            detached,
        )

    def _patched_worker_init(self, *args, **kwargs):
        try:
            _bind_current_worker_cuda_device()
            _original_worker_init(self, *args, **kwargs)
            _bind_current_worker_cuda_device(self)
        except Exception as exc:
            logger.warning("Failed to set worker CUDA device from LOCAL_RANK: %s", exc)
            raise

    def _patched_actor_rollout_ref_worker_init(self, *args, **kwargs):
        _bind_current_worker_cuda_device(self)
        return _original_actor_rollout_ref_worker_init(self, *args, **kwargs)

    RayWorkerGroup._create_worker = _patched_create_worker
    Worker.__init__ = _patched_worker_init
    fsdp_workers.ActorRolloutRefWorker.__init__ = _patched_actor_rollout_ref_worker_init
    _VERL_RAY_LOCAL_RANK_ENV_PATCHED = True
    logger.info("Patched Verl RayWorkerGroup to export local-rank env and set worker CUDA device")


def patch_verl_vllm_server_visible_devices() -> None:
    """Bind the vLLM server/engine process to the colocated rollout worker GPUs.

    Verl's async vLLM HTTP server actor is scheduled with node affinity only.
    On Ray versions that do not assign a GPU to that server actor, the vLLM
    EngineCore subprocess inherits visibility for the whole node and defaults
    to physical cuda:0.  With one colocated rollout worker per GPU, all engine
    cores then allocate their dummy model/KV cache on GPU 0 and fail before
    training starts.  The colocated workers already know their Ray-assigned
    visible device, so mirror that mapping into the server environment before
    vLLM is initialized.
    """
    global _VERL_VLLM_SERVER_VISIBLE_DEVICES_PATCHED
    if _VERL_VLLM_SERVER_VISIBLE_DEVICES_PATCHED:
        return

    from verl.workers.rollout.vllm_rollout import vllm_async_server as mod

    _original_launch_server = mod.vLLMHttpServerBase.launch_server

    async def _patched_launch_servers(self):
        import asyncio
        import os

        import ray

        assert len(self.workers) == self.world_size, (
            f"worker number {len(self.workers)} not equal to world size {self.world_size}"
        )

        worker_infos = await asyncio.gather(
            *[
                worker.__ray_call__.remote(
                    lambda self: (
                        ray.get_runtime_context().get_node_id(),
                        os.environ.get("CUDA_VISIBLE_DEVICES", "not set"),
                        os.environ.get("RLLM_NODE_LOCAL_RANK", os.environ.get("LOCAL_RANK", "0")),
                    )
                )
                for worker in self.workers
            ]
        )
        worker_node_ids = [node_id for node_id, _, _ in worker_infos]
        worker_visible_devices = []
        for _, visible_devices, node_local_rank in worker_infos:
            visible_list = [device.strip() for device in str(visible_devices).split(",") if device.strip()]
            if len(visible_list) > 1:
                try:
                    worker_visible_devices.append(visible_list[int(node_local_rank)])
                except (IndexError, TypeError, ValueError):
                    worker_visible_devices.append(str(node_local_rank))
            else:
                worker_visible_devices.append(visible_devices)

        nnodes, gpus_per_node = self.nnodes, self.gpus_per_node
        if self.config.data_parallel_size == 1:
            nnodes = 1
            gpus_per_node = self.world_size

        for node_rank in range(nnodes):
            start = node_rank * gpus_per_node
            stop = (node_rank + 1) * gpus_per_node
            workers = self.workers[start:stop]
            visible_values = worker_visible_devices[start:stop]
            node_id = worker_node_ids[start]
            name = (
                f"vllm_server_{self.replica_rank}_{node_rank}"
                if not self.is_reward_model
                else f"vllm_server_reward_{self.replica_rank}_{node_rank}"
            )

            ordered_devices = []
            for value in visible_values:
                for device in str(value).split(","):
                    device = device.strip()
                    if device and device != "not set" and device not in ordered_devices:
                        ordered_devices.append(device)
            env_vars = {}
            if ordered_devices:
                env_vars = {
                    "CUDA_VISIBLE_DEVICES": ",".join(ordered_devices),
                    "LOCAL_RANK": "0",
                    "LOCAL_WORLD_SIZE": str(len(ordered_devices)),
                }
                logger.info("Launching %s with CUDA_VISIBLE_DEVICES=%s", name, env_vars["CUDA_VISIBLE_DEVICES"])

            options = {
                "scheduling_strategy": ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_id,
                    soft=False,
                ),
                "name": name,
            }
            if env_vars:
                options["runtime_env"] = _vllm_server_runtime_env(env_vars)

            server = self.server_class.options(**options).remote(
                config=self.config,
                model_config=self.model_config,
                rollout_mode=self.rollout_mode,
                workers=workers,
                replica_rank=self.replica_rank,
                node_rank=node_rank,
                gpus_per_node=gpus_per_node,
                nnodes=nnodes,
            )
            self.servers.append(server)

        master_address, master_port = await self.servers[0].get_master_address.remote()
        await asyncio.gather(
            *[
                server.launch_server.remote(master_address=master_address, master_port=master_port)
                for server in self.servers
            ]
        )

        server_address, server_port = await self.servers[0].get_server_address.remote()
        self._server_handle = self.servers[0]
        self._server_address = (
            f"[{server_address}]:{server_port}"
            if mod.is_valid_ipv6_address(server_address)
            else f"{server_address}:{server_port}"
        )

    async def _patched_launch_server(self, *args, **kwargs):
        if getattr(self, "workers", None):
            import os

            import ray

            visible_devices = ray.get([worker.get_cuda_visible_devices.remote() for worker in self.workers])
            ordered_devices = []
            for value in visible_devices:
                for device in str(value).split(","):
                    device = device.strip()
                    if device and device != "not set" and device not in ordered_devices:
                        ordered_devices.append(device)
            if ordered_devices:
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(ordered_devices)
                os.environ["LOCAL_RANK"] = "0"
                os.environ["LOCAL_WORLD_SIZE"] = str(len(ordered_devices))
                logger.info("Bound vLLM server to CUDA_VISIBLE_DEVICES=%s", os.environ["CUDA_VISIBLE_DEVICES"])

        return await _original_launch_server(self, *args, **kwargs)

    mod.vLLMReplica.launch_servers = _patched_launch_servers
    mod.vLLMHttpServerBase.launch_server = _patched_launch_server
    _VERL_VLLM_SERVER_VISIBLE_DEVICES_PATCHED = True
    logger.info("Patched Verl vLLM server actors to inherit colocated worker CUDA_VISIBLE_DEVICES")


def patch_verl_vllm_rollout_local_rank() -> None:
    """Make colocated vLLM rollout workers use their assigned local CUDA rank.

    In environments where Ray leaves all node GPUs visible to each actor,
    verl's ``vLLMAsyncRollout._init_worker`` sets vLLM's ``local_rank`` to 0
    for every colocated rollout worker.  That makes all rollout engines place
    KV cache tensors on physical GPU 0 even though the actor worker itself has
    a distinct ``LOCAL_RANK``.  Preserve the local-rank env that
    ``patch_verl_ray_worker_local_rank_env`` exports before vLLM creates its
    ``WorkerWrapperBase``.
    """
    global _VERL_VLLM_ROLLOUT_LOCAL_RANK_PATCHED
    if _VERL_VLLM_ROLLOUT_LOCAL_RANK_PATCHED:
        return

    from verl.workers.rollout.vllm_rollout import vllm_rollout as mod

    _original_init_worker = mod.vLLMAsyncRollout._init_worker

    def _patched_init_worker(self, all_kwargs):
        import os

        from verl.utils.device import get_torch_device

        visible_devices = [
            device.strip()
            for device in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            if device.strip()
        ]
        node_local_rank = int(os.environ.get("RLLM_NODE_LOCAL_RANK", os.environ.get("LOCAL_RANK", "0")))
        local_rank = node_local_rank if len(visible_devices) != 1 else 0
        os.environ["LOCAL_RANK"] = str(local_rank)
        all_kwargs[0]["local_rank"] = local_rank
        get_torch_device().set_device(local_rank)

        old_noset = os.environ.get("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES")
        if len(visible_devices) != 1:
            os.environ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"
        try:
            return _original_init_worker(self, all_kwargs)
        finally:
            if len(visible_devices) != 1:
                if old_noset is None:
                    os.environ.pop("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = old_noset

    mod.vLLMAsyncRollout._init_worker = _patched_init_worker
    _VERL_VLLM_ROLLOUT_LOCAL_RANK_PATCHED = True
    logger.info("Patched Verl vLLM rollout workers to preserve local CUDA rank")


def patch_verl_vllm_async_server_ray_get() -> None:
    """Avoid blocking ray.get calls inside Verl's async vLLM server actor."""
    global _VERL_VLLM_ASYNC_SERVER_RAY_GET_PATCHED
    if _VERL_VLLM_ASYNC_SERVER_RAY_GET_PATCHED:
        return

    import inspect
    import textwrap

    from verl.workers.rollout.vllm_rollout import vllm_async_server as mod

    original = mod.vLLMHttpServerBase.launch_server
    src = inspect.getsource(original)
    old = "zmq_addresses = ray.get([worker.get_zeromq_address.remote() for worker in self.workers])"
    new = (
        "zmq_addresses = await asyncio.gather(\n"
        "            *[worker.get_zeromq_address.remote() for worker in self.workers]\n"
        "        )"
    )
    if old not in src:
        _VERL_VLLM_ASYNC_SERVER_RAY_GET_PATCHED = True
        logger.info("Verl vLLM async server ray.get patch: source already changed; nothing to patch.")
        return

    patched_src = textwrap.dedent(src).replace(old, new)
    namespace = dict(mod.__dict__)
    exec(compile(patched_src, mod.__file__, "exec"), namespace)
    mod.vLLMHttpServerBase.launch_server = namespace["launch_server"]

    _VERL_VLLM_ASYNC_SERVER_RAY_GET_PATCHED = True
    logger.info("Patched Verl vLLM async server to await worker ZMQ address refs without blocking ray.get")


def patch_vllm_worker_shared_lock_warning() -> None:
    """Avoid vLLM's shared_worker_lock warning when shm MM cache is not used."""
    global _VLLM_WORKER_SHARED_LOCK_WARNING_PATCHED
    if _VLLM_WORKER_SHARED_LOCK_WARNING_PATCHED:
        return

    import threading

    try:
        from vllm.v1.worker import worker_base as mod
    except ImportError:
        _VLLM_WORKER_SHARED_LOCK_WARNING_PATCHED = True
        return

    original = mod.WorkerWrapperBase.init_worker

    def _patched_init_worker(self, all_kwargs):
        kwargs = all_kwargs[self.rpc_rank]
        if "shared_worker_lock" not in kwargs:
            vllm_config = kwargs.get("vllm_config")
            mm_config = getattr(getattr(vllm_config, "model_config", None), "multimodal_config", None)
            cache_type = getattr(mm_config, "mm_processor_cache_type", None)
            if cache_type != "shm":
                kwargs = dict(kwargs)
                kwargs["shared_worker_lock"] = threading.Lock()
                all_kwargs = list(all_kwargs)
                all_kwargs[self.rpc_rank] = kwargs
        return original(self, all_kwargs)

    mod.WorkerWrapperBase.init_worker = _patched_init_worker
    _VLLM_WORKER_SHARED_LOCK_WARNING_PATCHED = True
    logger.info("Patched vLLM WorkerWrapperBase to avoid shared_worker_lock warning outside shm MM cache")


# ---------------------------------------------------------------------------
# Verl dynamic batch: sync micro-batch counts across DP ranks
# ---------------------------------------------------------------------------


def patch_verl_dynamic_batch_sync() -> None:
    """Patch ``prepare_dynamic_batch`` to sync micro-batch counts across DP ranks.

    Fixes `verl#5750 <https://github.com/verl-project/verl/issues/5750>`_:
    when ``use_dynamic_bsz=True``, each DP rank independently calculates
    ``num_micro_batches`` based on its local sequence lengths.  Different
    ranks can end up with different counts, causing NCCL collective
    operations (AllGather/ReduceScatter in FSDP) to deadlock.

    The fix defaults ``dp_group`` to ``torch.distributed.group.WORLD`` so
    that ``prepare_dynamic_batch`` performs an ``all_reduce(MAX)`` across
    ranks, forcing every rank to iterate through the same number of
    micro-batches.  This is the same approach as verl PR #5591.
    """
    global _VERL_DYNAMIC_BATCH_PATCHED
    if _VERL_DYNAMIC_BATCH_PATCHED:
        return

    import verl.utils.seqlen_balancing as sbl

    _original_prepare = sbl.prepare_dynamic_batch

    def _patched_prepare(data, max_token_len, dp_group=None, **kwargs):
        if dp_group is None:
            import torch.distributed

            if torch.distributed.is_initialized():
                dp_group = torch.distributed.group.WORLD
        return _original_prepare(data, max_token_len, dp_group=dp_group, **kwargs)

    sbl.prepare_dynamic_batch = _patched_prepare

    # Also patch the already-imported reference in dp_actor so both
    # compute_log_prob and update_policy use the patched version.
    try:
        from verl.workers.actor import dp_actor

        dp_actor.prepare_dynamic_batch = _patched_prepare
    except (ImportError, AttributeError):
        pass  # dp_actor may not be importable outside GPU workers

    _VERL_DYNAMIC_BATCH_PATCHED = True
    logger.info("Patched prepare_dynamic_batch to sync micro-batch counts across DP ranks (verl#5750)")


# ---------------------------------------------------------------------------
# Verl qwen3_vl: out-of-place add in dummy visual forward (backport PR #5881)
# ---------------------------------------------------------------------------


def patch_verl_qwen3_vl_dummy_inplace() -> None:
    """Backport `volcengine/verl#5881` for ``verl.models.transformers.qwen3_vl``.

    The dummy/no-image branch of ``_get_input_embeds`` mutates ``inputs_embeds``
    inplace::

        inputs_embeds += 0.0 * image_embeds.mean()
        for emb in dummy_deepstack_image_embeds or []:
            inputs_embeds += 0.0 * emb.mean()

    ``inputs_embeds`` is produced by ``model.get_input_embeddings()(input_ids)``
    and is a leaf with ``requires_grad=True`` after FSDP wrapping; the inplace
    ``+=`` raises a RuntimeError on the autograd backward and, in our 0.7.1
    pin, also surfaces as ``CUDNN_STATUS_NOT_INITIALIZED`` on the preceding
    ``model.visual(...)`` call due to the way the failure propagates inside
    Conv3d's cuDNN workspace setup. PR #5881 (merged main 2026-04-07; not in
    0.7.1) replaces both lines with out-of-place addition.

    We patch by re-exec'ing a fixed source of ``_get_input_embeds`` in the
    module's global namespace; ``qwen3_vl_base_forward`` looks up
    ``_get_input_embeds`` from module globals at call time, so the rebound
    function is picked up automatically.
    """
    global _VERL_QWEN3_VL_DUMMY_INPLACE_PATCHED
    if _VERL_QWEN3_VL_DUMMY_INPLACE_PATCHED:
        return

    import inspect

    from verl.models.transformers import qwen3_vl as mod

    src = inspect.getsource(mod._get_input_embeds)
    new_src = src.replace(
        "inputs_embeds += 0.0 * image_embeds.mean()",
        "inputs_embeds = inputs_embeds + 0.0 * image_embeds.mean()",
    ).replace(
        "inputs_embeds += 0.0 * emb.mean()",
        "inputs_embeds = inputs_embeds + 0.0 * emb.mean()",
    )

    if new_src == src:
        # Upstream already fixed (e.g. user bumped verl past 0.7.1).
        _VERL_QWEN3_VL_DUMMY_INPLACE_PATCHED = True
        logger.info("qwen3_vl PR #5881 patch: source already uses out-of-place add; nothing to patch.")
        return

    # Compile and exec into the module's globals so the rebound function shares
    # the module's namespace (torch, Optional, etc.) and so callers in the
    # same module (qwen3_vl_base_forward) pick up the patched version on the
    # next attribute lookup.
    exec(compile(new_src, mod.__file__, "exec"), mod.__dict__)

    _VERL_QWEN3_VL_DUMMY_INPLACE_PATCHED = True
    logger.info("Patched verl qwen3_vl._get_input_embeds: dummy visual path now uses out-of-place addition (backport of volcengine/verl#5881)")


# ---------------------------------------------------------------------------
# Verl tensordict utils: preserve _ragged_idx when rebuilding 3D NestedTensors
# (backport PR #6127)
# ---------------------------------------------------------------------------


def patch_verl_tensordict_jagged_layout() -> None:
    """Backport `volcengine/verl#6127` for ``verl.utils.tensordict_utils``.

    Fixes a bug in the rebuilding of 3D jagged NestedTensors after
    selection / chunking. ``torch.nested.as_nested_tensor(tensors,
    layout=torch.jagged)`` is ambiguous when all input tensors share the
    same last-dimension length — torch picks the wrong dimension as the
    jagged axis. For mRoPE ``position_ids`` with per-sample shape
    ``(num_heads, seq_len)``, this produces a rebuilt nested tensor whose
    ``_ragged_idx`` is 1 (heads dim) instead of 2 (seq dim), causing two
    downstream crashes:

    - ``index_select_tensor_dict`` → ``unbind()`` →
      ``torch.split(values, [batch_size], dim=ragged_idx-1)`` fails because
      ``values`` has total length = sum-of-row-lengths, not batch_size
      (hits the ``use_dynamic_bsz=True`` micro-batch partitioning path).
    - rmpad path's rotary cos/sin gets shape ``(B,)`` instead of ``(B, S)``
      and ``apply_rotary_pos_emb`` crashes with
      ``RuntimeError: The size of tensor a (<S>) must match the size of
      tensor b (<B>) at non-singleton dimension 2``.

    The fix introduces a ``nested_tensor_from_tensor_list(tensors,
    ragged_idx)`` helper that explicitly preserves the intended ragged
    dimension via ``torch.nested.nested_tensor_from_jagged`` plus an
    explicit ``_ragged_idx`` set, and uses it in ``concat_nested_tensors``,
    ``chunk_tensordict``, and ``index_select_tensor_dict``.

    Merged into verl main 2026-04-24; not in 0.7.1.
    """
    global _VERL_TENSORDICT_JAGGED_PATCHED
    if _VERL_TENSORDICT_JAGGED_PATCHED:
        return

    import torch
    from tensordict import TensorDict
    from verl.utils import tensordict_utils as tu

    if hasattr(tu, "nested_tensor_from_tensor_list"):
        # Upstream already provides the helper (e.g. user bumped verl past 0.7.1).
        _VERL_TENSORDICT_JAGGED_PATCHED = True
        logger.info("verl tensordict jagged-layout patch: upstream already provides nested_tensor_from_tensor_list; nothing to patch.")
        return

    def nested_tensor_from_tensor_list(tensors, ragged_idx=None):
        assert len(tensors) > 0, "Must provide at least one tensor"
        sample_dim = tensors[0].dim()
        if ragged_idx is None:
            ragged_idx = sample_dim
        assert ragged_idx == sample_dim, f"Only last-dimension ragged tensors are supported. Got {ragged_idx=} and {sample_dim=}"

        if sample_dim == 1:
            return torch.nested.as_nested_tensor(tensors, layout=torch.jagged)

        values = torch.cat(tensors, dim=-1)
        lengths = torch.tensor([t.shape[-1] for t in tensors], dtype=torch.long, device=values.device)
        offsets = torch.zeros(len(tensors) + 1, dtype=torch.long, device=values.device)
        torch.cumsum(lengths, dim=0, out=offsets[1:])

        nested_tensor = torch.nested.nested_tensor_from_jagged(values=values, offsets=offsets)
        nested_tensor._ragged_idx = ragged_idx
        return nested_tensor

    def concat_nested_tensors(tensors):
        for tensor in tensors:
            assert tensor.is_nested and tensor.is_contiguous()
        unbind_tensors = []
        for tensor in tensors:
            assert len(tensor.shape) >= 2, f"nested tensor must have 2 or more dimensions. Got {tensor.shape}"
            unbind_tensors.extend(list(tensor.unbind(0)))
        return nested_tensor_from_tensor_list(unbind_tensors, ragged_idx=tensors[0].dim() - 1)

    def chunk_tensordict(td, chunks):
        assert isinstance(td, TensorDict) and len(td) % chunks == 0, f"expecting td with length divisible by chunks, but got {len(td)} and {chunks}"
        chunk_size = len(td) // chunks
        nested_keys = {key for key, val in td.items() if isinstance(val, torch.Tensor) and val.is_nested}
        new_td = TensorDict({k: v for k, v in td.items() if k not in nested_keys}, batch_size=td.batch_size, device=td.device)
        tds = new_td.chunk(chunks=chunks)
        for key in nested_keys:
            nt = td[key]
            try:
                tensors = nt.unbind(dim=0)
            except RuntimeError:
                padded = nt.to_padded_tensor(0)
                padded_chunks = padded.chunk(chunks, dim=0)
                offsets = nt.offsets()
                lengths = offsets.diff().tolist()
                for i, chunk_td in enumerate(tds):
                    chunk_lengths = lengths[i * chunk_size : (i + 1) * chunk_size]
                    chunk_tensors = [padded_chunks[i][j, :seq_len] for j, seq_len in enumerate(chunk_lengths)]
                    chunk_td[key] = nested_tensor_from_tensor_list(chunk_tensors, ragged_idx=nt.dim() - 1)
                continue
            for i, chunk_td in enumerate(tds):
                chunk_td[key] = nested_tensor_from_tensor_list(list(tensors[i * chunk_size : (i + 1) * chunk_size]), ragged_idx=nt.dim() - 1)
        return tds

    def index_select_tensor_dict(batch, indices):
        if isinstance(indices, list):
            indices = torch.tensor(indices)
        assert indices.dim() == 1, "indices must be a 1D tensor"
        data_dict = {}
        batch_size = indices.shape[0]
        if batch is not None:
            for key, tensor in batch.items():
                if isinstance(tensor, torch.Tensor) and not tensor.is_nested:
                    data_dict[key] = tensor[indices]
                elif isinstance(tensor, torch.Tensor) and tensor.is_nested:
                    tensor_lst = tensor.unbind()
                    selected_tensors = [tensor_lst[idx] for idx in indices]
                    data_dict[key] = nested_tensor_from_tensor_list(selected_tensors, ragged_idx=tensor.dim() - 1)
                else:
                    if tensor.shape:
                        data_dict[key] = tensor[indices]
                    else:
                        data_dict[key] = tensor
            selected_batch = TensorDict(source=data_dict, batch_size=batch_size)
        else:
            selected_batch = None
        return selected_batch

    tu.nested_tensor_from_tensor_list = nested_tensor_from_tensor_list
    tu.concat_nested_tensors = concat_nested_tensors
    tu.chunk_tensordict = chunk_tensordict
    tu.index_select_tensor_dict = index_select_tensor_dict

    _VERL_TENSORDICT_JAGGED_PATCHED = True
    logger.info("Patched verl.utils.tensordict_utils: rebuild jagged NestedTensors with explicit _ragged_idx (backport of volcengine/verl#6127)")


# ---------------------------------------------------------------------------
# Verl vLLM async server: pass typed token inputs to vLLM 0.18+.
# ---------------------------------------------------------------------------


def patch_verl_vllm_typed_token_prompt() -> None:
    """Avoid vLLM 0.18+ raw-prompt deprecation warnings in verl's HTTP server.

    vLLM now expects the output of its renderer APIs, which are typed
    ``EngineInput`` dictionaries.  verl 0.7.x passes a legacy ``TokensPrompt``
    object when using token-in/token-out generation; vLLM still accepts it, but
    routes it through the deprecated raw-prompt branch and logs on every server
    process. For text-only prompts we can pass the equivalent typed token input
    directly. Multimodal requests stay on verl's original path because rendered
    multimodal inputs require placeholders and hashes that this older verl path
    does not construct.
    """
    global _VERL_VLLM_TYPED_TOKEN_PROMPT_PATCHED
    if _VERL_VLLM_TYPED_TOKEN_PROMPT_PATCHED:
        return

    try:
        from vllm.inputs.engine import tokens_input
        from verl.workers.rollout.vllm_rollout import vllm_async_server as mod
    except Exception:
        _VERL_VLLM_TYPED_TOKEN_PROMPT_PATCHED = True
        logger.info("verl vLLM typed-token prompt patch: required modules unavailable; nothing to patch.")
        return

    server_cls = getattr(mod, "vLLMHttpServerBase", None)
    if server_cls is None or not hasattr(server_cls, "generate"):
        _VERL_VLLM_TYPED_TOKEN_PROMPT_PATCHED = True
        logger.info("verl vLLM typed-token prompt patch: server class unavailable; nothing to patch.")
        return

    original_generate = server_cls.generate
    if getattr(original_generate, "_rllm_typed_token_prompt_patch", False):
        _VERL_VLLM_TYPED_TOKEN_PROMPT_PATCHED = True
        return

    async def _patched_generate(
        self,
        prompt_ids,
        sampling_params,
        request_id,
        image_data=None,
        video_data=None,
    ):
        if image_data is not None or video_data is not None:
            return await original_generate(
                self,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                request_id=request_id,
                image_data=image_data,
                video_data=video_data,
            )

        from typing import Optional

        from vllm.outputs import RequestOutput
        from vllm.sampling_params import SamplingParams
        from verl.workers.rollout.replica import TokenOutput

        max_possible_tokens = self.config.max_model_len - len(prompt_ids)
        if max_possible_tokens < 0:
            raise ValueError(
                f"Prompt length ({len(prompt_ids)}) exceeds the model's maximum context length "
                f"({self.config.max_model_len})."
            )

        sampling_params = sampling_params.copy()
        if "max_tokens" in sampling_params:
            max_tokens = sampling_params.pop("max_tokens")
        elif "max_new_tokens" in sampling_params:
            max_tokens = sampling_params.pop("max_new_tokens")
        else:
            max_tokens = self.config.response_length + self.config.prompt_length - len(prompt_ids)
        max_tokens = max(0, min(max_tokens, max_possible_tokens))

        sampling_params["logprobs"] = 0 if sampling_params.pop("logprobs", False) else None
        sampling_params.setdefault("repetition_penalty", self.config.get("repetition_penalty", 1.0))
        sampling_params = SamplingParams(max_tokens=max_tokens, **sampling_params)
        prompt_ids = mod._qwen2_5_vl_dedup_image_tokens(prompt_ids, self.model_config.processor)
        prompt = tokens_input(prompt_token_ids=prompt_ids)

        lora_request = None
        if self.model_config.lora_rank > 0:
            lora_loaded = mod.VLLM_LORA_INT_ID in await self.engine.list_loras()
            if lora_loaded:
                lora_request = mod.LoRARequest(
                    lora_name=mod.VLLM_LORA_NAME,
                    lora_int_id=mod.VLLM_LORA_INT_ID,
                    lora_path=mod.VLLM_LORA_PATH,
                )

        generator = self.engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
            lora_request=lora_request,
        )

        final_res: Optional[RequestOutput] = None
        async for output in generator:
            final_res = output
        assert final_res is not None

        token_ids = final_res.outputs[0].token_ids
        log_probs = None
        if sampling_params.logprobs is not None:
            log_probs = [logprobs[token_ids[i]].logprob for i, logprobs in enumerate(final_res.outputs[0].logprobs)]

        routed_experts = None
        if self.config.enable_rollout_routing_replay:
            routed_experts = final_res.outputs[0].routed_experts

        finish_reason = final_res.outputs[0].finish_reason
        if finish_reason == "abort":
            stop_reason = "aborted"
        elif finish_reason in ("stop", "length"):
            stop_reason = "completed"
        else:
            stop_reason = finish_reason

        return TokenOutput(
            token_ids=token_ids,
            log_probs=log_probs,
            routed_experts=routed_experts,
            stop_reason=stop_reason,
        )

    _patched_generate._rllm_typed_token_prompt_patch = True
    server_cls.generate = _patched_generate

    _VERL_VLLM_TYPED_TOKEN_PROMPT_PATCHED = True
    logger.info("Patched verl vLLM HTTP server to pass typed token inputs to vLLM for text-only prompts")


# ---------------------------------------------------------------------------
# Worker-side entry point (used as Ray runtime_env worker_process_setup_hook)
# ---------------------------------------------------------------------------

_ALL_VERL_PATCHES = {
    "patch_verl_ray_worker_local_rank_env": patch_verl_ray_worker_local_rank_env,
    "patch_verl_dynamic_batch_sync": patch_verl_dynamic_batch_sync,
    "patch_verl_qwen3_vl_dummy_inplace": patch_verl_qwen3_vl_dummy_inplace,
    "patch_verl_tensordict_jagged_layout": patch_verl_tensordict_jagged_layout,
    "patch_verl_vllm_typed_token_prompt": patch_verl_vllm_typed_token_prompt,
    "patch_verl_vllm_async_server_ray_get": patch_verl_vllm_async_server_ray_get,
    "patch_vllm_worker_shared_lock_warning": patch_vllm_worker_shared_lock_warning,
    "patch_verl_vllm_server_visible_devices": patch_verl_vllm_server_visible_devices,
    "patch_verl_vllm_rollout_local_rank": patch_verl_vllm_rollout_local_rank,
}


def apply_all_verl_patches() -> None:
    """Apply every Verl patch that is safe to run unconditionally on workers.

    Designed to be wired in as ``runtime_env.worker_process_setup_hook =
    "rllm.experimental.verl.patch.apply_all_verl_patches"`` so that each Ray
    worker process applies the patches in its own interpreter (driver-side
    monkey-patches do not propagate to worker processes).

    Each patch below is lazy and idempotent, so it is safe to call this
    repeatedly and from any process.

    Optional extension hook: if the ``RLLM_EXTRA_WORKER_SETUP_HOOK``
    environment variable is set, this function will additionally invoke the
    callable it names. The value is ``"<absolute-file-path.py>:<func>"``;
    the function is loaded directly from the file via ``importlib.util`` so
    it does not need to live in a package on ``sys.path``. This is intended
    for environment-specific workarounds (e.g. disabling cuDNN on a host
    with a broken cuDNN install) that should not be baked into rLLM itself.
    """
    for patch_name, patch_func in _ALL_VERL_PATCHES.items():
        try:
            patch_func()
        except Exception:  # pragma: no cover — patch is best-effort
            logger.exception(f"{patch_name} failed in worker setup hook")

    _run_extra_worker_setup_hook()


def _run_extra_worker_setup_hook() -> None:
    """Optionally invoke a user-supplied setup hook from RLLM_EXTRA_WORKER_SETUP_HOOK.

    Format: ``"<absolute path to .py file>:<function name>"``. The function
    is loaded via ``importlib.util.spec_from_file_location`` so it works
    even when the file is not on ``sys.path`` (e.g. lives under a
    gitignored ``tmp/`` directory). Failures are logged but never raised —
    this is a best-effort extension point and must not crash worker init.
    """
    import os

    spec_str = os.environ.get("RLLM_EXTRA_WORKER_SETUP_HOOK")
    if not spec_str:
        return

    path, _, func_name = spec_str.rpartition(":")
    if not path or not func_name:
        logger.warning(
            "RLLM_EXTRA_WORKER_SETUP_HOOK=%r is malformed; expected '<file.py>:<func>'",
            spec_str,
        )
        return

    try:
        import importlib.util

        mod_name = f"_rllm_extra_setup_hook_{os.getpid()}"
        spec = importlib.util.spec_from_file_location(mod_name, path)
        if spec is None or spec.loader is None:
            logger.warning("RLLM_EXTRA_WORKER_SETUP_HOOK: could not load spec for %s", path)
            return
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        getattr(mod, func_name)()
    except Exception:
        logger.exception("RLLM_EXTRA_WORKER_SETUP_HOOK=%r failed", spec_str)
