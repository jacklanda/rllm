"""Runner for agentic evaluation.

Drives rllm's ToolAgent + ToolEnvironment + MultiTurnWorkflow through
AgentWorkflowEngine using an in-process vLLM engine by default. Set
``backend="ray_sglang"`` to use a data-parallel SGLang pool, or
``backend="openai"`` to fall back to talking to an external
OpenAI-compatible server.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


try:
    from rich.console import Console
    from rich.table import Table
except ModuleNotFoundError:  # pragma: no cover - runtime fallback for base images.
    class Console:
        def log(self, message: str) -> None:
            print(message)

        def rule(self, title: str) -> None:
            print(f"\n=== {title} ===")

        def print(self, value) -> None:
            print(value)

    class Table:
        def __init__(self, title: str = "", show_lines: bool = False) -> None:
            del show_lines
            self.title = title
            self.columns: list[str] = []
            self.rows: list[tuple[str, ...]] = []

        def add_column(self, name: str, **_: Any) -> None:
            self.columns.append(name)

        def add_row(self, *values: object) -> None:
            self.rows.append(tuple(str(v) for v in values))

        def __str__(self) -> str:
            lines = [self.title] if self.title else []
            if self.columns:
                lines.append(" | ".join(self.columns))
            lines.extend(" | ".join(row) for row in self.rows)
            return "\n".join(lines)

import rllm.engine.agent_workflow_engine as _awe_mod
from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
from rllm.agents.system_prompts import SEARCH_SYSTEM_PROMPT

from evals.benchmark_loader import AgenticTask, load_benchmark
from evals.reward import build_dispatching_reward_fn, build_reward_fn
from evals.rllm_compat import EvalsAgent, EvalsEnvironment
from evals.workflow import EvalsWorkflow

console = Console()

# Suppress the per-rollout "Rollout completed. Rewards: ..." log from rllm.
_orig_colorful_print = _awe_mod.colorful_print


def _filtered_colorful_print(msg, **kwargs):
    if "Rollout completed." not in str(msg):
        _orig_colorful_print(msg, **kwargs)


_awe_mod.colorful_print = _filtered_colorful_print

# Fixed namespaces so problem_id / trajectory_id are stable across runs.
_PROBLEM_NS = uuid.UUID("6f3b8b1e-2a2a-4a1e-9b4a-1f5f1e3d7c01")
_TRAJECTORY_NS = uuid.UUID("6f3b8b1e-2a2a-4a1e-9b4a-1f5f1e3d7c02")


def _problem_uuid(data_source: str, task_id: str, question: str) -> str:
    return str(uuid.uuid5(_PROBLEM_NS, f"{data_source}\x1f{task_id}\x1f{question}"))


def _trajectory_uuid(problem_id: str, episode) -> str:
    """Derive a uuid5 from the full trajectory content.

    Identical trajectories (same model turns + tool outputs) map to the same
    id; any divergence in steps, tool results, or termination state produces a
    different id.
    """
    parts: list[str] = [problem_id]
    for traj in getattr(episode, "trajectories", []) or []:
        for step in getattr(traj, "steps", []) or []:
            parts.append(str(getattr(step, "model_response", "") or ""))
            parts.append(str(getattr(step, "action", "") or ""))
            parts.append(str(getattr(step, "observation", "") or ""))
    term = getattr(episode, "termination_reason", None)
    parts.append(getattr(term, "value", "") if term is not None else "")
    return str(uuid.uuid5(_TRAJECTORY_NS, "\x1e".join(parts)))


@dataclass
class AgenticEvalConfig:
    """Configuration for a single agentic-evaluation run.

    Model / backend
    ---------------
    The default backend (``vllm``) loads the model in-process via
    ``vllm.AsyncLLMEngine`` — no HTTP server required. Set
    ``backend="ray_vllm"`` to fan the workload across ``ray_num_replicas``
    Ray actors (each owning its own ``VLLMEngine`` on ``tensor_parallel_size``
    GPUs). Set ``backend="ray_sglang"`` to run ``ray_num_replicas`` SGLang
    worker processes. Set ``backend="openai"`` to talk to an external
    OpenAI-compatible server (the previous behavior); in that case ``model``
    is the name the server exposes and ``base_url``/``api_key`` point at it.
    """

    model: str
    backend: str = "vllm"
    harness: str = ""
    base_url: str = ""
    api_key: str = ""

    # Ray-distributed vLLM (only used when backend == "ray_vllm")
    ray_num_replicas: int = 1
    ray_address: str = ""

    # Benchmarks
    tasks: list[str] = field(
        default_factory=lambda: ["bamboogle", "hotpotqa", "2wiki", "musique"]
    )
    artifacts_dir: str = ""
    max_problems: int | None = None
    shuffle: bool = True
    shuffle_seed: int = 0

    # Agent / rollout
    tools: list[str] = field(default_factory=lambda: ["web_search"])
    parser_name: str = "qwen"
    system_prompt: str = SEARCH_SYSTEM_PROMPT
    user_prompt_template: str = "{problem_statement}"
    is_no_think_prompt: bool = False
    no_system_prompt: bool = False
    max_steps: int = 10
    max_response_length: int = 8192
    max_prompt_length: int = 32768
    max_new_tokens: int = 2048

    # Local retrieval server (matches rllm/experiments/fused/train_fused_agent.sh)
    retrieval_server_url: str = ""
    retrieval_max_results: int = 10
    summarize: int = 0
    retrieval_timeout: float = 3600.0

    # Sampling
    num_samples: int = 1
    passk: int = 1
    temperature: float = 0.6
    top_p: float = 0.95

    # Concurrency
    n_parallel_tasks: int = 32
    retry_limit: int = 2

    # Mixed-evals: pool rollouts from every benchmark in ``tasks`` into a
    # single ``execute_tasks`` call so slow tails on one benchmark overlap
    # with fast tails on another, keeping all vLLM replicas saturated
    # instead of idling between benchmarks. Scoring and per-benchmark
    # output paths are unchanged.
    mixed_evals: bool = False

    # vLLM backend (only used when backend == "vllm")
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    vllm_max_model_len: int | None = None
    vllm_dtype: str = "auto"
    vllm_enforce_eager: bool = False
    vllm_trust_remote_code: bool = True
    enable_thinking: bool = False

    # Output
    output_dir: str = "output_agentic"
    save_alias: str = ""
    overwrite: bool = False

    def resolved_base_url(self) -> str:
        return (
            self.base_url
            or os.environ.get("OPENAI_BASE_URL")
            or "http://localhost:30000/v1"
        )

    def resolved_api_key(self) -> str:
        return self.api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"


def estimate_pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    if num_samples < k:
        return 1.0 if num_correct == num_samples else 0.0
    return 1.0 - math.comb(num_samples - num_correct, k) / math.comb(num_samples, k)


def estimate_pass_hat_k(num_samples: int, num_correct: int, k: int) -> float:
    """Pass^k: probability that *all* k independently drawn samples are correct.

    C(c, k) / C(n, k). Returns 0.0 when k > num_correct (impossible).
    """
    if num_samples < k:
        return 1.0 if num_correct == num_samples else 0.0
    if num_correct < k:
        return 0.0
    return math.comb(num_correct, k) / math.comb(num_samples, k)


def _extract_final_answer(
    episode, ground_truth=None, data_source=None, question=None
) -> tuple[str, bool, dict]:
    """Pull the extracted answer, correctness flag, and reward metadata.

    The last step of the agent trajectory carries an ``info`` dict populated
    by ``ToolEnvironment.step`` which includes ``is_correct`` and
    ``metadata`` from the reward function (with ``extracted_answer``).

    We additionally re-apply the boxed-first extractor at save time: the
    reward path can return an empty or raw-response ``extracted_answer`` if
    the env's reward_fn was mis-wired or if the model emitted the boxed
    span without a ``finish`` tool call. Re-extracting here guarantees the
    JSON record carries the short predicted string (``\\boxed{...}``
    contents) rather than the full trajectory text.

    When ``ground_truth`` is provided, we re-score the extracted answer to
    fix stale correctness flags from rollout-time scoring of raw responses.
    """
    # Local import avoids cycles with reward selection helpers.
    from evals.reward import _BoxedFirstSearchFn, build_reward_fn

    extracted = ""
    is_correct = False
    metadata: dict[str, Any] = {}
    if episode.trajectories and episode.trajectories[0].steps:
        last = episode.trajectories[0].steps[-1]
        raw_response = str(last.model_response or "")
        is_correct = bool(last.info.get("is_correct", False))
        metadata = last.info.get("metadata") or {}
        extracted = str(metadata.get("extracted_answer", ""))

        # If metadata gave us nothing, fall back to the model's final response.
        if not extracted and raw_response:
            extracted = raw_response

        # Re-unbox if the stored value is clearly a raw response (contains a
        # boxed anchor or is long enough to be a trajectory). This keeps the
        # saved ``extracted_answer`` field holding just the predicted string.
        #if data_source != "gpqa_diamond" and extracted and (
        #    "\\boxed" in extracted or "boxed{" in extracted or len(extracted) > 200
        #):
        if extracted and (
            "\\boxed" in extracted or "boxed{" in extracted or len(extracted) > 200
        ):
            try:
                _fn = _BoxedFirstSearchFn.__new__(_BoxedFirstSearchFn)
                reextracted = _fn.extract_answer_from_response(extracted)
                if reextracted and reextracted != extracted:
                    extracted = reextracted
            except Exception:
                pass

        # Re-score with the benchmark-specific reward. This is important for
        # MCQ datasets such as GPQA Diamond: save-time extraction must not fall
        # back to the generic open-ended EM/F1 grader after rollout used a
        # stricter choice scorer.
        if ground_truth is not None and (extracted or raw_response):
            fn = build_reward_fn(data_source or "")
            task_info = {
                "ground_truth": ground_truth,
                "data_source": data_source or "",
            }
            if question is not None:
                # MCQ datasets whose GT is option prose (e.g. ``medqa``) need
                # the question text so ``evaluate_answer`` can map a bare
                # letter like ``"D"`` back to the D-option prose. Without it
                # the reward sees ``"D"`` vs prose GT and scores 0.
                task_info["question"] = question
            #action_for_rescore = (
            #    raw_response
            #    if data_source == "gpqa_diamond" and raw_response
            #    else extracted
            #)
            action_for_rescore = extracted
            result = fn(task_info, action_for_rescore)
            is_correct = result.is_correct
            metadata.update(result.metadata)
            extracted = str(metadata.get("extracted_answer", extracted))

    return extracted, is_correct, metadata


def _json_safe(value: Any) -> Any:
    """Recursively coerce a value to JSON-serializable primitives."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _extract_trajectory_messages(episode) -> list[dict[str, Any]]:
    """Return the full chat transcript (system/user/assistant/tool turns).

    ``ToolAgent`` stores the cumulative message list on every ``Step`` via
    ``chat_completions``; the last step holds the complete conversation
    including the agent's final assistant turn. We return that list
    directly (JSON-sanitized) so consumers can replay every role/content
    pair — plus tool-call metadata (``tool_calls`` on assistant turns,
    ``tool_call_id`` on tool turns) — exactly as the model saw it.
    """
    if not episode.trajectories:
        return []
    steps = episode.trajectories[0].steps
    if not steps:
        return []
    messages = getattr(steps[-1], "chat_completions", None) or []
    return _json_safe(messages)


def _build_rollout_engine(cfg: AgenticEvalConfig):
    """Instantiate the rollout engine once for the whole evaluation run.

    Loading a multi-billion parameter model per benchmark is wasteful, so
    the engine is constructed once and reused for every benchmark-specific
    ``AgentWorkflowEngine`` below.
    """
    backend = (cfg.backend or "vllm").lower()
    if backend == "vllm":
        from evals.vllm_engine import VLLMEngine

        console.log(
            f"[cyan]Initializing in-process vLLM engine for {cfg.model} "
            f"(tp={cfg.tensor_parallel_size}, mem={cfg.gpu_memory_utilization})"
        )
        return VLLMEngine(
            model=cfg.model,
            max_prompt_length=cfg.max_prompt_length,
            max_response_length=cfg.max_response_length,
            max_model_length=cfg.vllm_max_model_len,
            sampling_params={
                "temperature": cfg.temperature,
                "top_p": cfg.top_p,
                "max_tokens": cfg.max_new_tokens,
            },
            tensor_parallel_size=cfg.tensor_parallel_size,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            dtype=cfg.vllm_dtype,
            enforce_eager=cfg.vllm_enforce_eager,
            trust_remote_code=cfg.vllm_trust_remote_code,
            disable_thinking=not cfg.enable_thinking,
            no_system_prompt=cfg.no_system_prompt,
        )
    if backend == "ray_vllm":
        from evals.ray_vllm_engine import RayVLLMEngine

        console.log(
            f"[cyan]Initializing Ray-distributed vLLM pool for {cfg.model} "
            f"(replicas={cfg.ray_num_replicas}, tp={cfg.tensor_parallel_size}, "
            f"mem={cfg.gpu_memory_utilization})"
        )
        return RayVLLMEngine(
            model=cfg.model,
            num_replicas=cfg.ray_num_replicas,
            tensor_parallel_size=cfg.tensor_parallel_size,
            ray_address=cfg.ray_address or None,
            max_prompt_length=cfg.max_prompt_length,
            max_response_length=cfg.max_response_length,
            max_model_length=cfg.vllm_max_model_len,
            sampling_params={
                "temperature": cfg.temperature,
                "top_p": cfg.top_p,
                "max_tokens": cfg.max_new_tokens,
            },
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            dtype=cfg.vllm_dtype,
            enforce_eager=cfg.vllm_enforce_eager,
            trust_remote_code=cfg.vllm_trust_remote_code,
            disable_thinking=not cfg.enable_thinking,
            no_system_prompt=cfg.no_system_prompt,
        )
    if backend == "ray_sglang":
        from evals.sglang_dp_engine import SGLangDataParallelEngine

        console.log(
            f"[cyan]Initializing data-parallel SGLang pool for {cfg.model} "
            f"(replicas={cfg.ray_num_replicas}, tp_per_worker={cfg.tensor_parallel_size}, "
            f"mem={cfg.gpu_memory_utilization})"
        )
        return SGLangDataParallelEngine(
            model=cfg.model,
            data_parallel_size=cfg.ray_num_replicas,
            max_prompt_length=cfg.max_prompt_length,
            max_response_length=cfg.max_response_length,
            max_model_length=cfg.vllm_max_model_len,
            sampling_params={
                "temperature": cfg.temperature,
                "top_p": cfg.top_p,
                "max_tokens": cfg.max_new_tokens,
            },
            tensor_parallel_size=cfg.tensor_parallel_size,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            trust_remote_code=cfg.vllm_trust_remote_code,
            disable_thinking=not cfg.enable_thinking,
            no_system_prompt=cfg.no_system_prompt,
        )
    if backend == "openai":
        from rllm.engine.rollout.openai_engine import OpenAIEngine

        console.log(f"[cyan]Using OpenAI-compatible endpoint {cfg.resolved_base_url()}")
        return OpenAIEngine(
            model=cfg.model,
            tokenizer=None,
            base_url=cfg.resolved_base_url(),
            api_key=cfg.resolved_api_key(),
            max_prompt_length=cfg.max_prompt_length,
            max_response_length=cfg.max_response_length,
            sampling_params={
                "temperature": cfg.temperature,
                "top_p": cfg.top_p,
                "max_tokens": cfg.max_new_tokens,
            },
        )
    raise ValueError(
        f"Unknown backend: {cfg.backend!r} "
        "(expected 'vllm', 'ray_vllm', 'ray_sglang', or 'openai')"
    )


def _resolve_tools(cfg: AgenticEvalConfig) -> dict[str, Any]:
    """Translate ``cfg.tools`` into kwargs for ``ToolAgent`` / ``ToolEnvironment``.

    Tools registered in ``rllm.tools.tool_registry`` are passed through as the
    legacy ``tools=[...]`` list. ``web_search`` is not in the default registry,
    so when it's requested we switch to a ``tool_map={name: cls}`` construction
    that points at ``evals.web_search_tool.WebRetrievalTool``.

    Mixing registered tools with ``web_search`` in the same run isn't supported
    by rllm's ``MultiTool`` (which rejects both arguments together), so this
    raises if the user asks for that combination.
    """
    names = list(cfg.tools)
    wants_local = "web_search" in names

    if not wants_local:
        return {"tools": names}

    if len(names) > 1 and (len(names) != 2 or "finish" not in names):
        raise ValueError(
            "'web_search' cannot be combined with other tools in the same run "
            "because rllm's MultiTool accepts either 'tools' or 'tool_map', not both. "
            f"Got tools={names}."
        )

    from evals.web_search_tool import WebRetrievalTool
    from evals.finish_tool import FinishTool
    server_url = (
        cfg.retrieval_server_url
        or os.environ.get("RETRIEVAL_SERVER_URL")
        or "http://127.0.0.1:8000"
    )
    os.environ.setdefault("RETRIEVAL_SERVER_URL", server_url)
    max_results = cfg.retrieval_max_results
    timeout = cfg.retrieval_timeout
    summarize = cfg.summarize
    class _LocalSearch(WebRetrievalTool):
        def __init__(self, name: str = "web_search", description: str | None = None):
            super().__init__(
                name=name,
                description=description or WebRetrievalTool.DESCRIPTION,
                server_url=server_url,
                timeout=timeout,
                max_results=max_results,
            )

    class _FinishTool(FinishTool):
        def __init__(self, name: str = "finish", description: str | None = None):
            super().__init__(
                name=name,
                description=description or FinishTool.DESCRIPTION,
            )


    console.log(
        f"[cyan]Using WebRetrievalTool against {server_url} "
        f"(top_k={max_results}, summarize={os.environ.get('RLLM_RETRIEVAL_SUMMARIZE', '0') == '1'})"
    )
    return {"tool_map": {"web_search": _LocalSearch, "finish": _FinishTool}}


def _build_workflow_engine(
    cfg: AgenticEvalConfig,
    data_source: str,
    rollout_engine,
    reward_fn=None,
) -> AgentWorkflowEngine:
    """Wrap ``rollout_engine`` in an ``AgentWorkflowEngine`` for one benchmark.

    Each benchmark gets its own workflow pool because ``reward_fn`` bakes in
    the benchmark-specific ``data_source``, but they all share the single
    loaded rollout engine passed in.

    In mixed-evals mode callers pass a dispatching ``reward_fn`` instead so a
    single workflow pool can serve rollouts from multiple data sources.
    """
    if reward_fn is None:
        reward_fn = build_reward_fn(data_source)
    tool_kwargs = _resolve_tools(cfg)

    workflow_args = {
        "agent_cls": EvalsAgent,
        "env_cls": EvalsEnvironment,
        "agent_args": {
            "system_prompt": cfg.system_prompt,
            "user_prompt_template": cfg.user_prompt_template,
            "parser_name": cfg.parser_name,
            "model": cfg.model,
            "is_no_think_prompt": cfg.is_no_think_prompt,
            **tool_kwargs,
        },
        "env_args": {
            **tool_kwargs,
            "reward_fn": reward_fn,
            "max_steps": cfg.max_steps,
        },
        "max_steps": cfg.max_steps,
    }

    return AgentWorkflowEngine(
        workflow_cls=EvalsWorkflow,
        workflow_args=workflow_args,
        rollout_engine=rollout_engine,
        config=None,
        n_parallel_tasks=cfg.n_parallel_tasks,
        retry_limit=cfg.retry_limit,
        raise_on_error=False,
    )


async def _evaluate_one_benchmark(
    cfg: AgenticEvalConfig,
    bench: str,
    tasks: list[AgenticTask],
    rollout_engine,
) -> dict[str, Any]:
    console.rule(
        f"[bold cyan]{bench.strip()} (n={len(tasks)}, samples={cfg.num_samples})"
    )
    data_source = tasks[0].data_source if tasks else bench

    engine = _build_workflow_engine(
        cfg, data_source=data_source, rollout_engine=rollout_engine
    )

    # Replicate each task num_samples times; carry identical task_ids to group.
    expanded_tasks: list[dict] = []
    expanded_ids: list[str] = []
    for t in tasks:
        env_task = t.to_env_task()
        for _ in range(cfg.num_samples):
            expanded_tasks.append(env_task)
            expanded_ids.append(t.task_id)

    t0 = time.time()
    episodes = await engine.execute_tasks(expanded_tasks, task_ids=expanded_ids)
    elapsed = time.time() - t0

    # Group episodes back by task_id in input order.
    per_task: dict[str, list] = {}
    for tid, ep in zip(expanded_ids, episodes):
        per_task.setdefault(tid, []).append(ep)

    return _score_benchmark_results(cfg, bench, tasks, per_task, elapsed)


def _score_benchmark_results(
    cfg: AgenticEvalConfig,
    bench: str,
    tasks: list[AgenticTask],
    per_task: dict[str, list],
    elapsed: float,
) -> dict[str, Any]:
    """Aggregate per-task episodes into the per-benchmark payload.

    Shared by the sequential and mixed-evals paths — the only difference is
    whether ``per_task`` came from one ``execute_tasks`` call covering this
    benchmark only, or from the slice of a pooled mixed-evals run.
    """
    data_source = tasks[0].data_source if tasks else bench

    records: list[dict[str, Any]] = []
    total_correct = 0
    total_samples = 0
    passk_hits = 0
    per_task_accuracy: list[float] = []
    per_task_pass1: list[float] = []
    per_task_passn: list[float] = []
    passk_ks = sorted(
        {k for k in [1, 2, 4, 8, cfg.num_samples] if 1 <= k <= cfg.num_samples}
    )
    per_task_passk_vals: dict[int, list[float]] = {k: [] for k in passk_ks}
    passhatk_ks = sorted(set(passk_ks) | {cfg.passk})
    per_task_passhatk_vals: dict[int, list[float]] = {k: [] for k in passhatk_ks}
    # is_correct[task_idx][sample_idx]. A "run" = one sample index across
    # every task; std across runs is the std of per-run aggregate metrics
    # (matches parallel_reasoner/eval/evaluate.py inter-run std semantics).
    per_task_sample_correct: list[list[int]] = []

    for t in tasks:
        eps = per_task.get(t.task_id, [])
        problem_id = _problem_uuid(t.data_source, t.task_id, t.question)
        samples: list[dict[str, Any]] = []
        num_correct = 0
        sample_correct: list[int] = []
        gt = t.ground_truth[0] if len(t.ground_truth) == 1 else t.ground_truth
        for ep in eps:
            extracted, is_correct, meta = _extract_final_answer(
                ep, ground_truth=gt, data_source=t.data_source, question=t.question
            )
            num_correct += int(is_correct)
            sample_correct.append(int(bool(is_correct)))
            samples.append(
                {
                    "trajectory_id": _trajectory_uuid(problem_id, ep),
                    "extracted_answer": extracted,
                    "is_correct": is_correct,
                    "reward": float(getattr(ep.trajectories[0], "reward", 0.0))
                    if ep.trajectories
                    else 0.0,
                    "termination_reason": ep.termination_reason.value
                    if ep.termination_reason
                    else None,
                    "f1_score": meta.get("f1_score"),
                    "exact_match": meta.get("exact_match"),
                    "num_steps": len(ep.trajectories[0].steps)
                    if ep.trajectories
                    else 0,
                    "trajectory": _extract_trajectory_messages(ep),
                }
            )
        total_correct += num_correct
        total_samples += len(eps)
        per_task_sample_correct.append(sample_correct)
        pass1 = estimate_pass_at_k(len(eps), num_correct, 1) if eps else 0.0
        passk_val = estimate_pass_at_k(len(eps), num_correct, cfg.passk) if eps else 0.0
        passk_hits += int(passk_val > 0.5)
        per_task_accuracy.append(num_correct / len(eps) if eps else 0.0)
        per_task_pass1.append(pass1)
        passn_val = estimate_pass_at_k(len(eps), num_correct, len(eps)) if eps else 0.0
        per_task_passn.append(passn_val)
        for k in passk_ks:
            per_task_passk_vals[k].append(
                estimate_pass_at_k(len(eps), num_correct, k) if eps else 0.0
            )
        passhatk_val = (
            estimate_pass_hat_k(len(eps), num_correct, cfg.passk) if eps else 0.0
        )
        for k in passhatk_ks:
            per_task_passhatk_vals[k].append(
                estimate_pass_hat_k(len(eps), num_correct, k) if eps else 0.0
            )

        # Log per-problem metrics
        if cfg.passk == 1:
            console.log(
                f"[dim]{t.task_id}[/dim] Pass@1={pass1:.2f} ({num_correct}/{len(eps)})"
            )
        else:
            console.log(
                f"[dim]{t.task_id}[/dim] Pass@1={pass1:.2f} "
                f"Pass@{cfg.passk}={passk_val:.2f} "
                f"Pass^{cfg.passk}={passhatk_val:.2f} "
                f"({num_correct}/{len(eps)})"
            )

        records.append(
            {
                "problem_id": problem_id,
                "task_id": t.task_id,
                "data_source": t.data_source,
                "question": t.question,
                "ground_truth": t.ground_truth,
                "num_samples": len(eps),
                "num_correct": num_correct,
                f"pass@{cfg.passk}": passk_val,
                f"pass^{cfg.passk}": passhatk_val,
                "samples": samples,
            }
        )

    accuracy = (total_correct / total_samples) if total_samples else 0.0
    passk_acc = (passk_hits / len(tasks)) if tasks else 0.0
    n = len(per_task_accuracy)

    def _mean_std(vals: list[float]) -> tuple[float, float]:
        mean = sum(vals) / len(vals) if vals else 0.0
        std = (
            math.sqrt(sum((x - mean) ** 2 for x in vals) / len(vals))
            if len(vals) > 1
            else 0.0
        )
        return mean, std

    def _per_run_accuracy(run_idx: int) -> float:
        """Mean over tasks of ``is_correct`` at sample index ``run_idx``."""
        hits = 0
        denom = 0
        for sc in per_task_sample_correct:
            if run_idx < len(sc):
                hits += sc[run_idx]
                denom += 1
        return (hits / denom) if denom else 0.0

    def _per_run_passk(k: int) -> list[float]:
        """Partition each task's samples into ``num_samples // k`` disjoint
        k-slices; a run's pass@k = mean over tasks of "any hit in that slice"."""
        if k < 1 or cfg.num_samples < k:
            return []
        n_runs = cfg.num_samples // k
        runs: list[float] = []
        for r in range(n_runs):
            hit_sum = 0
            denom = 0
            for sc in per_task_sample_correct:
                if len(sc) < (r + 1) * k:
                    continue
                window = sc[r * k : (r + 1) * k]
                hit_sum += 1 if any(window) else 0
                denom += 1
            runs.append((hit_sum / denom) if denom else 0.0)
        return runs

    # Std across runs (per-sample-index aggregates), not across tasks.
    per_run_acc = [_per_run_accuracy(i) for i in range(cfg.num_samples)]
    accuracy_std = _mean_std(per_run_acc)[1]
    pass1_std = _mean_std(_per_run_passk(cfg.passk))[1]
    summary = {
        "benchmark": bench,
        "data_source": data_source,
        "num_tasks": len(tasks),
        "num_samples_per_task": cfg.num_samples,
        "accuracy_mean": accuracy,
        "accuracy_std": accuracy_std,
        f"pass@{cfg.passk}_threshold_{0.5}": passk_acc,
        f"pass@{cfg.passk}_std_threshold_{0.5}": pass1_std,
        "elapsed_seconds": elapsed,
        "model": cfg.model,
        "tools": list(cfg.tools),
        "max_steps": cfg.max_steps,
    }
    for k, vals in per_task_passk_vals.items():
        summary[f"pass@{k}"] = sum(vals) / len(vals) if vals else 0.0
        summary[f"pass@{k}_std"] = _mean_std(_per_run_passk(k))[1]
    for k, vals in per_task_passhatk_vals.items():
        mean, std = _mean_std(vals)
        summary[f"pass^{k}"] = mean
        summary[f"pass^{k}_std"] = std

    return {"summary": summary, "records": records}


async def _evaluate_benchmarks_mixed(
    cfg: AgenticEvalConfig,
    bench_tasks: list[tuple[str, list[AgenticTask]]],
    rollout_engine,
) -> list[tuple[str, dict[str, Any]]]:
    """Run every (bench, tasks) pair through one shared workflow engine.

    Every benchmark's rollouts share a single ``execute_tasks`` call so a
    long tail on one benchmark overlaps with fast-finishing rollouts on
    another, keeping the rollout pool saturated instead of draining
    between benchmarks. After the merged run, episodes are split back per
    benchmark and scored exactly as in the sequential path.
    """
    total_tasks = sum(len(ts) for _, ts in bench_tasks)
    bench_names = ", ".join(b for b, _ in bench_tasks)
    console.rule(
        f"[bold cyan]MIXED  [{bench_names}]  (n={total_tasks}, samples={cfg.num_samples})"
    )

    # Single workflow pool with a dispatching reward fn that resolves the
    # right grader from each rollout's ``data_source``.
    engine = _build_workflow_engine(
        cfg,
        data_source="__mixed__",
        rollout_engine=rollout_engine,
        reward_fn=build_dispatching_reward_fn(),
    )

    # Bench-prefixed engine task_ids so identical task_ids across
    # benchmarks (e.g. ``hotpotqa-0`` and ``2wiki-0``) don't collide.
    sep = "\x1f"
    expanded_tasks: list[dict] = []
    expanded_ids: list[str] = []
    for bench, tasks in bench_tasks:
        for t in tasks:
            env_task = t.to_env_task()
            mixed_id = f"{bench}{sep}{t.task_id}"
            for _ in range(cfg.num_samples):
                expanded_tasks.append(env_task)
                expanded_ids.append(mixed_id)

    t0 = time.time()
    episodes = await engine.execute_tasks(expanded_tasks, task_ids=expanded_ids)
    elapsed = time.time() - t0

    per_bench: dict[str, dict[str, list]] = {b: {} for b, _ in bench_tasks}
    for mid, ep in zip(expanded_ids, episodes):
        bench, _, task_id = mid.partition(sep)
        per_bench.setdefault(bench, {}).setdefault(task_id, []).append(ep)

    out: list[tuple[str, dict[str, Any]]] = []
    for bench, tasks in bench_tasks:
        payload = _score_benchmark_results(
            cfg, bench, tasks, per_bench.get(bench, {}), elapsed
        )
        out.append((bench, payload))
    return out


def _save_results(cfg: AgenticEvalConfig, bench: str, payload: dict[str, Any]) -> Path:
    out_root = Path(cfg.output_dir)
    model_tag = Path(cfg.model).name.replace("/", "_")
    alias = f"_{cfg.save_alias}" if cfg.save_alias else ""
    out_dir = out_root / f"{model_tag}{alias}" / bench
    out_dir.mkdir(parents=True, exist_ok=True)
    fpath = out_dir / "results.json"
    if fpath.exists() and not cfg.overwrite:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        fpath = out_dir / f"results_{stamp}.json"
    with fpath.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=4)
    return fpath


def _fmt(mean: float, std: float) -> str:
    return f"{mean * 100:.1f} (±{std * 100:.1f})"


def _print_summary_table(summaries: list[dict[str, Any]], passk: int) -> None:
    if not summaries:
        return
    num_samples = summaries[0]["num_samples_per_task"]
    extra_ks = sorted(
        {k for k in [2, 4, 8, num_samples] if k > passk and k <= num_samples}
    )
    tbl = Table(title="Agentic evaluation summary", show_lines=False)
    tbl.add_column("Benchmark", style="cyan")
    tbl.add_column("N tasks", justify="right")
    tbl.add_column("Samples/task", justify="right")
    tbl.add_column("Accuracy", justify="right")
    tbl.add_column(f"Pass@{passk}", justify="right")
    for k in extra_ks:
        tbl.add_column(f"Pass@{k}", justify="right")
    tbl.add_column(f"Pass^{passk}", justify="right")
    for k in extra_ks:
        tbl.add_column(f"Pass^{k}", justify="right")
    tbl.add_column("Elapsed (s)", justify="right")
    for s in summaries:
        row = [
            s["benchmark"],
            str(s["num_tasks"]),
            str(s["num_samples_per_task"]),
            _fmt(s["accuracy_mean"], s.get("accuracy_std", 0.0)),
            _fmt(s[f"pass@{passk}"], s.get(f"pass@{passk}_std", 0.0)),
        ]
        for k in extra_ks:
            row.append(_fmt(s.get(f"pass@{k}", 0.0), s.get(f"pass@{k}_std", 0.0)))
        row.append(_fmt(s.get(f"pass^{passk}", 0.0), s.get(f"pass^{passk}_std", 0.0)))
        for k in extra_ks:
            row.append(_fmt(s.get(f"pass^{k}", 0.0), s.get(f"pass^{k}_std", 0.0)))
        row.append(f"{s['elapsed_seconds']:.1f}")
        tbl.add_row(*row)
    console.print(tbl)


async def run_agentic_eval_async(cfg: AgenticEvalConfig) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    rollout_engine = _build_rollout_engine(cfg)
    try:
        loaded: list[tuple[str, list[AgenticTask]]] = []
        for bench in cfg.tasks:
            try:
                tasks = load_benchmark(
                    bench,
                    artifacts_dir=cfg.artifacts_dir or None,
                    max_problems=cfg.max_problems,
                    shuffle=cfg.shuffle,
                    shuffle_seed=cfg.shuffle_seed,
                )
            except (KeyError, FileNotFoundError) as e:
                console.log(f"[yellow]Skipping {bench}: {e}")
                continue
            if not tasks:
                console.log(f"[yellow]Benchmark {bench} yielded 0 tasks, skipping.")
                continue
            loaded.append((bench, tasks))

        if cfg.mixed_evals and len(loaded) > 1:
            console.log(
                f"[bold green]Mixed-evals enabled: pooling rollouts across "
                f"{len(loaded)} benchmarks into a single workflow engine."
            )
            results = await _evaluate_benchmarks_mixed(cfg, loaded, rollout_engine)
            for bench, payload in results:
                out_path = _save_results(cfg, bench, payload)
                console.log(f"[green]Saved {bench} -> {out_path}")
                summaries.append(payload["summary"])
        else:
            if cfg.mixed_evals and len(loaded) <= 1:
                console.log(
                    "[yellow]--mixed-evals set but only one benchmark loaded; "
                    "falling back to the sequential path."
                )
            for bench, tasks in loaded:
                payload = await _evaluate_one_benchmark(
                    cfg, bench, tasks, rollout_engine
                )
                out_path = _save_results(cfg, bench, payload)
                console.log(f"[green]Saved {bench} -> {out_path}")
                summaries.append(payload["summary"])
    finally:
        shutdown = getattr(rollout_engine, "shutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception:
                pass

    _print_summary_table(summaries, passk=cfg.passk)

    if summaries:
        overall_path = (
            Path(cfg.output_dir)
            / f"{Path(cfg.model).name.replace('/', '_')}{('_' + cfg.save_alias) if cfg.save_alias else ''}"
            / "overall_summary.json"
        )
        overall_path.parent.mkdir(parents=True, exist_ok=True)
        with overall_path.open("w", encoding="utf-8") as f:
            json.dump(
                {"config": asdict(cfg), "summaries": summaries},
                f,
                ensure_ascii=False,
                indent=4,
            )
        console.log(f"[green]Overall summary -> {overall_path}")
    return summaries


def run_agentic_eval(cfg: AgenticEvalConfig) -> list[dict[str, Any]]:
    """Synchronous entry point: loads every benchmark in ``cfg.tasks`` and
    evaluates it, writing per-benchmark and overall summaries under
    ``cfg.output_dir``.
    """
    return asyncio.run(run_agentic_eval_async(cfg))


__all__ = [
    "AgenticEvalConfig",
    "estimate_pass_at_k",
    "estimate_pass_hat_k",
    "run_agentic_eval",
    "run_agentic_eval_async",
]
