"""CLI entry point for agentic evaluation.

Default backend is in-process vLLM (no HTTP server needed). Use
``--backend ray_vllm --ray-num-replicas N`` to run N parallel vLLM
replicas on separate GPUs under Ray, ``--backend ray_sglang`` to run a
data-parallel SGLang pool, or ``--backend openai`` to talk to an external
OpenAI-compatible server.

Usage (see scripts/eval.sh for a batteries-included wrapper)::

    python -m evals.evaluate \
        --model /share/nlp/share/plm/Qwen3-4B-Thinking-2507 \
        --tasks bamboogle hotpotqa 2wiki musique gaia \
        --num-samples 1 --max-steps 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the ``evals`` package importable when the script is run directly.
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
for p in (str(_REPO_ROOT), str(_THIS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from evals.benchmark_loader import BENCHMARK_REGISTRY  # noqa: E402
from evals.harness import HARNESS_REGISTRY, get_harness  # noqa: E402
from evals.runner import AgenticEvalConfig, run_agentic_eval  # noqa: E402

_EVALS_PROMPTS_DIR = _THIS_DIR / "prompts"


def _parse_args() -> AgenticEvalConfig:
    parser = argparse.ArgumentParser(
        description="Agentic (tool-using) evaluation for benchmarks in experiments/artifacts/benchmarks/",
    )

    parser.add_argument(
        "--model", required=True, help="HF model path / name (local path or hub id)"
    )
    parser.add_argument(
        "--backend",
        choices=("vllm", "ray_vllm", "ray_sglang", "openai"),
        default="vllm",
        help="Inference backend. 'vllm' loads the model in-process (default); "
        "'ray_vllm' spawns multiple Ray actors, each hosting its own vLLM "
        "engine, for concurrent agents across GPUs; 'ray_sglang' spawns a "
        "data-parallel SGLang worker pool using --ray-num-replicas workers; "
        "'openai' talks to an external OpenAI-compatible server.",
    )
    parser.add_argument(
        "--harness",
        choices=sorted(HARNESS_REGISTRY),
        default="react",
        help="Agent harness preset. Selects a bundled system prompt, parser, "
        "and tool list. Explicit --prompt, --parser-name, or --tools flags "
        "override the harness defaults. Defaults to react.",
    )
    parser.add_argument(
        "--base-url",
        default="",
        help="OpenAI-compatible base URL (only used with --backend openai)",
    )
    parser.add_argument(
        "--api-key", default="", help="API key for the server (EMPTY for local)"
    )

    # Ray-distributed vLLM knobs (ignored unless --backend ray_vllm)
    parser.add_argument(
        "--ray-num-replicas",
        type=int,
        default=1,
        help="Number of Ray-hosted vLLM replicas. Each owns --tensor-parallel-size "
        "GPUs. Aggregate GPU count = replicas * TP.",
    )
    parser.add_argument(
        "--ray-address",
        default="",
        help="Ray cluster address (defaults to $RAY_ADDRESS or a local cluster).",
    )

    # vLLM engine knobs (ignored when --backend openai)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--vllm-max-model-len", type=int, default=None)
    parser.add_argument("--vllm-dtype", default="auto")
    parser.add_argument("--summarize",type=int, default=0)
    parser.add_argument("--is-no-think-prompt", action="store_true")
    parser.add_argument("--vllm-enforce-eager", action="store_true")
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable thinking (extended reasoning) for the evaluation model",
    )

    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["bamboogle", "hotpotqa", "2wiki", "musique"],
        choices=sorted(BENCHMARK_REGISTRY),
        help="Benchmarks to run",
    )
    parser.add_argument(
        "--artifacts-dir", default="", help="Override artifacts/benchmarks directory"
    )
    parser.add_argument("--max-problems", type=int, default=None)
    parser.add_argument(
        "--shuffle",
        dest="shuffle",
        action="store_true",
        default=True,
        help="Shuffle benchmark tasks with --shuffle-seed before truncation (default).",
    )
    parser.add_argument(
        "--no-shuffle",
        dest="shuffle",
        action="store_false",
        help="Preserve benchmark file order (disables the default shuffle).",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=0,
        help="Seed for the pre-eval task shuffle. Same seed -> same ordering.",
    )

    parser.add_argument(
        "--tools",
        nargs="+",
        default=None,
        help="rllm tool names to expose to the agent. 'web_search' routes to "
        "the local dense retrieval server (RETRIEVAL_SERVER_URL), matching the "
        "fused-agent training setup. Overrides harness default when set.",
    )
    parser.add_argument(
        "--retrieval-server-url",
        default="",
        help="Dense retrieval server URL for 'web_search' (overrides RETRIEVAL_SERVER_URL)",
    )
    parser.add_argument("--retrieval-max-results", type=int, default=10)
    parser.add_argument("--retrieval-timeout", type=float, default=3600.0)
    parser.add_argument(
        "--parser-name",
        default=None,
        help="rllm ToolParser name (qwen, r1, llama, ...). Overrides harness default when set.",
    )
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--max-response-length", type=int, default=8192)
    parser.add_argument("--max-prompt-length", type=int, default=32768)
    parser.add_argument("--max-new-tokens", type=int, default=2048)

    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--passk", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)

    parser.add_argument("--n-parallel-tasks", type=int, default=32)
    parser.add_argument("--retry-limit", type=int, default=2)
    parser.add_argument(
        "--mixed-evals",
        action="store_true",
        help="Pool rollouts across every benchmark in --tasks into a single "
        "workflow engine, so slow-tail problems on one benchmark overlap with "
        "fast finishers on another. Per-benchmark scoring and outputs are "
        "unchanged; only the dispatching changes. Falls back to the "
        "one-benchmark-at-a-time path when only one benchmark is loaded.",
    )

    parser.add_argument(
        "--prompt",
        default="",
        help="Path to a .txt file whose contents override the agent system prompt. "
        "Relative paths are resolved from evals/prompts/. "
        "Defaults to the selected harness system prompt.",
    )
    parser.add_argument("--no-system-prompt", action="store_true")
    parser.add_argument("--output-dir", default="output_agentic")
    parser.add_argument("--save-alias", default="")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    # Resolve harness defaults — explicit flags override harness values.
    harness_cfg = get_harness(args.harness) if args.harness else None

    # System prompt resolution: --no-system-prompt > --prompt > harness > built-in
    system_prompt = None
    if args.no_system_prompt:
        system_prompt = ""
    elif args.prompt:
        p = Path(args.prompt)
        if not p.is_absolute():
            p = p
        system_prompt = p.read_text(encoding="utf-8").strip()
    elif harness_cfg:
        system_prompt = harness_cfg.system_prompt

    if harness_cfg:
        if harness_cfg.user_prompt_template:
            user_prompt_template = harness_cfg.user_prompt_template
        else:
            user_prompt_template = "{problem_statement}"
    
    # Parser resolution: explicit --parser-name > harness > "qwen"
    parser_name = args.parser_name
    if parser_name is None:
        parser_name = harness_cfg.parser_name if harness_cfg else "qwen"

    # Tools resolution: explicit --tools > harness > ["web_search"]
    tools = args.tools
    if tools is None:
        tools = list(harness_cfg.tools) if harness_cfg else ["web_search"]

    print(f"load benchmarks from {args.artifacts_dir}")
    return AgenticEvalConfig(
        model=args.model,
        backend=args.backend,
        harness=args.harness or "",
        base_url=args.base_url,
        api_key=args.api_key,
        ray_num_replicas=args.ray_num_replicas,
        ray_address=args.ray_address,
        tasks=list(args.tasks),
        artifacts_dir=args.artifacts_dir,
        max_problems=args.max_problems,
        shuffle=args.shuffle,
        shuffle_seed=args.shuffle_seed,
        tools=list(tools),
        retrieval_server_url=args.retrieval_server_url,
        retrieval_max_results=args.retrieval_max_results,
        summarize=args.summarize,
        retrieval_timeout=args.retrieval_timeout,
        parser_name=parser_name,
        max_steps=args.max_steps,
        max_response_length=args.max_response_length,
        max_prompt_length=args.max_prompt_length,
        max_new_tokens=args.max_new_tokens,
        num_samples=args.num_samples,
        passk=args.passk,
        temperature=args.temperature,
        top_p=args.top_p,
        n_parallel_tasks=args.n_parallel_tasks,
        retry_limit=args.retry_limit,
        mixed_evals=args.mixed_evals,
        no_system_prompt=args.no_system_prompt,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        vllm_max_model_len=args.vllm_max_model_len,
        vllm_dtype=args.vllm_dtype,
        vllm_enforce_eager=args.vllm_enforce_eager,
        enable_thinking=args.enable_thinking,
        **({"system_prompt": system_prompt} if system_prompt is not None else {}),
        **({"user_prompt_template": user_prompt_template} if user_prompt_template is not None else {}),
        is_no_think_prompt=args.is_no_think_prompt,
        output_dir=args.output_dir,
        save_alias=args.save_alias,
        overwrite=args.overwrite,
    )


def main() -> None:
    cfg = _parse_args()
    run_agentic_eval(cfg)


if __name__ == "__main__":
    main()
