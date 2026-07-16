#!/usr/bin/env bash
# Launch agentic evaluation across benchmarks in experiments/artifacts/benchmarks/.
#
# All parameters are passed via explicit flags (see Usage below). Env vars
# of the same name still act as defaults so existing CI wrappers keep
# working, but flags win when both are set.
#
# The default backend is Ray-distributed vLLM — the model is loaded once
# per replica inside the eval process (no external HTTP server). Use
# --backend vllm for a single in-process engine, or --backend openai to
# talk to an external OpenAI-compatible server.
#
# Web search defaults to the local dense retrieval server, matching the
# fused-agent training setup (rllm/experiments/fused/train_fused_agent.sh).
#
# Vitabench (benchmarks/vitabench) is dispatched through `vita run` — name
# the domain as `vitabench:<domain>` where domain is delivery, ota, or
# instore, or use the bare alias `vitabench` to fan out across all three.
# The assistant agent is served by --model-path (vllm_local, or the
# --openai-base-url when --backend openai). user/evaluator agents default
# to gpt-4.1 via the shared litellm gateway; override with --vita-user-model
# / --vita-evaluator-model (and --vita-litellm-* knobs).
#
# Usage:
#   ./scripts/eval.sh \
#       --model-path /share/nlp/share/plm/Qwen3-4B-Thinking-2507 \
#       --benchmarks bamboogle,hotpotqa \
#       --backend ray_vllm --ray-replicas 8 --vllm-tp 1 \
#       --n-parallel 2048 --max-problems 128
#
#   # vitabench against a local HF checkpoint
#   ./scripts/eval.sh \
#       --model-path /path/to/hf_model --benchmarks vitabench \
#       --backend vllm --vllm-tp 1 --vllm-mem-fraction 0.22 \
#       --save-alias ckpt40
#
#   ./scripts/eval.sh --help

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults (env vars override the compile-time default; flags override env)
# ---------------------------------------------------------------------------

MODEL="${MODEL:-/share/nlp/share/plm/Qwen3-4B-Thinking-2507}"
BENCHMARKS_CSV=""
BACKEND="${BACKEND:-ray_vllm}"
RAY_REPLICAS="${RAY_REPLICAS:-8}"
RAY_ADDRESS="${RAY_ADDRESS:-}"
VLLM_TP="${VLLM_TP:-1}"
VLLM_MEM_FRACTION="${VLLM_MEM_FRACTION:-}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-}"
VLLM_DTYPE="${VLLM_DTYPE:-}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"
ENABLE_THINKING="${ENABLE_THINKING:-0}"
N_PARALLEL="${N_PARALLEL:-2048}"
MAX_PROBLEMS="${MAX_PROBLEMS:-128}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
PASSK="${PASSK:-1}"
MAX_STEPS="${MAX_STEPS:-96}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32768}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-}"
TOOLS_CSV="${TOOLS:-}"
RETRIEVAL_SERVER_URL="${RETRIEVAL_SERVER_URL:-http://127.0.0.1:16543}"
RETRIEVAL_MAX_RESULTS="${RETRIEVAL_MAX_RESULTS:-1}"
RLLM_RETRIEVAL_SUMMARIZE="${RLLM_RETRIEVAL_SUMMARIZE:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-output}"
SAVE_ALIAS="${SAVE_ALIAS:-}"
OVERWRITE="${OVERWRITE:-0}"
PARSER_NAME="${PARSER_NAME:-}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-}"
OPENAI_API_KEY="${OPENAI_API_KEY:-}"
SHUFFLE="${SHUFFLE:-1}"
SHUFFLE_SEED="${SHUFFLE_SEED:-0}"
PROMPT_FILE="${PROMPT_FILE:-}"
HARNESS="${HARNESS:-react}"
NO_SYSTEM_PROMPT="${NO_SYSTEM_PROMPT:-0}"
IS_NO_THINK_PROMPT="${IS_NO_THINK_PROMPT:-0}"
MIXED_EVALS="${MIXED_EVALS:-0}"
VERBOSE="${VERBOSE:-0}"

# vitabench-specific knobs (only read when --benchmarks includes vitabench[:domain]).
VITA_LITELLM_BASE_URL="${VITA_LITELLM_BASE_URL:-https://litellm.mybigai.ac.cn/}"
VITA_LITELLM_API_KEY="${VITA_LITELLM_API_KEY:-${OPENAI_API_KEY:-}}"
VITA_USER_MODEL="${VITA_USER_MODEL:-gpt-4.1}"
VITA_EVALUATOR_MODEL="${VITA_EVALUATOR_MODEL:-gpt-4.1}"
VITA_MAX_STEPS="${VITA_MAX_STEPS:-128}"
VITA_MAX_CONCURRENCY="${VITA_MAX_CONCURRENCY:-1}"
VITA_LANGUAGE="${VITA_LANGUAGE:-english}"
VITA_SAVE_PREFIX="${VITA_SAVE_PREFIX:-}"
VITA_CONDA_ENV="${VITA_CONDA_ENV:-vita}"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/eval.sh \
      --model-path /share/nlp/share/plm/Qwen3-4B-Thinking-2507 \
      --benchmarks bamboogle,hotpotqa --prompt prompts/react.txt \
      --backend ray_vllm --ray-replicas 8 --vllm-tp 1 \
      --n-parallel 2048 --max-problems 128

Flags:
  --model-path PATH                 HF model path or name (required)
  --benchmarks LIST                 Comma-separated benchmarks
                                    (e.g. bamboogle,hotpotqa,2wiki,musique)
  --backend {vllm|ray_vllm|ray_sglang|openai}
                                    Inference backend (default: ray_vllm)
  --ray-replicas N                  Ray vLLM replicas (ray_vllm only)
  --ray-address ADDR                Existing Ray cluster address
  --vllm-tp N                       vLLM tensor-parallel size per replica
  --sglang-tp N                     SGLang tensor-parallel size per replica
  --vllm-mem-fraction F             gpu-memory-utilization
  --vllm-max-model-len N            Override vLLM max_model_len
  --vllm-dtype STR                  auto|bfloat16|float16|...
  --vllm-enforce-eager              Disable CUDA graph capture
  --enable-thinking                 Enable thinking (extended reasoning) for the model
  --n-parallel N                    Concurrent agent rollouts
  --max-problems N                  Cap problems per benchmark
  --num-samples N                   Samples per problem
  --passk N                         Report pass@N
  --max-steps N                     Max tool-use turns per rollout
  --max-new-tokens N                Per-turn decode budget
  --max-prompt-length N             Max prompt length in tokens (default: 32768)
  --tools LIST                      Comma-separated rllm tool names.
                                    Defaults to the selected harness tools
                                    (react uses local_search; cot uses none).
  --retrieval-server-url URL        Override RETRIEVAL_SERVER_URL
  --retrieval-max-results N         Top-k docs returned per retrieval call
  --retrieval-summarize             Call the retrieval server /summarize
                                    endpoint after search (default: off)
  --no-retrieval-summarize          Return raw retrieved passages without
                                    summarization
  --output-dir DIR                  Where per-benchmark results.json lands
  --save-alias STR                  Suffix appended to the output dir name
  --overwrite                       Overwrite existing results.json
  --parser-name STR                 rllm ToolParser (qwen, r1, llama, ...)
  --openai-base-url URL             OpenAI-compatible endpoint (openai only)
  --openai-api-key KEY              Key for --openai-base-url
  --shuffle / --no-shuffle          Randomize tasks before truncation
                                    (default: on)
  --shuffle-seed N                  Seed for the pre-eval shuffle
  --prompt FILE                     Path to system-prompt .txt file
                                    (relative to evals, e.g. prompts/gem.txt)
  --harness NAME                    Agent harness preset (awm, simia, rlve,
                                    envscaler, toucan, gem, react, cot). Selects
                                    bundled system prompt, parser, and tools.
                                    Explicit --prompt/--parser-name/--tools
                                    override harness defaults.
  --no-system-prompt                Disable system prompt entirely
  --mixed-evals                     Pool rollouts across every benchmark into
                                    one workflow engine so slow tails on one
                                    benchmark overlap with fast finishers on
                                    another (keeps GPUs saturated end-to-end).
  --verbose                         Stream all eval logs (default: only print
                                    the final summary table)
  --is-no-think-prompt              Disable thinking prompt for the model (default: off)

Vitabench flags (only used when --benchmarks lists vitabench[:domain]):
  --vita-litellm-base-url URL       Gateway URL for user/evaluator agents
                                    (default: https://litellm.mybigai.ac.cn/)
  --vita-litellm-api-key KEY        Gateway API key (defaults to --openai-api-key,
                                    then $OPENAI_API_KEY)
  --vita-user-model NAME            Model id for user-agent     (default: gpt-4.1)
  --vita-evaluator-model NAME       Model id for evaluator-agent(default: gpt-4.1)
  --vita-max-steps N                vita run --max-steps        (default: 128)
  --vita-max-concurrency SPEC       vita run --max-concurrency  (default: 1)
                                    Either a single int (applied to every domain)
                                    or a comma-separated map, e.g.
                                    delivery=32,ota=16,instore=8. Domains missing
                                    from the map fall back to a `default=<N>`
                                    entry if present, otherwise to 1.
  --vita-language LANG              vita run --language         (default: english)
  --vita-save-prefix STR            Override the per-domain save-to prefix
                                    (default: derived from --save-alias)
  --vita-conda-env NAME             Conda env that has `vita`   (default: vita)

  -h, --help                        Show this help
EOF
}

die() { echo "[eval] ERROR: $*" >&2; exit 1; }

require_val() {
  [[ -n "${2:-}" ]] || die "flag '$1' requires a value"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path)            require_val "$1" "${2:-}"; MODEL="$2"; shift 2 ;;
    --benchmarks)            require_val "$1" "${2:-}"; BENCHMARKS_CSV="$2"; shift 2 ;;
    --backend)               require_val "$1" "${2:-}"; BACKEND="$2"; shift 2 ;;
    --ray-replicas)          require_val "$1" "${2:-}"; RAY_REPLICAS="$2"; shift 2 ;;
    --ray-address)           require_val "$1" "${2:-}"; RAY_ADDRESS="$2"; shift 2 ;;
    --vllm-tp)               require_val "$1" "${2:-}"; VLLM_TP="$2"; shift 2 ;;
    --sglang-tp)             require_val "$1" "${2:-}"; VLLM_TP="$2"; shift 2 ;;
    --vllm-mem-fraction)     require_val "$1" "${2:-}"; VLLM_MEM_FRACTION="$2"; shift 2 ;;
    --vllm-max-model-len)    require_val "$1" "${2:-}"; VLLM_MAX_MODEL_LEN="$2"; shift 2 ;;
    --vllm-dtype)            require_val "$1" "${2:-}"; VLLM_DTYPE="$2"; shift 2 ;;
    --vllm-enforce-eager)    VLLM_ENFORCE_EAGER=1; shift ;;
    --enable-thinking)       ENABLE_THINKING=1; shift ;;
    --n-parallel)            require_val "$1" "${2:-}"; N_PARALLEL="$2"; shift 2 ;;
    --max-problems)          require_val "$1" "${2:-}"; MAX_PROBLEMS="$2"; shift 2 ;;
    --num-samples)           require_val "$1" "${2:-}"; NUM_SAMPLES="$2"; shift 2 ;;
    --passk)                 require_val "$1" "${2:-}"; PASSK="$2"; shift 2 ;;
    --max-steps)             require_val "$1" "${2:-}"; MAX_STEPS="$2"; shift 2 ;;
    --max-new-tokens)        require_val "$1" "${2:-}"; MAX_NEW_TOKENS="$2"; shift 2 ;;
    --max-prompt-length)     require_val "$1" "${2:-}"; MAX_PROMPT_LENGTH="$2"; shift 2 ;;
    --tools)                 require_val "$1" "${2:-}"; TOOLS_CSV="$2"; shift 2 ;;
    --retrieval-server-url)  require_val "$1" "${2:-}"; RETRIEVAL_SERVER_URL="$2"; shift 2 ;;
    --retrieval-max-results) require_val "$1" "${2:-}"; RETRIEVAL_MAX_RESULTS="$2"; shift 2 ;;
    --retrieval-summarize)   RLLM_RETRIEVAL_SUMMARIZE=1; shift ;;
    --no-retrieval-summarize) RLLM_RETRIEVAL_SUMMARIZE=0; shift ;;
    --output-dir)            require_val "$1" "${2:-}"; OUTPUT_DIR="$2"; shift 2 ;;
    --save-alias)            require_val "$1" "${2:-}"; SAVE_ALIAS="$2"; shift 2 ;;
    --overwrite)             OVERWRITE=1; shift ;;
    --parser-name)           require_val "$1" "${2:-}"; PARSER_NAME="$2"; shift 2 ;;
    --openai-base-url)       require_val "$1" "${2:-}"; OPENAI_BASE_URL="$2"; shift 2 ;;
    --openai-api-key)        require_val "$1" "${2:-}"; OPENAI_API_KEY="$2"; shift 2 ;;
    --shuffle)               SHUFFLE=1; shift ;;
    --no-shuffle)            SHUFFLE=0; shift ;;
    --shuffle-seed)          require_val "$1" "${2:-}"; SHUFFLE_SEED="$2"; shift 2 ;;
    --prompt)                require_val "$1" "${2:-}"; PROMPT_FILE="$2"; shift 2 ;;
    --harness)               require_val "$1" "${2:-}"; HARNESS="$2"; shift 2 ;;
    --no-system-prompt)      NO_SYSTEM_PROMPT=1; shift ;;
    --is-no-think-prompt)    IS_NO_THINK_PROMPT=1; shift ;;
    --mixed-evals)           MIXED_EVALS=1; shift ;;
    --verbose)               VERBOSE=1; shift ;;
    --vita-litellm-base-url) require_val "$1" "${2:-}"; VITA_LITELLM_BASE_URL="$2"; shift 2 ;;
    --vita-litellm-api-key)  require_val "$1" "${2:-}"; VITA_LITELLM_API_KEY="$2"; shift 2 ;;
    --vita-user-model)       require_val "$1" "${2:-}"; VITA_USER_MODEL="$2"; shift 2 ;;
    --vita-evaluator-model)  require_val "$1" "${2:-}"; VITA_EVALUATOR_MODEL="$2"; shift 2 ;;
    --vita-max-steps)        require_val "$1" "${2:-}"; VITA_MAX_STEPS="$2"; shift 2 ;;
    --vita-max-concurrency)  require_val "$1" "${2:-}"; VITA_MAX_CONCURRENCY="$2"; shift 2 ;;
    --vita-language)         require_val "$1" "${2:-}"; VITA_LANGUAGE="$2"; shift 2 ;;
    --vita-save-prefix)      require_val "$1" "${2:-}"; VITA_SAVE_PREFIX="$2"; shift 2 ;;
    --vita-conda-env)        require_val "$1" "${2:-}"; VITA_CONDA_ENV="$2"; shift 2 ;;
    -h|--help)               usage; exit 0 ;;
    --) shift; break ;;
    *)  die "unknown argument: $1 (run '$0 --help' for the flag list)" ;;
  esac
done

[[ -n "$MODEL" ]] || die "--model-path is required"

IFS=',' read -r -a TASKS <<< "${BENCHMARKS_CSV}"
if [[ ${#TASKS[@]} -eq 1 && -z "${TASKS[0]}" ]]; then
  TASKS=(bamboogle hotpotqa 2wiki musique)
fi

# Partition requested tasks: vitabench domains go to `vita run`, everything
# else stays on the rllm `evals.evaluate` pipeline. `vitabench` (bare) is
# shorthand for all three domains. `vitabench:<domain>` selects just that one.
VITA_DOMAINS=()
RLLM_TASKS=()
for t in "${TASKS[@]}"; do
  case "$t" in
    vitabench)
      VITA_DOMAINS+=(delivery ota instore)
      ;;
    vitabench:delivery|vitabench:ota|vitabench:instore)
      VITA_DOMAINS+=("${t#vitabench:}")
      ;;
    vitabench:*)
      die "unknown vitabench domain '$t' (expected vitabench:delivery, vitabench:ota, or vitabench:instore)"
      ;;
    *)
      RLLM_TASKS+=("$t")
      ;;
  esac
done

TOOLS_ARR=()
if [[ -n "$TOOLS_CSV" ]]; then
  IFS=',' read -r -a TOOLS_ARR <<< "${TOOLS_CSV}"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EVALS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$EVALS_DIR"

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/rllm:${PYTHONPATH:-}"
if [[ -d "$HOME/app/rllm" ]]; then
  export PYTHONPATH="$HOME/app/rllm:${PYTHONPATH}"
fi
export RETRIEVAL_SERVER_URL
export RLLM_RETRIEVAL_SUMMARIZE

BACKEND_ARGS=(--backend "$BACKEND")

case "$BACKEND" in
  vllm)
    BACKEND_ARGS+=(
      --tensor-parallel-size "$VLLM_TP"
      --gpu-memory-utilization "${VLLM_MEM_FRACTION:-0.6}"
    )
    ;;
  ray_vllm)
    BACKEND_ARGS+=(
      --tensor-parallel-size "$VLLM_TP"
      --gpu-memory-utilization "${VLLM_MEM_FRACTION:-0.9}"
      --ray-num-replicas "$RAY_REPLICAS"
    )
    [[ -n "$RAY_ADDRESS" ]] && BACKEND_ARGS+=(--ray-address "$RAY_ADDRESS")
    ;;
  ray_sglang)
    BACKEND_ARGS+=(
      --tensor-parallel-size "$VLLM_TP"
      --gpu-memory-utilization "${VLLM_MEM_FRACTION:-0.9}"
      --ray-num-replicas "$RAY_REPLICAS"
    )
    ;;
  openai)
    BACKEND_ARGS+=(
      --base-url "${OPENAI_BASE_URL:-http://localhost:30000/v1}"
      --api-key "${OPENAI_API_KEY:-EMPTY}"
    )
    ;;
  *)
    die "unknown --backend '$BACKEND' (expected vllm, ray_vllm, ray_sglang, or openai)"
    ;;
esac

if [[ "$BACKEND" == "vllm" || "$BACKEND" == "ray_vllm" || "$BACKEND" == "ray_sglang" ]]; then
  [[ -n "$VLLM_MAX_MODEL_LEN" ]] && BACKEND_ARGS+=(--vllm-max-model-len "$VLLM_MAX_MODEL_LEN")
  [[ -n "$VLLM_DTYPE" ]] && BACKEND_ARGS+=(--vllm-dtype "$VLLM_DTYPE")
  [[ "$VLLM_ENFORCE_EAGER" == "1" ]] && BACKEND_ARGS+=(--vllm-enforce-eager)
  [[ "$ENABLE_THINKING" == "1" ]] && BACKEND_ARGS+=(--enable-thinking)
fi

if [[ "$SHUFFLE" == "1" ]]; then
  SHUFFLE_FLAG=(--shuffle)
else
  SHUFFLE_FLAG=(--no-shuffle)
fi

EXTRA_ARGS=()
[[ -n "$SAVE_ALIAS" ]] && EXTRA_ARGS+=(--save-alias "$SAVE_ALIAS")
[[ "$OVERWRITE" == "1" ]] && EXTRA_ARGS+=(--overwrite)
[[ -n "$MAX_PROBLEMS" ]] && EXTRA_ARGS+=(--max-problems "$MAX_PROBLEMS")
[[ -n "$PARSER_NAME" ]] && EXTRA_ARGS+=(--parser-name "$PARSER_NAME")
[[ -n "$PROMPT_FILE" ]] && EXTRA_ARGS+=(--prompt "$PROMPT_FILE")
[[ -n "$HARNESS" ]] && EXTRA_ARGS+=(--harness "$HARNESS")
[[ "$NO_SYSTEM_PROMPT" == "1" ]] && EXTRA_ARGS+=(--no-system-prompt)
[[ "$IS_NO_THINK_PROMPT" == "1" ]] && EXTRA_ARGS+=(--is-no-think-prompt)
[[ "$MIXED_EVALS" == "1" ]] && EXTRA_ARGS+=(--mixed-evals)
[[ -n "$MAX_PROMPT_LENGTH" ]] && EXTRA_ARGS+=(--max-prompt-length "$MAX_PROMPT_LENGTH")

TOOL_ARGS=()
if [[ ${#TOOLS_ARR[@]} -gt 0 ]]; then
  TOOL_ARGS=(--tools "${TOOLS_ARR[@]}")
fi

PY_CMD=(
  python -m evals.evaluate
  --model "$MODEL"
  "${BACKEND_ARGS[@]}"
  --tasks "${RLLM_TASKS[@]}"
  --artifacts-dir "/data/shuhan/eval_space/workspace/evals/dataset/benchmark"
  "${TOOL_ARGS[@]}"
  --retrieval-server-url "$RETRIEVAL_SERVER_URL"
  --retrieval-max-results "$RETRIEVAL_MAX_RESULTS"
  --summarize "$RLLM_RETRIEVAL_SUMMARIZE"
  --num-samples "$NUM_SAMPLES"
  --passk "$PASSK"
  --max-steps "$MAX_STEPS"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --n-parallel-tasks "$N_PARALLEL"
  --output-dir "$OUTPUT_DIR"
  "${SHUFFLE_FLAG[@]}"
  --shuffle-seed "$SHUFFLE_SEED"
  "${EXTRA_ARGS[@]}"
)

if [[ ${#RLLM_TASKS[@]} -gt 0 ]]; then
  echo "[eval] Effective config: backend=${BACKEND} ray_replicas=${RAY_REPLICAS} tp=${VLLM_TP} n_parallel=${N_PARALLEL} max_problems=${MAX_PROBLEMS} max_new_tokens=${MAX_NEW_TOKENS}" >&2
  echo "[eval] Prompt config: harness=${HARNESS:-<none>} no_system_prompt=${NO_SYSTEM_PROMPT} enable_thinking=${ENABLE_THINKING} mixed_evals=${MIXED_EVALS}" >&2
  if [[ ${#TOOLS_ARR[@]} -gt 0 ]]; then
    echo "[eval] Tasks: ${RLLM_TASKS[*]} | tools: ${TOOLS_ARR[*]} | output_dir=${OUTPUT_DIR}" >&2
  else
    echo "[eval] Tasks: ${RLLM_TASKS[*]} | tools: <harness-default> | output_dir=${OUTPUT_DIR}" >&2
  fi
  if [[ "$VERBOSE" == "1" ]]; then
    "${PY_CMD[@]}"
  else
    LOG_FILE="$(mktemp -t eval.XXXXXX.log)"
    echo "[eval] Running quietly; full log: $LOG_FILE" >&2
    set +e
    "${PY_CMD[@]}" >"$LOG_FILE" 2>&1
    rc=$?
    set -e
    if [[ $rc -ne 0 ]]; then
      echo "[eval] Run failed (exit $rc). Dumping log:" >&2
      cat "$LOG_FILE" >&2
      exit "$rc"
    fi
    awk '/Agentic evaluation summary/ {found=1} found {print}' "$LOG_FILE"
  fi
elif [[ ${#VITA_DOMAINS[@]} -eq 0 ]]; then
  die "no benchmarks to run (after stripping vitabench domains the task list was empty)"
fi

# ---------------------------------------------------------------------------
# Vitabench dispatch
# ---------------------------------------------------------------------------

if [[ ${#VITA_DOMAINS[@]} -gt 0 ]]; then
  VITA_REPO="$EVALS_DIR/benchmarks/vitabench"
  [[ -d "$VITA_REPO" ]] || die "vitabench repo not found at $VITA_REPO (did you run pip install -e benchmarks/vitabench in the '$VITA_CONDA_ENV' env?)"

  case "$BACKEND" in
    openai)
      VITA_ASSISTANT_BASE_URL="${OPENAI_BASE_URL:-http://localhost:30000/v1}"
      VITA_ASSISTANT_API_KEY="${OPENAI_API_KEY:-EMPTY}"
      VITA_ASSISTANT_MODE="litellm_http"
      ;;
    vllm|ray_vllm)
      VITA_ASSISTANT_MODE="vllm_local"
      ;;
    *)
      die "unsupported --backend '$BACKEND' for vitabench (use vllm, ray_vllm, or openai)"
      ;;
  esac

  # Resolve save-prefix: explicit flag > derived from --save-alias > 'vita'.
  if [[ -n "$VITA_SAVE_PREFIX" ]]; then
    VITA_PREFIX="$VITA_SAVE_PREFIX"
  elif [[ -n "$SAVE_ALIAS" ]]; then
    VITA_PREFIX="$SAVE_ALIAS"
  else
    VITA_PREFIX="vita"
  fi

  # Write a per-run models.yaml under configs/.
  VITA_CONFIG_DIR="$EVALS_DIR/configs"
  mkdir -p "$VITA_CONFIG_DIR"
  VITA_CONFIG="$VITA_CONFIG_DIR/vita_models.${VITA_PREFIX}.yaml"

  {
    echo "# Auto-generated by scripts/eval.sh for --save-alias=${VITA_PREFIX}."
    echo "# Assistant: ${VITA_ASSISTANT_MODE} on ${MODEL}"
    echo "# User/Evaluator: ${VITA_USER_MODEL}/${VITA_EVALUATOR_MODEL} via ${VITA_LITELLM_BASE_URL}"
    echo ""
    echo "default:"
    echo "  backend: litellm_http"
    echo "  api_key: ${VITA_LITELLM_API_KEY}"
    echo "  base_url: ${VITA_LITELLM_BASE_URL}"
    echo "  temperature: 0.0"
    echo "  max_tokens: 4096"
    echo "  max_input_tokens: 32768"
    echo "  headers:"
    echo "    Authorization: \"Bearer \""
    echo "    Content-Type: \"application/json\""
    echo ""
    echo "models:"
    echo "  - name: assistant-agent"
    if [[ "$VITA_ASSISTANT_MODE" == "litellm_http" ]]; then
      echo "    backend: litellm_http"
      echo "    base_url: ${VITA_ASSISTANT_BASE_URL}"
      echo "    api_key: ${VITA_ASSISTANT_API_KEY}"
      echo "    model: ${MODEL}"
    else
      echo "    backend: vllm_local"
      echo "    model: ${MODEL}"
      echo "    vllm:"
      echo "      tensor_parallel_size: ${VLLM_TP}"
      echo "      gpu_memory_utilization: ${VLLM_MEM_FRACTION:-0.6}"
      [[ -n "$VLLM_MAX_MODEL_LEN" ]] && echo "      max_model_len: ${VLLM_MAX_MODEL_LEN}"
      [[ -n "$VLLM_DTYPE" ]] && echo "      dtype: ${VLLM_DTYPE}"
      [[ "$VLLM_ENFORCE_EAGER" == "1" ]] && echo "      enforce_eager: true"
    fi
    echo ""
    echo "  - name: user-agent"
    echo "    model: ${VITA_USER_MODEL}"
    echo ""
    echo "  - name: evaluator-agent"
    echo "    model: ${VITA_EVALUATOR_MODEL}"
  } > "$VITA_CONFIG"

  echo "[eval] Wrote vitabench model config: $VITA_CONFIG" >&2

  # Activate the vita conda env without depending on the caller's shell state.
  CONDA_SH=""
  for cand in \
      "$HOME/app/anaconda3/etc/profile.d/conda.sh" \
      "/opt/conda/etc/profile.d/conda.sh" \
      "$HOME/miniconda3/etc/profile.d/conda.sh" \
      "$HOME/anaconda3/etc/profile.d/conda.sh"; do
    [[ -f "$cand" ]] && CONDA_SH="$cand" && break
  done
  [[ -n "$CONDA_SH" ]] || die "could not locate conda.sh; set CONDA_EXE or activate '$VITA_CONDA_ENV' before calling eval.sh"

  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate "$VITA_CONDA_ENV" || die "failed to activate conda env '$VITA_CONDA_ENV'"
  command -v vita >/dev/null 2>&1 || die "'vita' CLI not found in env '$VITA_CONDA_ENV' (pip install -e benchmarks/vitabench)"

  # The litellm gateway is internal — avoid forcing outbound traffic through
  # whatever proxy the caller's shell had configured.
  unset https_proxy http_proxy all_proxy HTTPS_PROXY HTTP_PROXY ALL_PROXY

  export VITA_MODEL_CONFIG_PATH="$VITA_CONFIG"

  # Allow `--overwrite` to force a fresh save file (vita prompts interactively
  # when the file already exists and otherwise requires a config-hash match).
  LOGDIR="$EVALS_DIR/eval_logs"
  mkdir -p "$LOGDIR"

  # Resolve VITA_MAX_CONCURRENCY into a per-domain lookup.
  # - Scalar (e.g. "32") applies to every domain.
  # - Map (e.g. "delivery=32,ota=16,instore=8") sets per-domain values; a
  #   "default=<N>" entry covers unlisted domains (falls back to 1 if absent).
  declare -A VITA_CONC_MAP=()
  VITA_CONC_DEFAULT=1
  if [[ "$VITA_MAX_CONCURRENCY" == *=* ]]; then
    IFS=',' read -r -a _vconc_pairs <<< "$VITA_MAX_CONCURRENCY"
    for _kv in "${_vconc_pairs[@]}"; do
      [[ -z "$_kv" ]] && continue
      _k="${_kv%%=*}"; _v="${_kv#*=}"
      [[ "$_v" =~ ^[0-9]+$ ]] || die "--vita-max-concurrency: non-integer value for '$_k' in '$VITA_MAX_CONCURRENCY'"
      if [[ "$_k" == "default" ]]; then
        VITA_CONC_DEFAULT="$_v"
      else
        VITA_CONC_MAP["$_k"]="$_v"
      fi
    done
  else
    [[ "$VITA_MAX_CONCURRENCY" =~ ^[0-9]+$ ]] || die "--vita-max-concurrency must be an int or key=val map, got '$VITA_MAX_CONCURRENCY'"
    VITA_CONC_DEFAULT="$VITA_MAX_CONCURRENCY"
  fi

  pushd "$VITA_REPO" >/dev/null

  VITA_STATUSES=()
  for d in "${VITA_DOMAINS[@]}"; do
    save_to="${VITA_PREFIX}_${d}"
    save_path="$VITA_REPO/data/simulations/${save_to}"
    if [[ "$OVERWRITE" == "1" && -f "$save_path" ]]; then
      rm -f "$save_path"
      echo "[eval] --overwrite: removed existing $save_path" >&2
    fi
    log_file="$LOGDIR/${save_to}.log"
    conc="${VITA_CONC_MAP[$d]:-$VITA_CONC_DEFAULT}"
    echo "[eval] vita run --domain=$d max-concurrency=$conc save-to=$save_to (log: $log_file)" >&2

    set +e
    yes | vita run \
      --domain "$d" \
      --user-llm user-agent \
      --agent-llm assistant-agent \
      --evaluator-llm evaluator-agent \
      --max-steps "$VITA_MAX_STEPS" \
      --max-concurrency "$conc" \
      --language "$VITA_LANGUAGE" \
      --save-to "$save_to" \
      --log-level INFO \
      >"$log_file" 2>&1
    rc=$?
    set -e
    VITA_STATUSES+=("$d=$rc")
    if [[ $rc -ne 0 ]]; then
      echo "[eval] vita run for domain '$d' exited with status $rc (see $log_file)" >&2
    fi
  done

  popd >/dev/null

  # Summarise vitabench results: avg reward per domain.
  echo ""
  echo "=== Vitabench summary (${VITA_PREFIX}) ==="
  for d in "${VITA_DOMAINS[@]}"; do
    save_path="$VITA_REPO/data/simulations/${VITA_PREFIX}_${d}"
    if [[ -f "$save_path" ]]; then
      python - "$save_path" "$d" <<'PY'
import json, sys
path, domain = sys.argv[1], sys.argv[2]
with open(path) as fp:
    data = json.load(fp)
sims = data.get("simulations", [])
rewards = [s.get("reward_info", {}).get("reward") for s in sims]
rewards = [r for r in rewards if r is not None]
avg = sum(rewards) / len(rewards) if rewards else 0.0
print(f"{domain:>10s}: n={len(sims):3d} scored={len(rewards):3d} avg_reward={avg:.3f}")
PY
    else
      echo "${d:>10s}: (no output at $save_path)"
    fi
  done

  # Fail the whole run if any domain returned non-zero.
  for entry in "${VITA_STATUSES[@]}"; do
    rc="${entry##*=}"
    [[ "$rc" == "0" ]] || exit "$rc"
  done
fi
