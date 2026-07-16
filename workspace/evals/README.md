# Parallel Reasoner Eval

Benchmark reasoning models on math problems (AIME, GPQA, MATH500, etc.) with distributed inference support.

## Setup

```bash
cd evaluation

conda create -n eval python=3.11
conda activate eval
pip install -r requirements.txt
```

## Quick Start

Run evaluation for NPR models on AIME 2025:

```bash
./scripts/eval.sh \
    --cuda 0,1,2,3,4,5,6,7 \
    --tp_size 2 \
    --dp_size 4 \
    --task "AIME25" \
    --max_eval_samples 30 \
    --eval_batch_size 8 \
    --model_path NPR-Warmup-4B-Inst \
    --prompt_path prompts/npr.txt \
    --engine parallel \
    --num_samples 1 \
    --k 1 \
    --max_new_tokens 40000 \
    --temperature 1.0 \
    --top_p 0.7 \
    --top_k -1 \
    --overwrite \
    --apply_chat
```

Test models on math benchmarks with pass@k scoring.

**Key flags:**
- `--tasks`: AIME24, AIME25, GPQA, MATH500, HMMT, Minerva, Olympiad, MMLU, BBEH, ZebraLogic
- `--dp_size`: Split work across N engines (default: # of GPUs)
- `--tp_size`: Tensor parallel for large models (default: 1)
- `--parallel_reasoning`: Enable structured parallel reasoning
- `--enable_thinking`: Use thinking mode (Qwen3-specific)
- `--temperature`, `--top_p`, `--top_k`: Sampling params

## Implementation Notes

**Parallelism modes:**
- **DP** (data parallel): Split dataset across N engines for higher throughput
- **TP** (tensor parallel): Split large model across N GPUs

Each DP engine processes its chunk independently, then results are merged.

**Memory:**
- TP mode: 80% GPU memory
- DP mode: 70% // dp_size per engine

**Datasets:**
- Math: AIME24/25, AMC23, MATH500, HMMT, Minerva, Olympiad
- Reasoning: GPQA, MMLU, BBEH, ZebraLogic

Results include pass@k metrics, format scores, and token counts.

## Agentic evaluation

Multi-turn, tool-using evaluation for the benchmarks under
`rllm/experiments/artifacts/benchmarks/` is implemented at the top of
`evals/` (see `evals/evaluate.py`). It drives
rllm's `ToolAgent` + `ToolEnvironment` + `MultiTurnWorkflow` against an
OpenAI-compatible server (vLLM / SGLang) and scores with
`rllm.rewards.reward_fn.search_reward_fn`.

```bash
# 1. Launch an OpenAI-compatible server for your model, e.g.
#    python -m sglang.launch_server --model-path Qwen/Qwen3-4B --port 30000
export OPENAI_BASE_URL=http://localhost:30000/v1
export OPENAI_API_KEY=EMPTY

# 2. Run the eval across one or more benchmarks
bash evals/scripts/eval_agentic.sh Qwen/Qwen3-4B bamboogle hotpotqa 2wiki musique gaia
```

Benchmarks supported: `bamboogle`, `hotpotqa`, `2wiki`, `musique`, `gaia`,
`medqa`, `browse_comp`, `browsecomp_plus`, `simpleqa_verified`, `scienceqa`,
`hle`, `deepsearchqa`. See `evals/benchmark_loader.py` to register more.

Outputs land in `output_agentic/<model>/<benchmark>/results.json` plus an
`overall_summary.json` with per-task accuracy and pass@k.

