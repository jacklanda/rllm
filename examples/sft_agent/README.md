# Agent SFT (FSDP2) for Qwen3-4B-Thinking-2507

Supervised fine-tuning of `Qwen3-4B-Thinking-2507` on high-reward agent trajectories
collected via offline rejection sampling. The model learns the successful multi-step
tool-use behavior by imitating the assistant turns of those trajectories.

This mirrors `examples/sft/` but is adapted for full multi-turn agent data and uses
**FSDP2** with **full fine-tuning** (no LoRA).

## What gets trained (masking semantics)

Each trajectory is a chat with these roles:

- `system` — the tool specification / agent instructions (**masked**, loss = 0)
- `user` — the task prompt and every environment observation / tool output (**masked**, loss = 0)
- `assistant` — the model's reasoning (`<think>...</think>`), text, and tool calls (**learning target**, loss = 1)

`agent_sft_dataset.py::AgentSFTDataset` renders each message with the project's
`QwenChatTemplateParser` and sets `loss_mask = 1` only on assistant tokens. Everything
else is masked out of the loss. This is the `cumulative` method: every assistant turn in
the trajectory contributes to the loss (not just the last one).

## Files

| File | Purpose |
|------|---------|
| `prepare_agent_sft_data.py` | Optionally convert RS `trajectories.json` → `data/train.parquet` + `data/val.parquet` |
| `agent_sft_dataset.py` | Multi-turn dataset with assistant-only loss masking; reads parquet, JSON, or JSONL |
| `train_agent_sft.py` | Hydra entrypoint → `AgentSFTTrainer` (verl `FSDPSFTTrainer`) |
| `train_agent_sft.sh` | `torchrun` launcher with the FSDP2 full-FT config overrides |

## 1. Prepare data (optional)

```bash
cd examples/sft_agent
python prepare_agent_sft_data.py \
    --input ../../experiments/rejection_sampling/offline-rs-fused-20260527101144/trajectories.json \
    --output-dir ./data
```

This reads `data["selected_trajectories"]`, drops empty/assistant-less trajectories, prints
a token-length histogram, and writes a stratified ~3% validation hold-out (by `data_source`).
Expected output: `train.parquet` (~2874 rows) and `val.parquet` (~90 rows).

You can also train directly from an offline-RS JSON file. From the repo root, the dataset
loader converts `selected_trajectories[*].trajectory` into `messages` in memory:

```bash
./examples/sft_agent/train_agent_sft.sh \
    data.train_files=experiments/rejection_sampling/offline-rs-fused-20260528163135/trajectories.json
```

## 2. Train

```bash
cd examples/sft_agent
bash train_agent_sft.sh
```

Defaults: 8 GPUs, `max_length=16384` (covers ~97% of trajectories, `truncation=right`),
global batch 32, micro-batch 1/GPU, lr 1e-5, 3 epochs, gradient checkpointing on.
With 2875 training trajectories the run is 89 steps/epoch (267 total). Checkpoints and
validation are written every 89 steps (once per epoch) under
`outputs/qwen3_4b_thinking_agent_sft/global_step_{89,178,267}/`. Each checkpoint includes a
consolidated `huggingface/` dir directly loadable with `AutoModelForCausalLM.from_pretrained`.

Override anything by appending Hydra args, e.g. a quick smoke test:

```bash
bash train_agent_sft.sh trainer.total_training_steps=2 data.train_max_samples=64 data.val_max_samples=16
```

Multiple train/validation data files can be passed directly as comma-separated paths:

```bash
bash train_agent_sft.sh \
    data.train_files=/path/train-000.parquet,/path/train-001.parquet \
    data.val_files=/path/val-000.parquet,/path/val-001.parquet
```

Use a different model or GPU count via env vars:

```bash
MODEL_PATH=/path/to/model NPROC=4 bash train_agent_sft.sh
```

## Memory / sequence-length tuning

Full fine-tuning of a 4B model at 16k context is memory heavy. If you hit OOM:

- Lower `data.max_length` (e.g. `8192`) — drops more tail tokens from long trajectories.
- Shard sequence length across GPUs: `ulysses_sequence_parallel_size=2` (or `4`).
  Note `train_batch_size` must stay divisible by `world_size / ulysses_sequence_parallel_size`.
- Offload params: `model.fsdp_config.cpu_offload=true model.fsdp_config.offload_params=true`.

To raise `max_length` to cover all trajectories (max ≈ 19.1k tokens), set
`data.max_length=24576` and compensate with sequence parallelism.

## How it fits together

`train_agent_sft.py` loads the `agent_sft_trainer` config (rllm) which composes verl's
`sft_trainer` defaults (where `model.strategy: fsdp2`). `AgentSFTTrainer` (backend `verl`)
runs verl's `FSDPSFTTrainer`. The `data.custom_cls` override points the trainer's
`create_sft_dataset` at `AgentSFTDataset`, so the assistant-only masking is applied during
tokenization. No edits to the installed `verl` or shared `rllm` package are required.
