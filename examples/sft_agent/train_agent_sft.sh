#!/usr/bin/env bash
# FSDP2 full fine-tuning of Qwen3-4B-Thinking-2507 on agent trajectories.
#
# Loss is computed ONLY on assistant utterances (incl. <think> reasoning + tool calls).
# The system tool-spec message and all user (tool-observation) messages are masked out,
# via examples/sft_agent/agent_sft_dataset.py (cumulative method).
#
# `data.train_files` and `data.val_files` may point at parquet, JSON, or JSONL data.
set -euo pipefail

PROJECT_DIR=$(pwd)
SCRIPT_DIR=${PROJECT_DIR}/examples/sft_agent
MODEL_PATH="${MODEL_PATH:-/share/nlp/share/plm/Qwen3-4B-Thinking-2507}"
NPROC="${NPROC:-8}"
ULYSSES_SEQUENCE_PARALLEL_SIZE="${ULYSSES_SEQUENCE_PARALLEL_SIZE:-${NPROC}}"

if (( NPROC % ULYSSES_SEQUENCE_PARALLEL_SIZE != 0 )); then
    echo "NPROC (${NPROC}) must be divisible by ULYSSES_SEQUENCE_PARALLEL_SIZE (${ULYSSES_SEQUENCE_PARALLEL_SIZE})." >&2
    exit 1
fi

normalize_comma_path_override() {
    local arg="$1"
    local key value item escaped normalized

    case "${arg}" in
        data.train_files=*|data.val_files=*)
            key="${arg%%=*}"
            value="${arg#*=}"
            ;;
        *)
            printf '%s\n' "${arg}"
            return
            ;;
    esac

    if [[ "${value}" != *,* || "${value}" == \[*\] ]]; then
        printf '%s\n' "${arg}"
        return
    fi

    normalized="["
    IFS=',' read -ra paths <<< "${value}"
    for item in "${paths[@]}"; do
        item="${item#"${item%%[![:space:]]*}"}"
        item="${item%"${item##*[![:space:]]}"}"
        [[ -z "${item}" ]] && continue

        escaped="${item//\\/\\\\}"
        escaped="${escaped//\"/\\\"}"
        normalized+="\"${escaped}\","
    done

    if [[ "${normalized}" == "[" ]]; then
        printf '%s\n' "${arg}"
    else
        normalized="${normalized%,}]"
        printf '%s\n' "${key}=${normalized}"
    fi
}

EXTRA_ARGS=()
for arg in "$@"; do
    EXTRA_ARGS+=("$(normalize_comma_path_override "${arg}")")
done

OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" \
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:128}" \
python -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NPROC}" \
    "${SCRIPT_DIR}/train_agent_sft.py" \
    model.partial_pretrain="${MODEL_PATH}" \
    model.strategy=fsdp2 \
    model.trust_remote_code=true \
    model.fsdp_config.model_dtype=bfloat16 \
    model.enable_gradient_checkpointing=true \
    model.lora_rank=0 \
    data.custom_cls.path="${SCRIPT_DIR}/agent_sft_dataset.py" \
    data.custom_cls.name=AgentSFTDataset \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.rllm.tokenize_and_mask_method=cumulative \
    data.train_files="${SCRIPT_DIR}/data/train.parquet" \
    data.val_files="${SCRIPT_DIR}/data/val.parquet" \
    data.max_length=80000 \
    data.truncation=right \
    data.train_batch_size=16 \
    data.micro_batch_size_per_gpu=1 \
    optim.lr=1e-5 \
    optim.weight_decay=0.01 \
    use_remove_padding=true \
    ulysses_sequence_parallel_size="${ULYSSES_SEQUENCE_PARALLEL_SIZE}" \
    trainer.total_epochs=1 \
    trainer.save_freq=89 \
    trainer.test_freq=89 \
    trainer.default_local_dir="${PROJECT_DIR}/checkpoints/sft_agent/qwen3-4b-thinking-2507-individual" \
    trainer.project_name=agent-sft \
    trainer.experiment_name=qwen3-4b-thinking-2507-individual \
    trainer.logger='["console","wandb"]' \
    "trainer.checkpoint.save_contents=[model,extra,hf]" \
    "${EXTRA_ARGS[@]}"
