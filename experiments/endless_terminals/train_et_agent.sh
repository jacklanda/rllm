#!/bin/bash
# ============================================================================
# Endless Terminals (ET) agent training script.
#
# Routes the experiments/artifacts/endless_terminals/train.parquet rows
# through ETEnv + ETAgent. Images are pre-built on the remote daemon at
# $DOCKER_HOST; the agent_ppo_trainer pre-flight ping + ET image spot-check
# guard against misconfigured connectivity.
#
# To run:
#   bash experiments/endless_terminals/train_et_agent.sh
# ============================================================================
set -x

unset HIP
unset CUDA_VISIBLE_DEVICES
unset ROCR_VISIBLE_DEVICES

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export TOKENIZERS_PARALLELISM=false

# Remote Docker daemon hosting the pre-built gemcli/task_* images.
export DOCKER_HOST=tcp://10.2.152.50:2375
export DOCKER_API_VERSION=1.44

python3 -m rllm.trainer.verl.train_agent_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=experiments/artifacts/endless_terminals/train.parquet \
    data.val_files=experiments/artifacts/endless_terminals/train.parquet \
    data.train_batch_size=8 \
    data.val_batch_size=128 \
    data.max_prompt_length=4096 \
    data.max_response_length=32768 \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=8 \
    actor_rollout_ref.model.path=/share/nlp/share/plm/Qwen3-4B-Thinking-2507 \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32000 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=8 \
    actor_rollout_ref.model.use_liger=False \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode="async" \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='ETRL' \
    trainer.experiment_name='gem-et-grpo-qwen3-4b-thinking-2507-v0.1.0' \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=1000 \
    trainer.total_epochs=10 \
    trainer.default_hdfs_dir=null \
    trainer.run_id=null \
    rllm.env.name=endless_terminals \
    rllm.agent.name=et_agent \
    rllm.agent.max_steps=32 \
    rllm.agent.overlong_filter=True \
    rllm.mask_truncated_samples=True \
    rllm.incremental_tokenization=True \
    rllm.trajectory_filtering.max_step_retries=0 \
    +rllm.trajectory_filtering.validate_boxed_per_step=False \
    +rllm.trajectory_filtering.enforce_react_structure=False \
    +rllm.rejection_sample.filter_zero_variance=True \
    +rllm.rejection_sample.multiplier=2 \
    rllm.agent.trajectory_timeout=600
