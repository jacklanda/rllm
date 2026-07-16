#!/usr/bin/env bash

#export CUDA_VISIBLE_DEVICES=0
#export CUDA_VISIBLE_DEVICES=0,1
#export CUDA_VISIBLE_DEVICES=1,2,3,4
#export CUDA_VISIBLE_DEVICES=4,5,6,7
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


# Models
# - 2nd SoTA Zero-format RL 4B: /share/nlp/wutong1/hf_models/dapo-RL-zero-4B-v10-140
# - SoTA Warmup 4B: /share/nlp/share/plm/checkpoint/MathReasoner-20250916 
# - Base Model: /share/nlp/share/plm/Qwen3-4B-Instruct-2507
# - Reasoning Model: /share/nlp/share/plm/Qwen3-4B

#python eval_parallel_old.py \
/share/nlp/wutong1/anaconda3/envs/sglang/bin/python -u evaluate.py \
    --tasks AIME24 \
    --model_path /share/nlp/share/plm/Qwen3-4B \
    --instruction "" \
    --tp_size 8 \
    --max_problems 30 \
    --log_samples 3 \
    --max_new_tokens 30000 \
    --temperature 0.6 \
    --batch_size 8 \
    --num_samples 8 \
    --passk 1 \
    --overwrite \
    --save_total_limit 3 \
    --apply_chat \
    --parallel_reasoning
    #--debug
