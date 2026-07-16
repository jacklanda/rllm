#!/usr/bin/env bash

#export CUDA_VISIBLE_DEVICES=1,2,3,4
#export CUDA_VISIBLE_DEVICES=4,5,6,7
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python eval_parallel.py \
    --tasks AIME24 AIME25 AMC23 \
    --model_path /share/nlp/liuyang/workspace/sr/train/Multiverse/ckpts/MathReasoner-20250902-015640/checkpoint-200 \
    --tp_size 8 \
    --max_new_tokens 30000 \
    --temperature 0.6 \
    --batch_size 128 \
    --num_samples 8 \
    --passk 1 \
    --overwrite \
    --apply_chat \
    --parallel_reasoning \
    --debug
