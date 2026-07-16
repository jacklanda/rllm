#!/usr/bin/env bash

#export CUDA_VISIBLE_DEVICES=0,1,2,3
#export CUDA_VISIBLE_DEVICES=4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python eval_vanilla.py \
    --tasks AIME25 AIME24 AMC23 MATH500 \
    --model_path /share/nlp/share/plm/Qwen3-32B \
    --tp_size 8 \
    --max_new_tokens 30000 \
    --temperature 0.6 \
    --batch_size 128 \
    --num_samples 8 \
    --passk 1 \
    --overwrite \
    --apply_chat
