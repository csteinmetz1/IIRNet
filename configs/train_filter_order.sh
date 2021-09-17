#!/bin/bash

for filter_order in 4 8 16 32 64
do
    CUDA_VISIBLE_DEVICES=0 python train.py \
    --gpus 1 \
    --model_order $filter_order \
    --max_train_order $filter_order \
    --batch_size 128 \
    --num_workers 8 \
    --lr 1e-8 \
    --gradient_clip_val 0.9 \
    --gradient_clip_algorithm norm \
    --hidden_dim 1024 \
    --shuffle \
    --filter_method all \
    --max_epochs 500 \
    --num_train_examples 8333 \
    --track_grad_norm 2 \
    --experiment_name filter_order &
done
