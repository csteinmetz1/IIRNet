#!/bin/bash

for filter_order in 4 8 16 32 64 128
do
    CUDA_VISIBLE_DEVICES=0 python train.py \
    --gpus 1 \
    --model_order $filter_order \
    --max_train_order $filter_order \
    --batch_size 128 \
    --num_workers 8 \
    --lr 2e-4 \
    --gradient_clip_val 1.0 \
    --hidden_dim 1024 \
    --shuffle \
    --filter_method all \
    --max_epochs 200 \
    --num_train_examples 100000
done
