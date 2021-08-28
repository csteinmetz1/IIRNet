#!/bin/bash

for hdim in 128 256 512 1024 2048 4096 8192
do
    CUDA_VISIBLE_DEVICES=0 python train.py \
    --gpus 1 \
    --model_order 16 \
    --max_train_order 16 \
    --batch_size 128 \
    --num_workers 8 \
    --lr 5e-7 \
    --gradient_clip_val 1.0 \
    --hidden_dim $hdim \
    --shuffle \
    --filter_method all \
    --max_epochs 500 \
    --num_train_examples 100000
done
