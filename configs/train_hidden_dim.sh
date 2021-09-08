#!/bin/bash
# 64 128 256 512 1024 2048 4096

for hdim in 64 128 256 512 1024 2048 4096
do
    CUDA_VISIBLE_DEVICES=0 python train.py \
    --gpus 1 \
    --model_order 16 \
    --max_train_order 16 \
    --batch_size 128 \
    --num_workers 24 \
    --lr 2e-5 \
    --gradient_clip_val 0.5 \
    --gradient_clip_algorithm norm \
    --hidden_dim $hdim \
    --shuffle \
    --filter_method all \
    --max_epochs 200 \
    --num_train_examples 50000 \
    --track_grad_norm 2 &
done
