#!/bin/bash

for filter_method in all normal_poly normal_biquad uniform_disk uniform_mag_disk char_poly uniform_parametric
do
    CUDA_VISIBLE_DEVICES=1 python train.py \
    --gpus 1 \
    --model_order 32 \
    --max_train_order 32 \
    --batch_size 128 \
    --num_workers 8 \
    --lr 2e-4 \
    --gradient_clip_val 1.0 \
    --hidden_dim 1024 \
    --shuffle \
    --filter_method $filter_method \
    --max_epochs 200 \
    --num_train_examples 100000
done