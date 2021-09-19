#!/bin/bash

for filter_method in all normal_poly normal_biquad uniform_disk uniform_mag_disk char_poly uniform_parametric
do
    if [ "$filter_method" = "all" ]; then 
        examples=8333
    else
        examples=50000
    fi
    CUDA_VISIBLE_DEVICES=0 python train.py \
    --gpus 1 \
    --model_order 16 \
    --max_train_order 16 \
    --batch_size 128 \
    --num_workers 8 \
    --lr 1e-5 \
    --gradient_clip_val 0.9 \
    --gradient_clip_algorithm norm \
    --hidden_dim 1024 \
    --shuffle \
    --filter_method "$filter_method" \
    --max_epochs 500 \
    --num_train_examples "$examples" \
    --seed 16 \
    --experiment_name filter_method & 
done