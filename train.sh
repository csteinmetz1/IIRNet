CUDA_VISIBLE_DEVICES=0 python train.py \
--gpus 1 \
--max_order 4 \
--batch_size 32 \
--num_workers 16 \
--lr 1e-6 \
--gradient_clip_val 0.5 \
--hidden_dim 2048 \
--shuffle \
--precision 16 \
