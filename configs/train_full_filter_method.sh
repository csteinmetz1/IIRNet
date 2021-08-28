CUDA_VISIBLE_DEVICES=0 python train.py \
--gpus 1 \
--model_order 32 \
--max_train_order 32 \
--batch_size 128 \
--num_workers 16 \
--lr 5e-7 \
--gradient_clip_val 1.0 \
--hidden_dim 8192 \
--shuffle \
--filter_method normal_poly \
--max_epochs 400 \
--num_train_examples 100000

CUDA_VISIBLE_DEVICES=0 python train.py \
--gpus 1 \
--model_order 32 \
--max_train_order 32 \
--batch_size 128 \
--num_workers 16 \
--lr 5e-7 \
--gradient_clip_val 1.0 \
--hidden_dim 8192 \
--shuffle \
--filter_method normal_biquad \
--max_epochs 400 \
--num_train_examples 100000

CUDA_VISIBLE_DEVICES=0 python train.py \
--gpus 1 \
--model_order 32 \
--max_train_order 32 \
--batch_size 128 \
--num_workers 16 \
--lr 5e-7 \
--gradient_clip_val 1.0 \
--hidden_dim 8192 \
--shuffle \
--filter_method uniform_disk \
--max_epochs 400 \
--num_train_examples 100000

CUDA_VISIBLE_DEVICES=0 python train.py \
--gpus 1 \
--model_order 32 \
--max_train_order 32 \
--batch_size 128 \
--num_workers 16 \
--lr 5e-7 \
--gradient_clip_val 1.0 \
--hidden_dim 8192 \
--shuffle \
--filter_method uniform_mag_disk \
--max_epochs 400 \
--num_train_examples 100000

CUDA_VISIBLE_DEVICES=0 python train.py \
--gpus 1 \
--model_order 32 \
--max_train_order 32 \
--batch_size 128 \
--num_workers 16 \
--lr 5e-7 \
--gradient_clip_val 1.0 \
--hidden_dim 8192 \
--shuffle \
--filter_method char_poly \
--max_epochs 400 \
--num_train_examples 100000

CUDA_VISIBLE_DEVICES=0 python train.py \
--gpus 1 \
--model_order 32 \
--max_train_order 32 \
--batch_size 128 \
--num_workers 16 \
--lr 5e-7 \
--gradient_clip_val 1.0 \
--hidden_dim 8192 \
--shuffle \
--filter_method uniform_parametric \
--max_epochs 400 \
--num_train_examples 100000

CUDA_VISIBLE_DEVICES=0 python train.py \
--gpus 1 \
--model_order 32 \
--max_train_order 32 \
--batch_size 128 \
--num_workers 16 \
--lr 5e-7 \
--gradient_clip_val 1.0 \
--hidden_dim 8192 \
--shuffle \
--filter_method all \
--max_epochs 400 \
--num_train_examples 16667