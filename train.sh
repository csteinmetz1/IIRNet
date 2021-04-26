python train.py \
--gpus 1 \
--model_order 24 \
--max_train_order 24 \
--batch_size 128 \
--num_workers 8 \
--lr 1e-5 \
--gradient_clip_val 1.0 \
--hidden_dim 512 \
--shuffle \
--max_epochs 3 \
--num_train_examples 100000


python train.py --gpus 1 --model_order 32 --max_train_order 32 --batch_size 128 --num_workers 8 --lr 1e-5 --gradient_clip_val 1.0 --hidden_dim 8192 --shuffle --max_epochs 100 --num_train_examples 100000

