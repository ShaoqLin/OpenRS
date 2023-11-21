CUDA_VISIBLE_DEVICES=$1 debugpy-run oe_bit.py  --model BiT-M-R50x1 \
--logdir ../log  \
--dataset cifar100 --datadir ../data \
--eval_every 400 --no-save --name $2 --base_lr 0.1 --batch 128 --score $3 \
--oe_batch_size $4
