# printf "\033[1m\033[45;33m 1_ GPU 2_ NAME 3_ LOSS{confidence, TCP, BCE, baseline} "
CUDA_VISIBLE_DEVICES=$1 debugpy-run train_bit.py --model BiT-M-R50x1 --logdir ../log  --dataset cifar100 --datadir ../data --eval_every 400 --no-save --name $2 --base_lr 0.1 --batch 128 --loss $3
# CUDA_VISIBLE_DEVICES=$1 python train_bit.py --model BiT-M-R50x1 --logdir ../log  --dataset cifar100 --datadir ../data --eval_every 400 --no-save --name $2 --base_lr 0.1 --batch 128 --loss $3
