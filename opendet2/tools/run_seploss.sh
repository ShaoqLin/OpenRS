#!/bin/bash
set -e

# too big learning rate to Nan
# CUDA_VISIBLE_DEVICES=2 \
# DETECTRON2_DATASETS=/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets \
# /home/dell/anaconda3/envs/lsqopen/bin/python tools/train_net_with_val.py \
# --resume \
# --num-gpus 1 \
# --config-file \
# /mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/opendet2/configs/opendet_seploss/faster_rcnn_R_50_FPN_2x_1e-3_bs16_opendet_seploss_dota-dior_10_14.yaml
# sleep 60

CUDA_VISIBLE_DEVICES=2 \
DETECTRON2_DATASETS=/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets \
/home/dell/anaconda3/envs/lsqopen/bin/python tools/train_net_with_val.py \
--resume \
--num-gpus 1 \
--config-file \
/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/opendet2/configs/opendet_seploss/faster_rcnn_R_50_FPN_2x_1e-3_bs16_opendet_seploss_fair1m-mar20_crop1024_10_10.yaml

sleep 60

# CUDA_VISIBLE_DEVICES=2 \
# DETECTRON2_DATASETS=/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets \
# /home/dell/anaconda3/envs/lsqopen/bin/python tools/train_net_with_val.py \
# --resume \
# --num-gpus 1 \
# --config-file \
# /mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/opendet2/configs/opendet_cfrpn/faster_rcnn_R_50_FPN_2x_1e-3_bs16_opendet_cfrpn_dota-dior_10_14.yaml

# sleep 60

# CUDA_VISIBLE_DEVICES=2 \
# DETECTRON2_DATASETS=/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets \
# /home/dell/anaconda3/envs/lsqopen/bin/python tools/train_net_with_val.py \
# --resume \
# --num-gpus 1 \
# --config-file \
# /mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/opendet2/configs/opendet_cfrpn/faster_rcnn_R_50_FPN_2x_1e-3_bs16_opendet_cfrpn_fair1m-mar20_crop1024_10_10.yaml

# sleep 60

# CUDA_VISIBLE_DEVICES=2 \
# DETECTRON2_DATASETS=/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets \
# /home/dell/anaconda3/envs/lsqopen/bin/python tools/train_net_with_val.py \
# --resume \
# --num-gpus 1 \
# --config-file \
# /mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/opendet2/configs/opendet_cfrpn/faster_rcnn_R_50_FPN_2x_1e-3_bs16_opendet_ic5e-1_cfrpn_dota-dior_10_14.yaml

# sleep 60

# CUDA_VISIBLE_DEVICES=2 \
# DETECTRON2_DATASETS=/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets \
# /home/dell/anaconda3/envs/lsqopen/bin/python tools/train_net_with_val.py \
# --resume \
# --num-gpus 1 \
# --config-file \
# /mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/opendet2/configs/opendet_cfrpn/faster_rcnn_R_50_FPN_2x_1e-3_bs16_opendet_ic5e-1_cfrpn_fair1m-mar20_crop1024_10_10.yaml

# sleep 60

# CUDA_VISIBLE_DEVICES=2 \
# DETECTRON2_DATASETS=/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets \
# /home/dell/anaconda3/envs/lsqopen/bin/python tools/train_net_with_val.py \
# --resume \
# --num-gpus 1 \
# --config-file \
# /mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/opendet2/configs/opendet_seploss_cfrpn/faster_rcnn_R_50_FPN_2x_1e-3_bs16_opendet_cfrpn_seploss_dota-dior_10_14.yaml

# sleep 60

# CUDA_VISIBLE_DEVICES=2 \
# DETECTRON2_DATASETS=/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets \
# /home/dell/anaconda3/envs/lsqopen/bin/python tools/train_net_with_val.py \
# --resume \
# --num-gpus 1 \
# --config-file \
# /mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/opendet2/configs/opendet_seploss_cfrpn/faster_rcnn_R_50_FPN_2x_1e-3_bs16_opendet_cfrpn_seploss_fair1m-mar20_crop1024_10_10.yaml

# sleep 60

/home/dell/anaconda3/envs/lsqopen/bin/python /mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/tools/alldone.py
