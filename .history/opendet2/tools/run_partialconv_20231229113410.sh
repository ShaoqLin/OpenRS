#!/bin/bash
set -e

# too big learning rate to Nan
CUDA_VISIBLE_DEVICES=1 \
DETECTRON2_DATASETS=/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets/detectron2_dataset \
/home/dell/anaconda3/envs/kill_if_u_need/bin/python tools/train_net.py \
--resume \
--num-gpus 1 \
--config-file \
/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/opendet2/configs/faster_rcnn_R_50_FPN_3x_opendet_partial_conv.yaml
sleep 60

CUDA_VISIBLE_DEVICES=1 \
DETECTRON2_DATASETS=/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets/detectron2_dataset \
/home/dell/anaconda3/envs/kill_if_u_need/bin/python tools/train_net.py \
--resume \
--num-gpus 1 \
--config-file \
/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/opendet2/configs/faster_rcnn_R_50_FPN_3x_opendet.yaml
sleep 60

/home/dell/anaconda3/envs/kill_if_u_need/bin/python /mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/tools/alldone.py
