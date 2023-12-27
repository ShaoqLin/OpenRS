#!/bin/bash
set -e

CUDA_VISIBLE_DEVICES=1 \
DETECTRON2_DATASETS=/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets \
python tools/train_net_with_val.py \
--resume \
--num-gpus 1 \
--dist-url='tcp://127.0.0.1:50155' \
--config-file \
/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/opendet2/configs/fair1m_mar20/faster_rcnn_R_50_FPN_1x_1e-3_bs16_baseline_fair1m-mar20_crop1024_10_10.yaml \
DATASETS.VAL 'fair1m_val1024_airplane'
sleep 60

CUDA_VISIBLE_DEVICES=1 \
DETECTRON2_DATASETS=/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets \
python tools/train_net_with_val.py \
--resume \
--num-gpus 1 \
--dist-url='tcp://127.0.0.1:50155' \
--config-file \
/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/opendet2/configs/fair1m_mar20/faster_rcnn_R_50_FPN_1x_1e-3_bs16_opendet_fair1m-mar20_crop1024_10_10.yaml \
DATASETS.VAL 'fair1m_val1024_airplane'
sleep 60

CUDA_VISIBLE_DEVICES=1 \
DETECTRON2_DATASETS=/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets \
python tools/train_net_with_val.py \
--resume \
--num-gpus 1 \
--dist-url='tcp://127.0.0.1:50155' \
--config-file \
/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/opendet2/configs/fair1m_mar20/faster_rcnn_R_50_FPN_1x_1e-3_bs16_opendet_gmm_fair1m-mar20_crop1024_10_10.yaml \
DATASETS.VAL 'fair1m_val1024_airplane'
sleep 60

CUDA_VISIBLE_DEVICES=1 \
DETECTRON2_DATASETS=/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets \
python tools/train_net_with_val.py \
--resume \
--num-gpus 1 \
--dist-url='tcp://127.0.0.1:50155' \
--config-file \
/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/opendet2/configs/fair1m_mar20/faster_rcnn_R_50_FPN_1x_1e-3_bs16_opendet_siren_fair1m-mar20_crop1024_10_10.yaml \
DATASETS.VAL 'fair1m_val1024_airplane'
sleep 60

CUDA_VISIBLE_DEVICES=1 \
DETECTRON2_DATASETS=/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets \
python tools/train_net_with_val.py \
--resume \
--num-gpus 1 \
--dist-url='tcp://127.0.0.1:50155' \
--config-file \
/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/opendet2/configs/fair1m_mar20/faster_rcnn_R_50_FPN_2x_1e-3_bs16_baseline_fair1m-mar20_crop1024_10_10.yaml \
DATASETS.VAL 'fair1m_val1024_airplane'
sleep 60

CUDA_VISIBLE_DEVICES=1 \
DETECTRON2_DATASETS=/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets \
python tools/train_net_with_val.py \
--resume \
--num-gpus 1 \
--dist-url='tcp://127.0.0.1:50155' \
--config-file \
/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/opendet2/configs/fair1m_mar20/faster_rcnn_R_50_FPN_2x_1e-3_bs16_opendet_fair1m-mar20_crop1024_10_10.yaml \
DATASETS.VAL 'fair1m_val1024_airplane'
sleep 60

CUDA_VISIBLE_DEVICES=1 \
DETECTRON2_DATASETS=/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets \
python tools/train_net_with_val.py \
--resume \
--num-gpus 1 \
--dist-url='tcp://127.0.0.1:50155' \
--config-file \
/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/opendet2/configs/fair1m_mar20/faster_rcnn_R_50_FPN_2x_1e-3_bs16_opendet_gmm_fair1m-mar20_crop1024_10_10.yaml \
DATASETS.VAL 'fair1m_val1024_airplane'
sleep 60

CUDA_VISIBLE_DEVICES=1 \
DETECTRON2_DATASETS=/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets \
python tools/train_net_with_val.py \
--resume \
--num-gpus 1 \
--dist-url='tcp://127.0.0.1:50155' \
--config-file \
/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/opendet2/configs/fair1m_mar20/faster_rcnn_R_50_FPN_2x_1e-3_bs16_opendet_siren_fair1m-mar20_crop1024_10_10.yaml \
DATASETS.VAL 'fair1m_val1024_airplane'
sleep 60

# too big learning rate to Nan
CUDA_VISIBLE_DEVICES=1 \
DETECTRON2_DATASETS=/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets \
python tools/train_net_with_val.py \
--resume \
--num-gpus 1 \
--dist-url='tcp://127.0.0.1:50155' \
--config-file \
/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/opendet2/configs/fair1m_mar20/faster_rcnn_R_50_FPN_1x_1e-4_bs16_baseline_fair1m-mar20_crop1024_10_10.yaml \
DATASETS.VAL 'fair1m_val1024_airplane'
sleep 60

CUDA_VISIBLE_DEVICES=1 \
DETECTRON2_DATASETS=/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets \
python tools/train_net_with_val.py \
--resume \
--num-gpus 1 \
--dist-url='tcp://127.0.0.1:50155' \
--config-file \
/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/opendet2/configs/fair1m_mar20/faster_rcnn_R_50_FPN_1x_1e-4_bs16_opendet_fair1m-mar20_crop1024_10_10.yaml \
DATASETS.VAL 'fair1m_val1024_airplane'
sleep 60

CUDA_VISIBLE_DEVICES=1 \
DETECTRON2_DATASETS=/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets \
python tools/train_net_with_val.py \
--resume \
--num-gpus 1 \
--dist-url='tcp://127.0.0.1:50155' \
--config-file \
/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/opendet2/configs/fair1m_mar20/faster_rcnn_R_50_FPN_1x_1e-4_bs16_opendet_gmm_fair1m-mar20_crop1024_10_10.yaml \
DATASETS.VAL 'fair1m_val1024_airplane'
sleep 60

CUDA_VISIBLE_DEVICES=1 \
DETECTRON2_DATASETS=/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets \
python tools/train_net_with_val.py \
--resume \
--num-gpus 1 \
--dist-url='tcp://127.0.0.1:50155' \
--config-file \
/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/opendet2/configs/fair1m_mar20/faster_rcnn_R_50_FPN_1x_1e-4_bs16_opendet_siren_fair1m-mar20_crop1024_10_10.yaml \
DATASETS.VAL 'fair1m_val1024_airplane'
sleep 60

CUDA_VISIBLE_DEVICES=1 \
DETECTRON2_DATASETS=/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets \
python tools/train_net_with_val.py \
--resume \
--num-gpus 1 \
--dist-url='tcp://127.0.0.1:50155' \
--config-file \
/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/opendet2/configs/fair1m_mar20/faster_rcnn_R_50_FPN_2x_1e-4_bs16_baseline_fair1m-mar20_crop1024_10_10.yaml \
DATASETS.VAL 'fair1m_val1024_airplane'
sleep 60

CUDA_VISIBLE_DEVICES=1 \
DETECTRON2_DATASETS=/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets \
python tools/train_net_with_val.py \
--resume \
--num-gpus 1 \
--dist-url='tcp://127.0.0.1:50155' \
--config-file \
/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/opendet2/configs/fair1m_mar20/faster_rcnn_R_50_FPN_2x_1e-4_bs16_opendet_fair1m-mar20_crop1024_10_10.yaml \
DATASETS.VAL 'fair1m_val1024_airplane'
sleep 60

CUDA_VISIBLE_DEVICES=1 \
DETECTRON2_DATASETS=/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets \
python tools/train_net_with_val.py \
--resume \
--num-gpus 1 \
--dist-url='tcp://127.0.0.1:50155' \
--config-file \
/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/opendet2/configs/fair1m_mar20/faster_rcnn_R_50_FPN_2x_1e-4_bs16_opendet_gmm_fair1m-mar20_crop1024_10_10.yaml \
DATASETS.VAL 'fair1m_val1024_airplane'
sleep 60

CUDA_VISIBLE_DEVICES=1 \
DETECTRON2_DATASETS=/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets \
python tools/train_net_with_val.py \
--resume \
--num-gpus 1 \
--dist-url='tcp://127.0.0.1:50155' \
--config-file \
/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/opendet2/configs/fair1m_mar20/faster_rcnn_R_50_FPN_2x_1e-4_bs16_opendet_siren_fair1m-mar20_crop1024_10_10.yaml \
DATASETS.VAL 'fair1m_val1024_airplane'
sleep 60

python /mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/tools/alldone.py

