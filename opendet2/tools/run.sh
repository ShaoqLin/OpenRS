# too big learning rate to Nan
CUDA_VISIBLE_DEVICES=2 \
DETECTRON2_DATASETS=/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets \
python tools/train_net.py \
--resume \
--num-gpus 1 \
--config-file \
/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/opendet2/configs/dota_dior_3_7_siren/faster_rcnn_R_50_FPN_3x_1e-1_bs8_opendet_siren_dota-dior37.yaml

sleep 60

CUDA_VISIBLE_DEVICES=2 \
DETECTRON2_DATASETS=/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets \
python tools/train_net.py \
--resume \
--num-gpus 1 \
--config-file \
/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/opendet2/configs/dota_dior_3_7_siren/faster_rcnn_R_50_FPN_3x_1e-1_bs16_opendet_siren_dota-dior37.yaml

sleep 60

CUDA_VISIBLE_DEVICES=2 \
DETECTRON2_DATASETS=/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets \
python tools/train_net.py \
--resume \
--num-gpus 1 \
--config-file \
/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/opendet2/configs/dota_dior_3_7_siren/faster_rcnn_R_50_FPN_3x_1e-2_bs8_opendet_siren_dota-dior37.yaml

sleep 60

CUDA_VISIBLE_DEVICES=2 \
DETECTRON2_DATASETS=/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets \
python tools/train_net.py \
--resume \
--num-gpus 1 \
--config-file \
/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/opendet2/configs/dota_dior_3_7_siren/faster_rcnn_R_50_FPN_3x_1e-2_bs16_opendet_siren_dota-dior37.yaml

sleep 60

CUDA_VISIBLE_DEVICES=2 \
DETECTRON2_DATASETS=/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets \
python tools/train_net.py \
--resume \
--num-gpus 1 \
--config-file \
/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/opendet2/configs/dota_dior_3_7_siren/faster_rcnn_R_50_FPN_3x_1e-3_bs8_opendet_siren_dota-dior37.yaml

sleep 60

CUDA_VISIBLE_DEVICES=2 \
DETECTRON2_DATASETS=/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets \
python tools/train_net.py \
--resume \
--num-gpus 1 \
--config-file \
/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/opendet2/configs/dota_dior_3_7_siren/faster_rcnn_R_50_FPN_3x_1e-3_bs16_opendet_siren_dota-dior37.yaml

sleep 60

CUDA_VISIBLE_DEVICES=2 \
DETECTRON2_DATASETS=/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets \
python tools/train_net.py \
--resume \
--num-gpus 1 \
--config-file \
/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/opendet2/configs/dota_dior_3_7_siren/faster_rcnn_R_50_FPN_3x_1e-4_bs16_opendet_siren_dota-dior37.yaml

sleep 60

CUDA_VISIBLE_DEVICES=2 \
DETECTRON2_DATASETS=/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets \
python tools/train_net.py \
--resume \
--num-gpus 1 \
--config-file \
/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/opendet2/configs/dota_dior_3_7_siren/faster_rcnn_R_50_FPN_3x_2e-2_bs8_opendet_siren_dota-dior37.yaml

sleep 60

CUDA_VISIBLE_DEVICES=2 \
DETECTRON2_DATASETS=/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets \
python tools/train_net.py \
--resume \
--num-gpus 1 \
--config-file \
/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/opendet2/configs/dota_dior_3_7_siren/faster_rcnn_R_50_FPN_3x_2e-2_bs16_opendet_siren_dota-dior37.yaml

sleep 60

CUDA_VISIBLE_DEVICES=2 \
DETECTRON2_DATASETS=/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets \
python tools/train_net.py \
--resume \
--num-gpus 1 \
--config-file \
/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/opendet2/configs/dota_dior_3_7_siren/faster_rcnn_R_50_FPN_3x_2e-3_bs8_opendet_siren_dota-dior37.yaml

sleep 60

CUDA_VISIBLE_DEVICES=2 \
DETECTRON2_DATASETS=/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets \
python tools/train_net.py \
--resume \
--num-gpus 1 \
--config-file \
/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/opendet2/configs/dota_dior_3_7_siren/faster_rcnn_R_50_FPN_3x_2e-3_bs16_opendet_siren_dota-dior37.yaml

sleep 60

CUDA_VISIBLE_DEVICES=2 \
DETECTRON2_DATASETS=/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets \
python tools/train_net.py \
--resume \
--num-gpus 1 \
--config-file \
/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/opendet2/configs/dota_dior_3_7_siren/faster_rcnn_R_50_FPN_3x_3e-3_bs8_opendet_siren_dota-dior37.yaml

sleep 60
CUDA_VISIBLE_DEVICES=2 \
DETECTRON2_DATASETS=/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets \
python tools/train_net.py \
--resume \
--num-gpus 1 \
--config-file \
/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/opendet2/configs/dota_dior_3_7_siren/faster_rcnn_R_50_FPN_3x_3e-3_bs16_opendet_siren_dota-dior37.yaml

sleep 60

CUDA_VISIBLE_DEVICES=2 \
DETECTRON2_DATASETS=/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets \
python tools/train_net.py \
--resume \
--num-gpus 1 \
--config-file \
/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/opendet2/configs/dota_dior_3_7_siren/faster_rcnn_R_50_FPN_3x_4e-3_bs8_opendet_siren_dota-dior37.yaml

sleep 60

CUDA_VISIBLE_DEVICES=2 \
DETECTRON2_DATASETS=/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets \
python tools/train_net.py \
--resume \
--num-gpus 1 \
--config-file \
/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/opendet2/configs/dota_dior_3_7_siren/faster_rcnn_R_50_FPN_3x_4e-3_bs16_opendet_siren_dota-dior37.yaml

sleep 60

CUDA_VISIBLE_DEVICES=2 \
DETECTRON2_DATASETS=/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets \
python tools/train_net.py \
--resume \
--num-gpus 1 \
--config-file \
/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/opendet2/configs/dota_dior_3_7_siren/faster_rcnn_R_50_FPN_3x_5e-2_bs8_opendet_siren_dota-dior37.yaml

sleep 60

CUDA_VISIBLE_DEVICES=2 \
DETECTRON2_DATASETS=/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets \
python tools/train_net.py \
--resume \
--num-gpus 1 \
--config-file \
/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/opendet2/configs/dota_dior_3_7_siren/faster_rcnn_R_50_FPN_3x_5e-2_bs16_opendet_siren_dota-dior37.yaml

sleep 60

CUDA_VISIBLE_DEVICES=2 \
DETECTRON2_DATASETS=/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets \
python tools/train_net.py \
--resume \
--num-gpus 1 \
--config-file \
/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/opendet2/configs/dota_dior_3_7_siren/faster_rcnn_R_50_FPN_3x_5e-3_bs8_opendet_siren_dota-dior37.yaml

sleep 60

CUDA_VISIBLE_DEVICES=2 \
DETECTRON2_DATASETS=/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets \
python tools/train_net.py \
--resume \
--num-gpus 1 \
--config-file \
/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/opendet2/configs/dota_dior_3_7_siren/faster_rcnn_R_50_FPN_3x_5e-3_bs16_opendet_siren_dota-dior37.yaml

sleep 60

python /mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/tools/alldone.py

# CUDA_VISIBLE_DEVICES=2 \
# DETECTRON2_DATASETS=/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets \
# python tools/train_net.py \
# --resume \
# --num-gpus 1 \
# --config-file \
# /mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/opendet2/configs/dota_dior/faster_rcnn_R_50_FPN_3x_5e-2_bs8_opendet_dota-dior37.yaml
