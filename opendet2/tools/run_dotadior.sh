# # too big learning rate to Nan
# CUDA_VISIBLE_DEVICES=2 \
# DETECTRON2_DATASETS=/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets \
# python tools/train_net.py \
# --resume \
# --num-gpus 1 \
# --config-file \
# /mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/opendet2/configs/dota_dior_10_14/faster_rcnn_R_50_FPN_2x_1e-3_bs16_baseline_dota-dior_10_14.yaml

# sleep 10

CUDA_VISIBLE_DEVICES=2 \
DETECTRON2_DATASETS=/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets \
python tools/train_net.py \
--resume \
--num-gpus 1 \
--config-file \
/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/opendet2/configs/dota_dior_10_14/faster_rcnn_R_50_FPN_2x_1e-3_bs16_ds_dota-dior_10_14.yaml

sleep 10

CUDA_VISIBLE_DEVICES=2 \
DETECTRON2_DATASETS=/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets \
python tools/train_net.py \
--resume \
--num-gpus 1 \
--config-file \
/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/opendet2/configs/dota_dior_10_14/faster_rcnn_R_50_FPN_2x_1e-3_bs16_opendet_dota-dior_10_14.yaml

sleep 10

CUDA_VISIBLE_DEVICES=2 \
DETECTRON2_DATASETS=/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets \
python tools/train_net.py \
--resume \
--num-gpus 1 \
--config-file \
/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/opendet2/configs/dota_dior_10_14/faster_rcnn_R_50_FPN_2x_1e-3_bs16_proser_dota-dior_10_14.yaml

sleep 10

CUDA_VISIBLE_DEVICES=2 \
DETECTRON2_DATASETS=/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets \
python tools/train_net.py \
--resume \
--num-gpus 1 \
--config-file \
/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/opendet2/configs/dota_dior_10_14/faster_rcnn_R_50_FPN_2x_1e-3_bs16_opendet_siren_dota-dior_10_14.yaml

sleep 10

python /mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/tools/alldone.py
