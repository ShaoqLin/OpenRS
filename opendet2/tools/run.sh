# # too big learning rate to Nan
CUDA_VISIBLE_DEVICES=0 \
DETECTRON2_DATASETS=/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets \
python tools/train_net.py \
--resume \
--num-gpus 1 \
--config-file \
/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/opendet2/configs/fair1m_mar20/faster_rcnn_R_50_FPN_2x_1e-3_bs16_ds_fair1m-mar20_crop1024_10_10.yaml

CUDA_VISIBLE_DEVICES=0 \
DETECTRON2_DATASETS=/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets \
python tools/train_net.py \
--resume \
--num-gpus 1 \
--config-file \
/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/opendet2/output/faster_rcnn_R_50_FPN_2x_1e-3_bs16_proser_fair1m-mar20_crop1024_10_10

