# too big learning rate to Nan
CUDA_VISIBLE_DEVICES=0,3 \
DETECTRON2_DATASETS=/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets \
/home/dell/anaconda3/envs/kill_if_u_need/bin/python tools/train_net.py \
--resume \
--num-gpus 2 \
--config-file \
/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/opendet2/configs/fair1m_mar20/faster_rcnn_R_50_FPN_8x_5e-4_bs8_baseline_fair1m-mar20_crop1024_10_10.yaml \

sleep 60

CUDA_VISIBLE_DEVICES=0,3 \
DETECTRON2_DATASETS=/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets \
/home/dell/anaconda3/envs/kill_if_u_need/bin/python tools/train_net.py \
--resume \
--num-gpus 2 \
--config-file \
/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/opendet2/configs/fair1m_mar20/faster_rcnn_R_50_FPN_8x_5e-4_bs8_opendet_fair1m-mar20_crop1024_10_10.yaml \

sleep 60

CUDA_VISIBLE_DEVICES=0,3 \
DETECTRON2_DATASETS=/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets \
/home/dell/anaconda3/envs/kill_if_u_need/bin/python tools/train_net.py \
--resume \
--num-gpus 2 \
--config-file \
/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/opendet2/configs/fair1m_mar20/faster_rcnn_R_50_FPN_8x_5e-4_bs8_ds_fair1m-mar20_crop1024_10_10.yaml \

sleep 60

CUDA_VISIBLE_DEVICES=0,3 \
DETECTRON2_DATASETS=/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets \
/home/dell/anaconda3/envs/kill_if_u_need/bin/python tools/train_net.py \
--resume \
--num-gpus 2 \
--config-file \
/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/opendet2/configs/fair1m_mar20/faster_rcnn_R_50_FPN_8x_5e-4_bs8_proser_fair1m-mar20_crop1024_10_10.yaml \

sleep 60

python /mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/tools/alldone.py
