CUDA_VISIBLE_DEVICES=2 \
DETECTRON2_DATASETS=/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets \
/home/dell/anaconda3/envs/lsqopen/bin/python tools/train_net_with_val.py \
--resume \
--num-gpus 1 \
--config-file \
/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/opendet2/configs/opendet_cfrpn/faster_rcnn_R_50_FPN_2x_5e-3_bs16_opendet_cfrpn_fair1m-mar20_crop1024_10_10.yaml
sleep 60

CUDA_VISIBLE_DEVICES=2 \
DETECTRON2_DATASETS=/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets \
/home/dell/anaconda3/envs/lsqopen/bin/python tools/train_net_with_val.py \
--resume \
--num-gpus 1 \
--config-file \
/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/opendet2/configs/opendet_cfrpn/faster_rcnn_R_50_FPN_2x_1e-3_bs16_opendet_cfrpn-threst-2-4_fair1m-mar20_crop1024_10_10.yaml

sleep 60

CUDA_VISIBLE_DEVICES=2 \
DETECTRON2_DATASETS=/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets \
/home/dell/anaconda3/envs/lsqopen/bin/python tools/train_net_with_val.py \
--resume \
--num-gpus 1 \
--config-file \
/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/opendet2/configs/opendet_cfrpn/faster_rcnn_R_50_FPN_2x_1e-3_bs16_opendet_cfrpn-threst-3-5_fair1m-mar20_crop1024_10_10.yaml

sleep 60

CUDA_VISIBLE_DEVICES=2 \
DETECTRON2_DATASETS=/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets \
/home/dell/anaconda3/envs/lsqopen/bin/python tools/train_net_with_val.py \
--resume \
--num-gpus 1 \
--config-file \
/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/opendet2/configs/opendet_cfrpn/faster_rcnn_R_50_FPN_2x_5e-3_bs16_opendet_cfrpn-threst-2-4_fair1m-mar20_crop1024_10_10.yaml

sleep 60

CUDA_VISIBLE_DEVICES=2 \
DETECTRON2_DATASETS=/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets \
/home/dell/anaconda3/envs/lsqopen/bin/python tools/train_net_with_val.py \
--resume \
--num-gpus 1 \
--config-file \
/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/opendet2/configs/opendet_cfrpn/faster_rcnn_R_50_FPN_2x_5e-3_bs16_opendet_cfrpn-threst-3-5_fair1m-mar20_crop1024_10_10.yaml

sleep 60

python /mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/tools/alldone.py