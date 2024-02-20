# too big learning rate to Nan
CUDA_VISIBLE_DEVICES=2 \
DETECTRON2_DATASETS=/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets \
python tools/train_net.py \
--num-gpus 1 \
--config-file \
/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/opendet2/configs/dota_dior/fornothing.yaml

sleep 60

python /mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/tools/alldone.py
