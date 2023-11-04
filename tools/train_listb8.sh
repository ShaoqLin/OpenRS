python tools/train8.py \
    '/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/configs/dota/benchmark/benchmark_faster-rcnn_1gpu_r50_fpn_2x_lr1e-2_dotav1_0.py' \

sleep 60


python tools/train8.py \
    '/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/configs/dota/benchmark/benchmark_faster-rcnn_1gpu_r50_fpn_2x_lr1e-3_dotav1_0.py' \

sleep 60

python tools/train8.py \
    '/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/configs/dota/benchmark/benchmark_faster-rcnn_1gpu_r50_fpn_2x_lr1e-4_dotav1_0.py' \

sleep 60

python tools/train8.py \
    '/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/configs/dota/benchmark/benchmark_faster-rcnn_1gpu_r50_fpn_2x_lr2e-3_dotav1_0.py' \

sleep 60

python tools/train8.py \
    '/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/configs/dota/benchmark/benchmark_faster-rcnn_1gpu_r50_fpn_2x_lr5e-3_dotav1_0.py' \

sleep 60
python /mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/tools/alldone.py