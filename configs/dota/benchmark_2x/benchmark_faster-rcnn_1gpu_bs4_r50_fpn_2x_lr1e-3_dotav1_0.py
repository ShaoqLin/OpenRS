_base_ = [
    './benchmark_faster-rcnn_r50_fpn.py',
    '../../_base_/datasets/DOTA.py',
    '../../_base_/schedules/schedule_2x.py', '../../_base_/default_runtime.py'
]
dataset_type = 'DOTADataset'
data_root = '/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets/DOTA1024/'
backend_args = None
model = dict(roi_head=dict(bbox_head=dict(num_classes=15)))
# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001))

# dataset
train_dataloader = dict(
    batch_size=4,
    num_workers=8,
)

# inference on test dataset and
# format the output results for submission.
# test_pipeline = [
#     dict(type='LoadImageFromFile', backend_args=backend_args),
#     dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
#     dict(
#         type='PackDetInputs',
#         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
#                    'scale_factor'))
# ]
# test_dataloader = dict(
#     batch_size=1,
#     num_workers=2,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         data_prefix=dict(img = data_root + 'test/images'),
#         test_mode=True,
#         pipeline=test_pipeline))
# test_evaluator = dict(
#     type='DOTAMetric',
#     format_only=True,
#     merge_patches=True,
#     outfile_prefix='/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/work_dirs/benchmark_faster-rcnn_1gpu_bs4_r50_fpn_1x_lr1e-3_dotav1_0/output')