_base_ = [
    './benchmark_faster-rcnn_r50_fpn.py',
    '../../_base_/datasets/DOTA.py',
    '../../_base_/schedules/schedule_1x.py', '../../_base_/default_runtime.py'
]
model = dict(roi_head=dict(bbox_head=dict(num_classes=15)))

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=0.0001))

# dataset
train_dataloader = dict(
    batch_size=8,
    num_workers=16,
)

# scheduler
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=24,
        by_epoch=True,
        milestones=[16, 22],
        gamma=0.1)
]