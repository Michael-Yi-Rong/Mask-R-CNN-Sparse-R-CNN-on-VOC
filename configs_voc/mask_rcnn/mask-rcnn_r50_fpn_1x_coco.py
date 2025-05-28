_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# 指定要用的 GPU id 列表，比如用 4 张卡 [0,1,2,3]
gpu_ids = range(0, 8)

# 自动根据总 batch size 缩放 lr（base_batch_size = 卡数 × samples_per_gpu）
auto_scale_lr = dict(enable=True, base_batch_size=8)  # 4 cards × 2 imgs/card = 8

# 确保每张卡的样本数（一般不用改，默认 samples_per_gpu=2）
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=8,
)

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=20),
        mask_head=dict(num_classes=20)
    )
)

visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[
        # 启用 TensorBoard 日志记录
        dict(type='TensorboardVisBackend', save_dir='work_dirs/mask-rcnn_r50_fpn_1x_coco/tensorboard'),
        # 可选：同时保留本地日志（如损失曲线、指标等）
        dict(type='LocalVisBackend'),
    ],
    name='visualizer'
)

# 配置 metainfo（用于 bbox 和 segm）
metainfo = dict(
    classes=(
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'),
)

train_dataloader = dict(dataset=dict(metainfo=metainfo))
val_dataloader = dict(dataset=dict(metainfo=metainfo))
test_dataloader = dict(dataset=dict(metainfo=metainfo))
