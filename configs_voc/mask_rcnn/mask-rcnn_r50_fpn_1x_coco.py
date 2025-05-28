# --------------------------------------------------------
# 基础配置（继承各模块默认设置）
# --------------------------------------------------------
_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',   # Mask R-CNN with ResNet-50 + FPN
    '../_base_/datasets/coco_instance.py',     # COCO 数据集格式
    '../_base_/schedules/schedule_2x.py',      # 默认 36 epochs 训练计划
    '../_base_/default_runtime.py'             # 默认运行时设置
]

# --------------------------------------------------------
# 数据集与类别
# --------------------------------------------------------
# Pascal VOC 20 类（已转换为 COCO 格式）
classes = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
    'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
    'sofa', 'train', 'tvmonitor'
]

# 数据集类型及根目录
DATASET_TYPE = 'CocoDataset'
DATA_ROOT = 'data/coco/'

# --------------------------------------------------------
# 数据加载器（DataLoader）配置
# --------------------------------------------------------
# 训练集 DataLoader
train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=DATASET_TYPE,
        data_root=DATA_ROOT,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        metainfo=dict(classes=classes),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(type='RandomResize', scale=[(1333, 480), (1333, 960)], keep_ratio=True),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PackDetInputs')
        ]
    )
)

# 验证集 DataLoader
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=DATASET_TYPE,
        data_root=DATA_ROOT,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        metainfo=dict(classes=classes),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(1333, 800), keep_ratio=True),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(type='PackDetInputs')
        ]
    )
)

# 测试集 DataLoader 复用验证集配置
test_dataloader = val_dataloader

# --------------------------------------------------------
# 评估器（Evaluator）
# --------------------------------------------------------
val_evaluator = dict(type='CocoMetric', metric=['bbox', 'segm'])
test_evaluator = val_evaluator

# --------------------------------------------------------
# 模型头部（ROI Head）类别数修改
# --------------------------------------------------------
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=len(classes)),
        mask_head=dict(num_classes=len(classes))
    )
)

# --------------------------------------------------------
# 优化器与学习率调度
# --------------------------------------------------------
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
)

param_scheduler = [
    # 线性 warmup
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    # 多步下降：在第 28、33 epochs 处学习率乘以 0.1
    dict(type='MultiStepLR', begin=0, end=36, by_epoch=True,
         milestones=[28, 33], gamma=0.1)
]

# --------------------------------------------------------
# 训练、验证、测试流程
# --------------------------------------------------------
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=36, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# --------------------------------------------------------
# 日志与检查点
# --------------------------------------------------------
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1),
    logger=dict(type='LoggerHook', interval=10)
)

log_processor = dict(
    type='LogProcessor',
    by_epoch=True,
    window_size=50
)

# --------------------------------------------------------
# 可视化：TensorBoard
# --------------------------------------------------------
vis_backends = [
    dict(type='TensorboardVisBackend', save_dir='maskrcnn_results/curves')
]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)

# --------------------------------------------------------
# 工作目录
# --------------------------------------------------------
work_dir = 'work_dirs_maskrcnn/mask_rcnn'
