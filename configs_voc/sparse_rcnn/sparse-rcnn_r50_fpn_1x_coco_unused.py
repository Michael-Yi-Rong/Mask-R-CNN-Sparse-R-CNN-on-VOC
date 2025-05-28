# --------------------------------------------------------
# 基础配置（继承各模块默认设置）
# --------------------------------------------------------
_base_ = [
    '../_base_/datasets/coco_detection.py',  # COCO 检测格式
    '../_base_/schedules/schedule_1x.py',    # 默认 12 epochs 训练计划
    '../_base_/default_runtime.py'           # 默认运行时设置
]

# --------------------------------------------------------
# 模型超参数
# --------------------------------------------------------
num_stages = 6
num_proposals = 100

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

DATASET_TYPE = 'CocoDataset'
DATA_ROOT = 'data/coco/'

# --------------------------------------------------------
# 数据加载器（DataLoader）配置
# --------------------------------------------------------
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
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
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', scale=(1333, 800), keep_ratio=True),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PackDetInputs')
        ]
    )
)

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
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='PackDetInputs')
        ]
    )
)

# 测试集复用验证集配置
test_dataloader = val_dataloader

# --------------------------------------------------------
# 评估器（Evaluator）
# --------------------------------------------------------
val_evaluator = dict(
    type='CocoMetric',
    metric='bbox',
    format_only=False
)
test_evaluator = val_evaluator

# --------------------------------------------------------
# 模型配置：SparseRCNN
# --------------------------------------------------------
model = dict(
    type='SparseRCNN',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32
    ),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_input',
        num_outs=4
    ),
    rpn_head=dict(
        type='EmbeddingRPNHead',
        num_proposals=num_proposals,
        proposal_feature_channel=256
    ),
    roi_head=dict(
        type='SparseRoIHead',
        num_stages=num_stages,
        stage_loss_weights=[1.0] * num_stages,
        proposal_feature_channel=256,
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]
        ),
        bbox_head=[
            dict(
                type='DIIHead',
                num_classes=len(classes),
                num_ffn_fcs=2,
                num_heads=8,
                num_cls_fcs=1,
                num_reg_fcs=3,
                feedforward_channels=2048,
                in_channels=256,
                dropout=0.0,
                ffn_act_cfg=dict(type='ReLU', inplace=True),
                dynamic_conv_cfg=dict(
                    type='DynamicConv',
                    in_channels=256,
                    feat_channels=64,
                    out_channels=256,
                    input_feat_shape=7,
                    act_cfg=dict(type='ReLU', inplace=True),
                    norm_cfg=dict(type='LN')
                ),
                loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0
                ),
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    clip_border=False,
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.5, 0.5, 1.0, 1.0]
                )
            ) for _ in range(num_stages)
        ]
    ),
    train_cfg=dict(
        rpn=None,
        rcnn=[
            dict(
                assigner=dict(
                    type='HungarianAssigner',
                    match_costs=[
                        dict(type='FocalLossCost', weight=2.0),
                        dict(type='BBoxL1Cost', weight=5.0, box_format='xyxy'),
                        dict(type='IoUCost', iou_mode='giou', weight=2.0)
                    ]
                ),
                sampler=dict(type='PseudoSampler'),
                pos_weight=1
            ) for _ in range(num_stages)
        ]
    ),
    test_cfg=dict(
        rpn=None,
        rcnn=dict(max_per_img=num_proposals)
    )
)

# --------------------------------------------------------
# 优化器与梯度裁剪
# --------------------------------------------------------
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=2.5e-05,
        weight_decay=0.0001
    ),
    clip_grad=dict(max_norm=1, norm_type=2)
)

# --------------------------------------------------------
# 训练、验证、测试流程
# --------------------------------------------------------
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# --------------------------------------------------------
# 日志与检查点
# --------------------------------------------------------
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1),
    logger=dict(type='LoggerHook', interval=10)
)
log_processor = dict(type='LogProcessor', by_epoch=True, window_size=50)

# --------------------------------------------------------
# 可视化：TensorBoard
# --------------------------------------------------------
vis_backends = [
    dict(type='TensorboardVisBackend', save_dir='sparsercnn_results/curves')
]
visualizer = dict(type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# --------------------------------------------------------
# 工作目录
# --------------------------------------------------------
work_dir = 'work_dirs/sparse_rcnn'
