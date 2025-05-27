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
    workers_per_gpu=4,
)
