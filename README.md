# Mask-R-CNN-Sparse-R-CNN-on-VOC

VOC数据集训练测试Mask R-CNN和Sparse R-CNN，实现目标检测与语义分割

## 项目结构

```
Mask-R-CNN-Sparse-R-CNN-on-VOC/
├── data/                        # 数据集目录
│   ├── VOC2007/                 # VOC2007 数据集
│   └── VOC2012/                 # VOC2012 数据集
│
├── configs_voc/                 # 配置文件目录
│   ├── _base_                   # 基础配置文件
│   ├── mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py
│   └── sparse_rcnn/sparse-rcnn_r50_fpn_1x_coco.py
│
├── tools                       # 训练测试工具
│   ├── train.py
│   ├── test.py
│   ├── dist_train.sh
│   └── dist_test.sh
│
├── workdirs/                     # 输出目录
│   └── ...
│
├── Other MMDetection Files & Folders...
│
├── simulate_proposals.py        # 展示Proposal脚本
├── inference.py                 # 推理脚本
├── voc2coco.py                  # VOC2COCO脚本
├── requirements.txt             # 环境配置
└── README.md                    # 项目说明
```

## 环境要求

pip install requirements.txt


## 数据集准备

1. 下载 [VOC 2007/2012](https://data.caltech.edu/records/mzrjq-6wc02](http://host.robots.ox.ac.uk/pascal/VOC/ ) 数据集
2. 解压数据集到 `data/` 目录下
3. 确保数据路径符合以下结构:
   ```
    VOCdevkit/                 # 数据集根目录
    └── VOC2007/               # 数据集年份标识（VOC2007或VOC2012）
        ├── Annotations/       # 目标检测标注文件（XML格式）
        ├── ImageSets/         # 数据集划分文件
        │   ├── Layout/        # 人体布局任务划分（train/val等）
        │   ├── Main/          # 主分类任务划分（包含train.txt等）
        │   └── Segmentation/  # 分割任务划分
        ├── JPEGImages/        # 原始图像文件（.jpg格式）
        ├── SegmentationClass/ # 语义分割类别标注（PNG图像）
        └── SegmentationObject/# 实例分割标注（PNG图像）
   ```
4. 运行 `python voc2coco.py` 进行数据格式转换

## 模型训练

### 配置文件

模型训练参数通过YAML配置文件定义，位于`configs_voc/`目录：
- `mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py`: Mask R-CNN配置文件
- `sparse_rcnn/sparse-rcnn_r50_fpn_1x_coco.py`: Sparse R-CNN配置文件

### 训练Shell脚本
具体见`train.sh`
```bash
bash train.sh
```
- 单GPU

```bash
# single gpu training
python tools/train.py configs_voc/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py
```
```bash
python tools/train.py configs_voc/sparse_rcnn/sparse-rcnn_r50_fpn_1x_coco.py
```

- 分布式训练(DDP)

```bash
# distributed training
bash tools/dist_train.sh \
  configs_voc/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py \
  8
```
```bash
# distributed training
bash tools/dist_train.sh \
  configs_voc/sparse_rcnn/sparse-rcnn_r50_fpn_1x_coco.py \
  8
```

## 模型测试

- 单GPU

```bash
python tools/test.py configs_voc/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py
python tools/test.py configs_voc/sparse_rcnn/sparse-rcnn_r50_fpn_1x_coco.py work_dirs/mask-rcnn_r50_fpn_1x_coco/epoch_50.pth \
  --out work_dirs/results.pkl --show-dir work_dirs/show_results/ \
```

```bash
bash tools/dist_test.sh \
  configs_voc/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py \
  8
```

- 分布式训练(DDP)

```bash
bash tools/dist_test.sh \
   configs_voc/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py \
   work_dirs/mask-rcnn_r50_fpn_1x_coco/epoch_50.pth \
   8 \
   --out work_dirs/results.pkl \
   --show-dir work_dirs/show_results/
```

## 可视化

训练过程中的损失曲线和准确率可以通过TensorBoard查看:

```bash
tensorboard --logdir workdirs/.../logs --port=6006
```
## 推理过程

```bash
python inference.py
```

## 训练模型

本项目的训练模型已上传至以下链接:

https://pan.baidu.com/s/1gOaJODKnvjeBesK4QI2YLw?pwd=8848 
