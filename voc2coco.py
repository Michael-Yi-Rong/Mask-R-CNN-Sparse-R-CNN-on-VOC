import os
import json
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np

def convert_voc_to_coco(voc_dir, output_dir, split='trainval', year='2007'):
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化COCO数据结构
    coco = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # 添加类别信息 (VOC有20个类别)
    classes = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    
    for i, cls in enumerate(classes, 1):
        coco["categories"].append({
            "id": i,
            "name": cls,
            "supercategory": "none"
        })
    
    # 读取VOC图像列表
    with open(os.path.join(voc_dir, f'VOC{year}/ImageSets/Main/{split}.txt'), 'r') as f:
        image_names = f.read().strip().split()
    
    annotation_id = 1
    
    for img_id, img_name in enumerate(image_names, 1):
        # 图像信息
        img_path = os.path.join(voc_dir, f'VOC{year}/JPEGImages/{img_name}.jpg')
        img = Image.open(img_path)
        width, height = img.size
        
        coco["images"].append({
            "id": img_id,
            "file_name": f"{img_name}.jpg",
            "width": width,
            "height": height
        })
        
        # 标注信息
        ann_path = os.path.join(voc_dir, f'VOC{year}/Annotations/{img_name}.xml')
        tree = ET.parse(ann_path)
        root = tree.getroot()
        
        for obj in root.findall('object'):
            cls = obj.find('name').text
            category_id = classes.index(cls) + 1
            
            # 处理bbox
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            width = xmax - xmin
            height = ymax - ymin
            
            # 处理segmentation (VOC的segmentation是polygon格式)
            segmentation = []
            if obj.find('segmented').text == '1':
                seg_obj = obj.find('segm')
                if seg_obj is not None:
                    for poly in seg_obj.findall('polygon'):
                        segmentation.append([float(p) for p in poly.text.split()])
            
            # 添加到COCO标注
            coco["annotations"].append({
                "id": annotation_id,
                "image_id": img_id,
                "category_id": category_id,
                "bbox": [xmin, ymin, width, height],
                "area": width * height,
                "segmentation": segmentation,
                "iscrowd": 0
            })
            annotation_id += 1
    
    # 保存COCO格式的JSON文件
    output_file = os.path.join(output_dir, f'voc{year}_{split}_coco.json')
    with open(output_file, 'w') as f:
        json.dump(coco, f)
    
    print(f"转换完成，结果保存在: {output_file}")

# 转换VOC2007和VOC2012
convert_voc_to_coco('data/VOCdevkit', 'coco_annotations', 'trainval', '2007')
convert_voc_to_coco('data/VOCdevkit', 'coco_annotations', 'trainval', '2012')
