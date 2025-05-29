import numpy as np
import cv2
import matplotlib.pyplot as plt
from mmdet.apis import init_detector, inference_detector
import random
from mmdet.registry import VISUALIZERS

def simulate_proposals(final_bboxes, img_shape, num_proposals=100):
    """
    根据最终bbox模拟生成proposals
    Args:
        final_bboxes: [N, 4]格式的最终预测框（xyxy）
        img_shape: 图片高宽
        num_proposals: 生成的proposal总数
    Returns:
        proposals: [num_proposals, 4]的模拟框
    """
    h, w = img_shape[:2]
    proposals = []
    
    # 1. 基于真实框生成抖动提案（60%）
    for bbox in final_bboxes:
        x1, y1, x2, y2 = bbox
        for _ in range(int(0.6*num_proposals//len(final_bboxes))):
            # 在真实框周围随机偏移
            offset_x = random.uniform(-0.3*(x2-x1), 0.3*(x2-x1))
            offset_y = random.uniform(-0.3*(y2-y1), 0.3*(y2-y1))
            new_box = [
                max(0, x1 + offset_x),
                max(0, y1 + offset_y),
                min(w, x2 + offset_x),
                min(h, y2 + offset_y)
            ]
            proposals.append(new_box)
    
    # 2. 随机噪声提案（40%）
    for _ in range(num_proposals - len(proposals)):
        x1 = random.uniform(0, w-10)
        y1 = random.uniform(0, h-10)
        x2 = random.uniform(x1+10, w)
        y2 = random.uniform(y1+10, h)
        proposals.append([x1, y1, x2, y2])
    
    return np.array(proposals[:num_proposals])

def visualize_comparison(img_path, model,i):
    img = cv2.imread(img_path)
    result = inference_detector(model, img)
    
    # 获取预测框
    pred = result.pred_instances
    final_bboxes = pred.bboxes.cpu().numpy()
    scores = pred.scores.cpu().numpy()
    final_bboxes = final_bboxes[scores > 0.5]

    # 模拟proposals
    simulated_proposals = simulate_proposals(final_bboxes, img.shape)

    # 初始化可视化工具
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    # 绘制最终结果
    result_img = img.copy()
    visualizer.add_datasample(
        'result',
        result_img,
        data_sample=result,
        draw_gt=False,
        show=False
    )
    result_img = visualizer.get_image()

    # 绘制模拟proposals
    proposal_img = img.copy()
    for box in simulated_proposals:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(
            proposal_img,
            (x1, y1), (x2, y2),
            color=(255, 0, 0),
            thickness=1,
            lineType=cv2.LINE_AA
        )

    # # 显示对比
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    # ax1.imshow(cv2.cvtColor(proposal_img, cv2.COLOR_BGR2RGB))
    # ax1.set_title('Simulated Proposals')
    # ax2.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    # ax2.set_title('Final Prediction')
    # plt.savefig(f'comparison_{i}.png')
    # plt.close()
    # 假设 proposal_img 和 result_img 都是 BGR 格式的 OpenCV 图像
    # 确保它们的高度一致
    if proposal_img.shape[0] != result_img.shape[0]:
        height = min(proposal_img.shape[0], result_img.shape[0])
        proposal_img = cv2.resize(proposal_img, (int(proposal_img.shape[1] * height / proposal_img.shape[0]), height))
        result_img = cv2.resize(result_img, (int(result_img.shape[1] * height / result_img.shape[0]), height))

    # 横向拼接（左右拼图）
    combined = np.concatenate((proposal_img, result_img), axis=1)

    # 保存结果（仍为 BGR 格式）
    cv2.imwrite(f'comparison_{i}.png', combined)

# 使用示例
config_file = '/home/rongyi/to_SSD_DISK/projects/mmdetection/configs_voc/mask_rcnn/mask-rcnn_r50_fpn_1x_coco_old.py'
checkpoint_file = '/home/rongyi/to_SSD_DISK/projects/mmdetection/work_dirs/mask-rcnn_r50_fpn_1x_coco/epoch_12.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# 选择VOC测试集中的4张图片
image_paths = [
    '/home/rongyi/to_SSD_DISK/projects/mmdetection/data/data/coco/val2017/VOC2007/JPEGImages/000001.jpg',
    '/home/rongyi/to_SSD_DISK/projects/mmdetection/data/data/coco/val2017/VOC2007/JPEGImages/000105.jpg',
    '/home/rongyi/to_SSD_DISK/projects/mmdetection/data/data/coco/val2017/VOC2007/JPEGImages/000053.jpg',
    '/home/rongyi/to_SSD_DISK/projects/mmdetection/data/data/coco/val2017/VOC2007/JPEGImages/000067.jpg'
]
i=0
for path in image_paths:
    visualize_comparison(path, model,i)
    i+=1