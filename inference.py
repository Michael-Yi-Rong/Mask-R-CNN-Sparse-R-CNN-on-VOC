import os
import cv2
import mmcv
from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS

def inference_and_save(
    config_file: str,
    checkpoint_file: str,
    img_path: str,
    out_path: str,
    score_thr: float = 0.3,
    device: str = 'cuda:0'
):
    """
    单张图片推理、可视化（bbox + mask + label + score）并保存。

    Args:
        config_file (str): 模型 config 路径
        checkpoint_file (str): 模型权重路径
        img_path (str): 输入图片
        out_path (str): 可视化结果保存路径
        score_thr (float): 置信度阈值
        device (str): 运行设备
    """
    # 1. 初始化模型
    model = init_detector(config_file, checkpoint_file, device=device)

    # 2. 构建 Visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)  # 构造可视化后端:contentReference[oaicite:0]{index=0}
    visualizer.dataset_meta = model.dataset_meta      # 传入类名和调色板信息

    # 3. 读图并推理
    img = mmcv.imread(img_path, channel_order='bgr')
    result = inference_detector(model, img)

    # 4. 转为 RGB 并绘制
    img_rgb = mmcv.imconvert(img, 'bgr', 'rgb')
    visualizer.add_datasample(
        name='result',
        image=img_rgb,
        data_sample=result,
        draw_gt=False,
        pred_score_thr=score_thr,
        show=False
    )

    # 5. 获取可视化图并保存
    vis_img = visualizer.get_image()               # RGB 格式图像
    vis_bgr = mmcv.imconvert(vis_img, 'rgb', 'bgr')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, vis_bgr)
    print(f"[✅] 可视化结果已保存到 {out_path}")

if __name__ == '__main__':
    config_file     = 'configs_voc/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py'
    checkpoint_file = 'work_dirs/mask-rcnn_r50_fpn_1x_coco/epoch_12.pth'
    img_path        = 'demo_pics/demo3.jpeg'
    out_path        = 'demo_pics/demo_vis3.jpg'

    inference_and_save(
        config_file,
        checkpoint_file,
        img_path,
        out_path,
        score_thr=0.3,
        device='cuda:0'
    )
