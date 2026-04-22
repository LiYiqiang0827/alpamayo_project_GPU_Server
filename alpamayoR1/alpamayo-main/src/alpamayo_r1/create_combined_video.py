#!/usr/bin/env python3
"""
创建组合视频
- 上面: 16宫格 (4摄像头 × 4帧)
- 左下: CoC文字 + ADE信息
- 右下: 轨迹图 (新坐标轴范围: X-15~15, Y0~60)
"""
import os
import cv2
import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# 配置
RESULT_DIR = '/data01/vla/data/data_sample_chunk0/infer/01d3588e-bca7-4a18-8e74-c6cfe9e996db/result'
DATA_DIR = '/data01/vla/data/data_sample_chunk0/infer/01d3588e-bca7-4a18-8e74-c6cfe9e996db/data'
CAMERA_IMG_DIR = f'{DATA_DIR}/camera_images'
VIDEO_OUTPUT = f'{RESULT_DIR}/combined_video.mp4'

# 坐标轴范围 (新)
X_MIN, X_MAX = -15, 15  # 横轴 -15m ~ +15m
Y_MIN, Y_MAX = 0, 60    # 纵轴 0m ~ 60m

# 相机顺序和列名
cameras = [
    'camera_cross_left_120fov',
    'camera_front_wide_120fov',
    'camera_cross_right_120fov',
    'camera_front_tele_30fov'
]

cam_cols = {
    'camera_cross_left_120fov': ['cam_left_f0', 'cam_left_f1', 'cam_left_f2', 'cam_left_f3'],
    'camera_front_wide_120fov': ['cam_front_f0', 'cam_front_f1', 'cam_front_f2', 'cam_front_f3'],
    'camera_cross_right_120fov': ['cam_right_f0', 'cam_right_f1', 'cam_right_f2', 'cam_right_f3'],
    'camera_front_tele_30fov': ['cam_tele_f0', 'cam_tele_f1', 'cam_tele_f2', 'cam_tele_f3'],
}

def rotate_90cc(xy):
    """逆时针旋转90度"""
    return np.stack([-xy[1], xy[0]], axis=0)

def create_16_grid(row, data_dir):
    """创建16宫格图像 (4×4)"""
    # 每张图尺寸
    img_w, img_h = 480, 270  # 16:9比例
    
    # 16宫格总尺寸 (4×4)
    grid_w = img_w * 4
    grid_h = img_h * 4
    
    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    
    for cam_idx, cam_name in enumerate(cameras):
        for frame_idx, col in enumerate(cam_cols[cam_name]):
            img_path = f'{data_dir}/{row[col]}'
            img = cv2.imread(img_path)
            if img is None:
                # 如果读取失败，用灰色填充
                img = np.full((img_h, img_w, 3), 128, dtype=np.uint8)
            else:
                img = cv2.resize(img, (img_w, img_h))
            
            # 放置在16宫格中
            y_start = cam_idx * img_h
            x_start = frame_idx * img_w
            grid[y_start:y_start+img_h, x_start:x_start+img_w] = img
    
    return grid

def create_trajectory_plot(infer_idx, result_dir, data_dir):
    """创建轨迹图 (新坐标轴范围)"""
    # 加载预测和真值
    pred_path = f'{result_dir}/pred_{infer_idx:06d}.npy'
    pred_xyz = np.load(pred_path)
    if pred_xyz.ndim == 2:
        pred_xyz = pred_xyz[np.newaxis, :, :]
    
    # 加载真值
    ego_file = f'{data_dir}/egomotion/ego_{infer_idx:06d}_future_gt.npy'
    future = np.load(ego_file, allow_pickle=True).item()
    gt_xy = future['xyz'][:, :2]
    
    # 画图
    plt.figure(figsize=(10, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, pred_xyz.shape[0]))
    for i in range(pred_xyz.shape[0]):
        pred_xy = pred_xyz[i].T
        pred_xy_rot = rotate_90cc(pred_xy)
        plt.plot(pred_xy_rot[0], pred_xy_rot[1], "o-", 
                color=colors[i], label=f"Pred #{i+1}", markersize=2, linewidth=1.5)
    
    gt_xy_rot = rotate_90cc(gt_xy.T)
    plt.plot(gt_xy_rot[0], gt_xy_rot[1], "r-", label="GT", linewidth=2.5)
    
    # 新坐标轴范围
    plt.xlim(X_MIN, X_MAX)
    plt.ylim(Y_MIN, Y_MAX)
    
    plt.xlabel("X (m)", fontsize=10)
    plt.ylabel("Y (m)", fontsize=10)
    plt.title(f"Frame {infer_idx:06d}", fontsize=12)
    plt.legend(loc="upper left", fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # 保存到内存
    plt.tight_layout()
    plt.savefig(f'/tmp/traj_{infer_idx}.png', dpi=100, bbox_inches='tight')
    plt.close()
    
    # 读取
    traj_img = cv2.imread(f'/tmp/traj_{infer_idx}.png')
    os.remove(f'/tmp/traj_{infer_idx}.png')
    
    return traj_img

def create_info_panel(row, width, height):
    """创建信息面板 (CoC + ADE)"""
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 解析CoC
    coc_text = json.loads(row['coc_text'])
    if isinstance(coc_text, list) and len(coc_text) > 0:
        coc_str = coc_text[0][0] if isinstance(coc_text[0], list) else str(coc_text[0])
    else:
        coc_str = str(coc_text)
    
    # 限制长度
    if len(coc_str) > 100:
        coc_str = coc_str[:97] + "..."
    
    # 文字信息
    info_lines = [
        f"Frame: {row['infer_idx']}",
        f"ADE: {row['ade']:.3f} m",
        "",
        "CoC:",
        coc_str
    ]
    
    # 绘制文字
    y_offset = 30
    for line in info_lines:
        cv2.putText(panel, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (255, 255, 255), 1, cv2.LINE_AA)
        y_offset += 25
    
    return panel

def create_frame(row, result_dir, data_dir):
    """创建一帧组合画面"""
    infer_idx = row['infer_idx']
    
    # 1. 16宫格 (上面)
    grid = create_16_grid(row, data_dir)
    grid_h, grid_w = grid.shape[:2]
    
    # 2. 轨迹图 (右下)
    traj_img = create_trajectory_plot(infer_idx, result_dir, data_dir)
    traj_h, traj_w = 540, 960  # 固定尺寸
    traj_img = cv2.resize(traj_img, (traj_w, traj_h))
    
    # 3. 信息面板 (左下)
    info_w = grid_w - traj_w
    info_h = traj_h
    info_panel = create_info_panel(row, info_w, info_h)
    
    # 4. 组合下半部分
    bottom = np.hstack([info_panel, traj_img])
    
    # 5. 组合完整画面
    frame = np.vstack([grid, bottom])
    
    return frame

def main():
    print('=== 创建组合视频 ===\n')
    
    # 加载CSV
    csv_path = f'{RESULT_DIR}/continuous_inference_results.csv'
    df = pd.read_csv(csv_path)
    print(f'加载 {len(df)} 帧数据')
    
    # 加载高质量索引以获取图片路径
    index_path = f'{DATA_DIR}/inference_index_high_quality.csv'
    index_df = pd.read_csv(index_path)
    
    # 合并数据
    merged = df.merge(index_df[['infer_idx'] + [col for cols in cam_cols.values() for col in cols]], 
                     on='infer_idx', how='left')
    
    # 视频参数
    frame = create_frame(merged.iloc[0], RESULT_DIR, DATA_DIR)
    height, width = frame.shape[:2]
    fps = 10  # 10fps
    
    print(f'视频尺寸: {width}x{height}, FPS: {fps}')
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(VIDEO_OUTPUT, fourcc, fps, (width, height))
    
    print('\n生成视频帧...')
    for idx in tqdm(range(len(merged))):
        row = merged.iloc[idx]
        try:
            frame = create_frame(row, RESULT_DIR, DATA_DIR)
            out.write(frame)
        except Exception as e:
            print(f"\n  Frame {row['infer_idx']} 错误: {e}")
    
    out.release()
    
    # 检查视频
    video_size = os.path.getsize(VIDEO_OUTPUT) / (1024*1024)
    print(f'\n视频保存至: {VIDEO_OUTPUT}')
    print(f'文件大小: {video_size:.1f} MB')

if __name__ == '__main__':
    main()
