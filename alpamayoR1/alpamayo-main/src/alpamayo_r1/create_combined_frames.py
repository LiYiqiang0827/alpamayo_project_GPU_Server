#!/usr/bin/env python3
"""
创建组合帧图片 (用于后期合成视频)
- 上面: 16宫格 (4摄像头 × 4帧)
- 左下: CoC文字 + ADE信息
- 右下: 轨迹图 (新坐标轴范围: X-15~15, Y0~60)
"""
import os
import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# 配置
RESULT_DIR = '/data01/vla/data/data_sample_chunk0/infer/01d3588e-bca7-4a18-8e74-c6cfe9e996db/result'
DATA_DIR = '/data01/vla/data/data_sample_chunk0/infer/01d3588e-bca7-4a18-8e74-c6cfe9e996db/data'
FRAME_OUTPUT_DIR = f'{RESULT_DIR}/combined_frames'

# 坐标轴范围 (新)
X_MIN, X_MAX = -15, 15  # 横轴 -15m ~ +15m
Y_MIN, Y_MAX = 0, 60    # 纵轴 0m ~ 60m

os.makedirs(FRAME_OUTPUT_DIR, exist_ok=True)

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
    img_w, img_h = 320, 180  # 16:9比例，总尺寸1280x720
    grid_w = img_w * 4
    grid_h = img_h * 4
    
    grid = Image.new('RGB', (grid_w, grid_h), color=(0, 0, 0))
    
    for cam_idx, cam_name in enumerate(cameras):
        for frame_idx, col in enumerate(cam_cols[cam_name]):
            img_path = f'{data_dir}/{row[col]}'
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize((img_w, img_h))
            except:
                img = Image.new('RGB', (img_w, img_h), color=(128, 128, 128))
            
            x_pos = frame_idx * img_w
            y_pos = cam_idx * img_h
            grid.paste(img, (x_pos, y_pos))
    
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
    fig, ax = plt.subplots(figsize=(9.6, 5.4), dpi=100)
    
    colors = plt.cm.tab10(np.linspace(0, 1, pred_xyz.shape[0]))
    for i in range(pred_xyz.shape[0]):
        pred_xy = pred_xyz[i].T
        pred_xy_rot = rotate_90cc(pred_xy)
        ax.plot(pred_xy_rot[0], pred_xy_rot[1], "o-", 
                color=colors[i], label=f"Pred #{i+1}", markersize=2, linewidth=1.5)
    
    gt_xy_rot = rotate_90cc(gt_xy.T)
    ax.plot(gt_xy_rot[0], gt_xy_rot[1], "r-", label="GT", linewidth=2.5)
    
    # 新坐标轴范围
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xlabel("X (m)", fontsize=9)
    ax.set_ylabel("Y (m)", fontsize=9)
    ax.set_title(f"Frame {infer_idx:06d}", fontsize=11)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 保存到内存
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))  # RGBA
    img_array = img_array[:, :, :3]  # 去掉alpha通道
    traj_img = Image.fromarray(img_array)
    plt.close(fig)
    
    return traj_img

def create_info_panel(row, width, height):
    """创建信息面板 (CoC + ADE)"""
    panel = Image.new('RGB', (width, height), color=(40, 40, 40))
    draw = ImageDraw.Draw(panel)
    
    # 解析CoC
    try:
        coc_text = json.loads(row['coc_text'])
        if isinstance(coc_text, list) and len(coc_text) > 0:
            coc_str = coc_text[0][0] if isinstance(coc_text[0], list) else str(coc_text[0])
        else:
            coc_str = str(coc_text)
    except:
        coc_str = str(row['coc_text'])
    
    # 限制长度
    if len(coc_str) > 80:
        coc_str = coc_str[:77] + "..."
    
    # 文字
    lines = [
        f"Frame: {row['infer_idx']}",
        f"ADE: {row['ade']:.3f} m",
        f"Time: {row['inference_time_ms']:.0f} ms",
        "",
        "CoC Reasoning:",
        coc_str
    ]
    
    y_offset = 20
    for line in lines:
        draw.text((10, y_offset), line, fill=(255, 255, 255))
        y_offset += 25
    
    return panel

def create_combined_frame(row, result_dir, data_dir):
    """创建一帧组合画面"""
    infer_idx = row['infer_idx']
    
    # 1. 16宫格 (上面) 1280x720
    grid = create_16_grid(row, data_dir)
    
    # 2. 轨迹图 (右下) 960x540
    traj_img = create_trajectory_plot(infer_idx, result_dir, data_dir)
    traj_img = traj_img.resize((960, 540))
    
    # 3. 信息面板 (左下) 320x540
    info_panel = create_info_panel(row, 320, 540)
    
    # 4. 组合下半部分 (1280x540)
    bottom = Image.new('RGB', (1280, 540))
    bottom.paste(info_panel, (0, 0))
    bottom.paste(traj_img, (320, 0))
    
    # 5. 组合完整画面 (1280x1260)
    frame = Image.new('RGB', (1280, 1260))
    frame.paste(grid, (0, 0))
    frame.paste(bottom, (0, 720))
    
    return frame

def main():
    print('=== 创建组合帧 ===\n')
    
    # 加载CSV
    csv_path = f'{RESULT_DIR}/continuous_inference_results.csv'
    df = pd.read_csv(csv_path)
    print(f'加载 {len(df)} 帧数据')
    
    # 加载高质量索引
    index_path = f'{DATA_DIR}/inference_index_high_quality.csv'
    index_df = pd.read_csv(index_path)
    
    # 合并数据
    all_cols = ['infer_idx'] + [col for cols in cam_cols.values() for col in cols]
    merged = df.merge(index_df[all_cols], on='infer_idx', how='left')
    
    print(f'\n生成 {len(merged)} 帧组合画面...')
    for idx in tqdm(range(len(merged))):
        row = merged.iloc[idx]
        try:
            frame = create_combined_frame(row, RESULT_DIR, DATA_DIR)
            output_path = f'{FRAME_OUTPUT_DIR}/frame_{row["infer_idx"]:06d}.png'
            frame.save(output_path, quality=95)
        except Exception as e:
            print(f"\n  Frame {row['infer_idx']} 错误: {e}")
    
    print(f'\n完成！帧保存至: {FRAME_OUTPUT_DIR}/')
    print(f'共 {len(os.listdir(FRAME_OUTPUT_DIR))} 张图片')
    print('\n后期可用ffmpeg合成视频:')
    print(f'  ffmpeg -framerate 10 -i {FRAME_OUTPUT_DIR}/frame_%06d.png -c:v libx264 -pix_fmt yuv420p output.mp4')

if __name__ == '__main__':
    main()
