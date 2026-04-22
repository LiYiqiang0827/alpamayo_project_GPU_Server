#!/usr/bin/env python3
"""
创建组合帧样例 - 调整布局
- 文字更大
- 轨迹图宽高比 1:2
"""
import os
import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

# 配置
RESULT_DIR = '/data01/vla/data/data_sample_chunk0/infer/01d3588e-bca7-4a18-8e74-c6cfe9e996db/result'
DATA_DIR = '/data01/vla/data/data_sample_chunk0/infer/01d3588e-bca7-4a18-8e74-c6cfe9e996db/data'

# 新坐标轴范围
X_MIN, X_MAX = -15, 15
Y_MIN, Y_MAX = 0, 60

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
    return np.stack([-xy[1], xy[0]], axis=0)

def create_16_grid(row, data_dir):
    """创建16宫格"""
    img_w, img_h = 320, 180
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
    """创建轨迹图 - 宽高比 1:2"""
    pred_path = f'{result_dir}/pred_{infer_idx:06d}.npy'
    pred_xyz = np.load(pred_path)
    if pred_xyz.ndim == 2:
        pred_xyz = pred_xyz[np.newaxis, :, :]
    
    ego_file = f'{data_dir}/egomotion/ego_{infer_idx:06d}_future_gt.npy'
    future = np.load(ego_file, allow_pickle=True).item()
    gt_xy = future['xyz'][:, :2]
    
    # 宽高比 1:2 (宽:高 = 1:2) -> 高度是宽度的2倍
    # 但为了适应画面，我们设定宽度540，高度1080
    fig, ax = plt.subplots(figsize=(5.4, 10.8), dpi=100)  # 1:2 比例
    
    colors = plt.cm.tab10(np.linspace(0, 1, pred_xyz.shape[0]))
    for i in range(pred_xyz.shape[0]):
        pred_xy = pred_xyz[i].T
        pred_xy_rot = rotate_90cc(pred_xy)
        ax.plot(pred_xy_rot[0], pred_xy_rot[1], "o-", 
                color=colors[i], label=f"Pred #{i+1}", markersize=2, linewidth=1.5)
    
    gt_xy_rot = rotate_90cc(gt_xy.T)
    ax.plot(gt_xy_rot[0], gt_xy_rot[1], "r-", label="GT", linewidth=2.5)
    
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xlabel("X (m)", fontsize=10)
    ax.set_ylabel("Y (m)", fontsize=10)
    ax.set_title(f"Frame {infer_idx:06d}", fontsize=12)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 调整布局让图占满
    plt.tight_layout()
    
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    img_array = img_array[:, :, :3]
    traj_img = Image.fromarray(img_array)
    plt.close(fig)
    
    return traj_img

def create_info_panel(row, width, height):
    """创建信息面板 - 更大的文字"""
    panel = Image.new('RGB', (width, height), color=(40, 40, 40))
    draw = ImageDraw.Draw(panel)
    
    # 尝试加载字体，如果失败用默认
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        font_medium = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except:
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # 解析CoC
    try:
        coc_text = json.loads(row['coc_text'])
        if isinstance(coc_text, list) and len(coc_text) > 0:
            coc_str = coc_text[0][0] if isinstance(coc_text[0], list) else str(coc_text[0])
        else:
            coc_str = str(coc_text)
    except:
        coc_str = str(row['coc_text'])
    
    # 文字换行处理
    def wrap_text(text, max_width, font):
        lines = []
        words = text.split()
        current_line = ""
        for word in words:
            test_line = current_line + " " + word if current_line else word
            bbox = draw.textbbox((0, 0), test_line, font=font)
            if bbox[2] - bbox[0] <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        return lines[:6]  # 最多6行
    
    coc_lines = wrap_text(coc_str, width - 40, font_small)
    
    # 绘制文字
    y_offset = 30
    
    # Frame
    draw.text((20, y_offset), f"Frame: {row['infer_idx']}", fill=(255, 255, 0), font=font_large)
    y_offset += 50
    
    # ADE
    draw.text((20, y_offset), f"ADE: {row['ade']:.3f} m", fill=(0, 255, 0), font=font_medium)
    y_offset += 45
    
    # Time
    draw.text((20, y_offset), f"Time: {row['inference_time_ms']:.0f} ms", fill=(200, 200, 200), font=font_medium)
    y_offset += 60
    
    # CoC标题
    draw.text((20, y_offset), "CoC Reasoning:", fill=(100, 200, 255), font=font_medium)
    y_offset += 40
    
    # CoC内容
    for line in coc_lines:
        draw.text((20, y_offset), line, fill=(255, 255, 255), font=font_small)
        y_offset += 28
    
    return panel

def create_combined_frame(row, result_dir, data_dir):
    """创建一帧组合画面 - 新布局"""
    infer_idx = row['infer_idx']
    
    # 1. 16宫格 (上面) 1280×720
    grid = create_16_grid(row, data_dir)
    
    # 2. 轨迹图 (右下) 540×1080 (宽高比1:2)
    traj_img = create_trajectory_plot(infer_idx, result_dir, data_dir)
    traj_img = traj_img.resize((540, 1080))
    
    # 3. 信息面板 (左下) 740×1080 (更大空间给文字)
    info_w = 1280 - 540  # 740
    info_h = 1080
    info_panel = create_info_panel(row, info_w, info_h)
    
    # 4. 组合下半部分 (1280×1080)
    bottom = Image.new('RGB', (1280, 1080))
    bottom.paste(info_panel, (0, 0))
    bottom.paste(traj_img, (740, 0))
    
    # 5. 组合完整画面 (1280×1800)
    frame = Image.new('RGB', (1280, 1800))
    frame.paste(grid, (0, 0))
    frame.paste(bottom, (0, 720))
    
    return frame

def main():
    print('=== 创建组合帧样例 (新布局) ===\n')
    
    # 加载CSV
    csv_path = f'{RESULT_DIR}/continuous_inference_results.csv'
    df = pd.read_csv(csv_path)
    
    # 加载高质量索引
    index_path = f'{DATA_DIR}/inference_index_high_quality.csv'
    index_df = pd.read_csv(index_path)
    
    # 合并数据
    all_cols = ['infer_idx'] + [col for cols in cam_cols.values() for col in cols]
    merged = df.merge(index_df[all_cols], on='infer_idx', how='left')
    
    # 生成样例 (Frame 500)
    sample_idx = 388
    row = merged[merged['infer_idx'] == sample_idx].iloc[0]
    
    print(f'生成 Frame {sample_idx} 样例...')
    frame = create_combined_frame(row, RESULT_DIR, DATA_DIR)
    
    output_path = f'{RESULT_DIR}/combined_frame_sample_v2.png'
    frame.save(output_path, quality=95)
    
    print(f'\n样例保存至: {output_path}')
    print(f'尺寸: {frame.size}')

if __name__ == '__main__':
    main()
