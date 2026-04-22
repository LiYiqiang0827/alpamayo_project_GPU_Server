#!/usr/bin/env python3
"""
批量生成组合帧 - 新布局
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

X_MIN, X_MAX = -15, 15
Y_MIN, Y_MAX = 0, 60

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
    img_w, img_h = 320, 180
    grid_w, grid_h = img_w * 4, img_h * 4
    grid = Image.new('RGB', (grid_w, grid_h), color=(0, 0, 0))
    
    for cam_idx, cam_name in enumerate(cameras):
        for frame_idx, col in enumerate(cam_cols[cam_name]):
            img_path = f'{data_dir}/{row[col]}'
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize((img_w, img_h))
            except:
                img = Image.new('RGB', (img_w, img_h), color=(128, 128, 128))
            grid.paste(img, (frame_idx * img_w, cam_idx * img_h))
    return grid

def create_trajectory_plot(infer_idx, result_dir, data_dir):
    """创建轨迹图 - 使用动态坐标范围"""
    pred = np.load(f'{result_dir}/pred_{infer_idx:06d}.npy')
    if pred.ndim == 2:
        pred = pred[np.newaxis, :, :]
    
    future = np.load(f'{data_dir}/egomotion/ego_{infer_idx:06d}_future_gt.npy', allow_pickle=True).item()
    gt_xy = future['xyz'][:, :2]  # (64, 2)
    
    fig, ax = plt.subplots(figsize=(5.4, 10.8), dpi=100)
    
    # 绘制预测轨迹
    colors = plt.cm.tab10(np.linspace(0, 1, pred.shape[0]))
    for i in range(pred.shape[0]):
        pred_xy = pred[i]
        ax.plot(pred_xy[:, 0], pred_xy[:, 1], "o-", color=colors[i], 
                label=f"Pred #{i+1}", markersize=2, linewidth=1.5)
    
    # 绘制真值轨迹
    ax.plot(gt_xy[:, 0], gt_xy[:, 1], "r-", label="GT", linewidth=2.5)
    
    # 标记起点
    ax.plot(0, 0, "k*", markersize=15, label="Ego (t0)")
    
    # 动态计算坐标范围 - 使用轨迹数据范围
    all_x = np.concatenate([gt_xy[:, 0], pred[:, :, 0].flatten()])
    all_y = np.concatenate([gt_xy[:, 1], pred[:, :, 1].flatten()])
    
    x_margin = (all_x.max() - all_x.min()) * 0.1 + 1  # 10% margin, at least 1m
    y_margin = (all_y.max() - all_y.min()) * 0.1 + 1
    
    x_min, x_max = all_x.min() - x_margin, all_x.max() + x_margin
    y_min, y_max = all_y.min() - y_margin, all_y.max() + y_margin
    
    # 确保Y从0开始（前方为正）
    y_min = min(0, y_min)
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("X (m) - Right", fontsize=10)
    ax.set_ylabel("Y (m) - Forward", fontsize=10)
    ax.set_title(f"Frame {infer_idx:06d}", fontsize=12)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.tight_layout()
    
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    img_array = img_array[:, :, :3]
    traj_img = Image.fromarray(img_array)
    plt.close(fig)
    
    return traj_img

def create_info_panel(row, width, height):
    panel = Image.new('RGB', (width, height), color=(40, 40, 40))
    draw = ImageDraw.Draw(panel)
    
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        font_medium = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except:
        font_large = font_medium = font_small = ImageFont.load_default()
    
    try:
        coc_text = json.loads(row['coc_text'])
        coc_str = coc_text[0][0] if isinstance(coc_text[0], list) else str(coc_text[0])
    except:
        coc_str = str(row['coc_text'])
    
    # 换行
    def wrap_text(text, max_w, font):
        lines, current = [], ""
        for word in text.split():
            test = current + " " + word if current else word
            if draw.textbbox((0,0), test, font=font)[2] <= max_w:
                current = test
            else:
                if current: lines.append(current)
                current = word
        if current: lines.append(current)
        return lines[:8]
    
    coc_lines = wrap_text(coc_str, width-40, font_small)
    
    y = 30
    draw.text((20, y), f"Frame: {row['infer_idx']}", fill=(255,255,0), font=font_large); y += 50
    draw.text((20, y), f"ADE: {row['ade']:.3f} m", fill=(0,255,0), font=font_medium); y += 45
    draw.text((20, y), f"Time: {row['inference_time_ms']:.0f} ms", fill=(200,200,200), font=font_medium); y += 60
    draw.text((20, y), "CoC Reasoning:", fill=(100,200,255), font=font_medium); y += 40
    
    for line in coc_lines:
        draw.text((20, y), line, fill=(255,255,255), font=font_small); y += 28
    
    return panel

def create_combined_frame(row, result_dir, data_dir):
    infer_idx = row['infer_idx']
    
    grid = create_16_grid(row, data_dir)
    
    traj_img = create_trajectory_plot(infer_idx, result_dir, data_dir)
    traj_img = traj_img.resize((540, 1080))
    
    info_panel = create_info_panel(row, 740, 1080)
    
    bottom = Image.new('RGB', (1280, 1080))
    bottom.paste(info_panel, (0, 0))
    bottom.paste(traj_img, (740, 0))
    
    frame = Image.new('RGB', (1280, 1800))
    frame.paste(grid, (0, 0))
    frame.paste(bottom, (0, 720))
    
    return frame

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Step 3: 批量生成组合帧')
    parser.add_argument('--clip', type=str, required=True, help='Clip ID')
    parser.add_argument('--chunk', type=int, default=0, help='Chunk number (default: 0)')
    args = parser.parse_args()
    
    clip_id = args.clip
    chunk = args.chunk
    RESULT_DIR = f'/data01/vla/data/data_sample_chunk{chunk}/infer/{clip_id}/result'
    DATA_DIR = f'/data01/vla/data/data_sample_chunk{chunk}/infer/{clip_id}/data'
    FRAME_OUTPUT_DIR = f'{RESULT_DIR}/combined_frames_v2'
    
    print(f'=== Step 3: 批量生成组合帧 ({clip_id}) ===\n')
    os.makedirs(FRAME_OUTPUT_DIR, exist_ok=True)
    
    df = pd.read_csv(f'{RESULT_DIR}/continuous_inference_results.csv')
    
    # 尝试读取高质量索引，如果不存在则使用普通索引
    index_path = f'{DATA_DIR}/inference_index_high_quality.csv'
    if not os.path.exists(index_path):
        index_path = f'{DATA_DIR}/inference_index.csv'
    index_df = pd.read_csv(index_path)
    
    all_cols = ['infer_idx'] + [c for cols in cam_cols.values() for c in cols]
    merged = df.merge(index_df[all_cols], on='infer_idx', how='left')
    
    print(f'生成 {len(merged)} 帧...')
    for i in tqdm(range(len(merged))):
        row = merged.iloc[i]
        try:
            frame = create_combined_frame(row, RESULT_DIR, DATA_DIR)
            frame.save(f'{FRAME_OUTPUT_DIR}/frame_{row["infer_idx"]:06d}.jpg', 'JPEG', quality=90)
        except Exception as e:
            print(f"\nFrame {row['infer_idx']}: {e}")
    
    print(f'\n完成！保存至: {FRAME_OUTPUT_DIR}/')
    print(f'共 {len(os.listdir(FRAME_OUTPUT_DIR))} 帧')

if __name__ == '__main__':
    main()
