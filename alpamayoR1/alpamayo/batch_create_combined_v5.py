#!/usr/bin/env python3
"""
批量生成组合帧 - V5 严格版本
适配 result_strict 目录和 inference_index_strict.csv 格式
"""
import os
import sys
import argparse
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

CAMERA_ORDER = [
    'camera_cross_left_120fov',
    'camera_front_wide_120fov', 
    'camera_cross_right_120fov',
    'camera_front_tele_30fov'
]

def load_images_for_frame_v5(row, data_dir):
    """从V5严格索引格式加载图片 - 16宫格"""
    images = []
    for cam in CAMERA_ORDER:
        for t in range(4):
            idx_col = f'{cam}_f{t}_idx'
            frame_idx = int(row[idx_col])
            img_path = f'{data_dir}/camera_images/{cam}/{frame_idx:06d}.jpg'
            try:
                img = Image.open(img_path).convert('RGB')
            except:
                img = Image.new('RGB', (320, 180), color=(128, 128, 128))
            img = img.resize((320, 180))
            images.append(img)
    return images

def create_16_grid(images):
    """创建16宫格图像 (4摄像头 × 4时间帧)"""
    img_w, img_h = 320, 180
    grid_w, grid_h = img_w * 4, img_h * 4
    grid = Image.new('RGB', (grid_w, grid_h), color=(0, 0, 0))
    
    idx = 0
    for cam_idx in range(4):  # 4个摄像头
        for time_idx in range(4):  # 4个时间帧
            grid.paste(images[idx], (time_idx * img_w, cam_idx * img_h))
            idx += 1
    
    return grid

def create_trajectory_plot_v5(frame_id, result_dir, data_dir):
    """创建轨迹图 - V5版本"""
    # 加载预测结果 (已经是局部坐标系)
    pred_path = f'{result_dir}/pred_{frame_id:06d}.npy'
    pred = np.load(pred_path)
    if pred.ndim == 2:
        pred = pred[np.newaxis, :, :]  # (1, 64, 2)
    
    # 加载真值历史 (用于计算 t0 位姿)
    history_path = f'{data_dir}/egomotion/frame_{frame_id:06d}_history.npy'
    history = np.load(history_path, allow_pickle=False)
    # history: (16, 11) - columns: timestamp, qx, qy, qz, qw, x, y, z, vx, vy, vz
    hist_xyz_world = history[:, 5:8]  # x, y, z (columns 5,6,7)
    hist_quat = history[:, 1:5]  # qx, qy, qz, qw (columns 1,2,3,4)
    
    # 加载真值未来 (世界坐标系)
    future_path = f'{data_dir}/egomotion/frame_{frame_id:06d}_future_gt.npy'
    future = np.load(future_path, allow_pickle=False)
    # future: (64, 4) - columns: timestamp, x, y, z
    future_xyz_world = future[:, 1:4]  # x, y, z (columns 1,2,3)
    
    # 获取 t0 时刻的位姿（历史最后一个点）
    t0_xyz = hist_xyz_world[-1].copy()
    t0_quat = hist_quat[-1].copy()
    
    # 将真值未来轨迹转换到局部坐标系
    from scipy.spatial.transform import Rotation as R
    t0_rot = R.from_quat(t0_quat)
    t0_rot_inv = t0_rot.inv()
    future_xyz_local = t0_rot_inv.apply(future_xyz_world - t0_xyz)  # (64, 3)
    
    # 提取 XY (X=前进, Y=侧向)
    gt_xy = future_xyz_local[:, :2]  # (64, 2) - x, y
    
    # 在 GT 轨迹前添加原点 (0,0)，让轨迹从 t=0 开始
    gt_xy = np.vstack([np.array([[0, 0]]), gt_xy])  # (65, 2)
    
    fig, ax = plt.subplots(figsize=(5.4, 10.8), dpi=100)
    
    # 坐标系修正：数据中 X 是前进方向，Y 是侧向（右）
    # 画图时：横轴 = 侧向(Y)，纵轴 = 前进(X)
    # 这样车辆前进时轨迹是向上的
    
    # 绘制预测轨迹
    colors = plt.cm.tab10(np.linspace(0, 1, pred.shape[0]))
    for i in range(pred.shape[0]):
        pred_xy = pred[i]
        # pred_xy[:, 0] = X (前进), pred_xy[:, 1] = Y (侧向)
        ax.plot(pred_xy[:, 1], pred_xy[:, 0], "o-", color=colors[i], 
                label=f"Pred #{i+1}", markersize=2, linewidth=1.5)
    
    # 绘制真值轨迹
    # gt_xy[:, 0] = X (前进), gt_xy[:, 1] = Y (侧向)
    ax.plot(gt_xy[:, 1], gt_xy[:, 0], "r-", label="GT", linewidth=2.5)
    
    # 标记起点
    ax.plot(0, 0, "k*", markersize=15, label="Ego (t0)")
    
    # 固定坐标轴范围
    # 横轴（侧向）：-15 到 +15 米
    # 纵轴（前进）：0 到 60 米
    x_min, x_max = -15, 15
    y_min, y_max = 0, 60
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Lateral (m) - Right", fontsize=10)
    ax.set_ylabel("Forward (m)", fontsize=10)
    ax.set_title(f"Frame {frame_id:06d}", fontsize=12)
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

def create_info_panel(frame_id, row, width, height):
    """创建信息面板"""
    panel = Image.new('RGB', (width, height), color=(40, 40, 40))
    draw = ImageDraw.Draw(panel)
    
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        font_medium = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except:
        font_large = font_medium = font_small = ImageFont.load_default()
    
    # 解析 CoC
    try:
        coc_text = json.loads(row['coc_text'])
        if isinstance(coc_text, list) and len(coc_text) > 0:
            coc_str = coc_text[0] if isinstance(coc_text[0], str) else str(coc_text[0])
        else:
            coc_str = str(coc_text)
    except:
        coc_str = str(row.get('coc_text', 'N/A'))
    
    # 换行函数
    def wrap_text(text, max_w, font):
        if not text:
            return []
        lines, current = [], ""
        for word in str(text).split():
            test = current + " " + word if current else word
            try:
                bbox = draw.textbbox((0,0), test, font=font)
                if bbox[2] <= max_w:
                    current = test
                else:
                    if current: lines.append(current)
                    current = word
            except:
                if len(test) * 10 <= max_w:
                    current = test
                else:
                    if current: lines.append(current)
                    current = word
        if current: lines.append(current)
        return lines[:10]
    
    coc_lines = wrap_text(coc_str, width-40, font_small)
    
    y = 30
    draw.text((20, y), f"Frame: {frame_id}", fill=(255,255,0), font=font_large); y += 50
    draw.text((20, y), f"ADE: {row.get('ade', 0):.3f} m", fill=(0,255,0), font=font_medium); y += 45
    draw.text((20, y), f"Time: {row.get('inference_time_ms', 0):.0f} ms", fill=(200,200,200), font=font_medium); y += 60
    draw.text((20, y), "CoC Reasoning:", fill=(100,200,255), font=font_medium); y += 40
    
    for line in coc_lines:
        draw.text((20, y), line, fill=(255,255,255), font=font_small); y += 28
    
    return panel

def create_combined_frame_v5(frame_id, row, result_dir, data_dir):
    """创建合成帧 - V5版本"""
    # 加载图片
    images = load_images_for_frame_v5(row, data_dir)
    grid = create_16_grid(images)
    
    # 创建轨迹图
    traj_img = create_trajectory_plot_v5(frame_id, result_dir, data_dir)
    traj_img = traj_img.resize((540, 1080))
    
    # 创建信息面板
    info_panel = create_info_panel(frame_id, row, 740, 1080)
    
    # 底部面板
    bottom = Image.new('RGB', (1280, 1080))
    bottom.paste(info_panel, (0, 0))
    bottom.paste(traj_img, (740, 0))
    
    # 最终合成
    frame = Image.new('RGB', (1280, 1800))
    frame.paste(grid, (0, 0))
    frame.paste(bottom, (0, 720))
    
    return frame

def main():
    parser = argparse.ArgumentParser(description='Step 3: 批量生成组合帧 (V5 严格版本)')
    parser.add_argument('--clip', type=str, required=True, help='Clip ID')
    parser.add_argument('--chunk', type=int, default=0, help='Chunk number (default: 0)')
    args = parser.parse_args()
    
    clip_id = args.clip
    chunk = args.chunk
    
    # V5 使用 result_strict 目录
    RESULT_DIR = f'/data01/vla/data/data_sample_chunk{chunk}/infer/{clip_id}/result_strict'
    DATA_DIR = f'/data01/vla/data/data_sample_chunk{chunk}/infer/{clip_id}/data'
    FRAME_OUTPUT_DIR = f'{RESULT_DIR}/combined_frames'
    
    print(f'=== Step 3: 批量生成组合帧 V5 ({clip_id}) ===\n')
    os.makedirs(FRAME_OUTPUT_DIR, exist_ok=True)
    
    # 读取 V5 推理结果
    results_path = f'{RESULT_DIR}/inference_results_strict.csv'
    if not os.path.exists(results_path):
        print(f"❌ 错误: 找不到结果文件 {results_path}")
        print(f"   请先运行 V5 推理脚本")
        sys.exit(1)
    
    results_df = pd.read_csv(results_path)
    print(f'读取到 {len(results_df)} 帧推理结果')
    
    # 读取 V5 严格索引
    index_path = f'{DATA_DIR}/inference_index_strict.csv'
    if not os.path.exists(index_path):
        print(f"❌ 错误: 找不到索引文件 {index_path}")
        sys.exit(1)
    
    index_df = pd.read_csv(index_path)
    print(f'读取到 {len(index_df)} 帧索引')
    
    # 合并结果和索引
    merged = results_df.merge(index_df, left_on='frame_id', right_on='frame_id', how='inner')
    print(f'合并后: {len(merged)} 帧\n')
    
    # 生成组合帧
    success_count = 0
    for i in tqdm(range(len(merged)), desc='生成组合帧'):
        row = merged.iloc[i]
        frame_id = int(row['frame_id'])
        
        try:
            frame = create_combined_frame_v5(frame_id, row, RESULT_DIR, DATA_DIR)
            frame.save(f'{FRAME_OUTPUT_DIR}/frame_{frame_id:06d}.jpg', 'JPEG', quality=90)
            success_count += 1
        except Exception as e:
            print(f"\n❌ Frame {frame_id}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f'\n✅ 完成! 成功 {success_count}/{len(merged)} 帧')
    print(f'   保存至: {FRAME_OUTPUT_DIR}/')

if __name__ == '__main__':
    main()
