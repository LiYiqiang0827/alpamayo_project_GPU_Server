#!/usr/bin/env python3
"""
创建完整的场景分析可视化 - 包含图像、轨迹图、CoC文本
"""
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import pandas as pd
import numpy as np
import torch
import av
import json
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import scipy.spatial.transform as spt
from einops import rearrange

def load_video_frames(video_path: str, num_frames: int = 4):
    container = av.open(video_path)
    stream = container.streams.video[0]
    total_frames = stream.frames if stream.frames else 100
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    for i, frame in enumerate(container.decode(video=0)):
        if i in indices:
            img = frame.to_ndarray(format='rgb24')
            frames.append(img)
        if len(frames) >= num_frames:
            break
    container.close()
    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
    frames = np.stack(frames)
    frames = rearrange(frames, "t h w c -> t c h w")
    return torch.from_numpy(frames).float()

def load_egomotion(ego_path: str):
    df = pd.read_parquet(ego_path)
    return df[['timestamp', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']].values

def extract_frame_single(video_path: str, frame_idx: int):
    container = av.open(video_path)
    frame_img = None
    for i, frame in enumerate(container.decode(video=0)):
        if i == frame_idx:
            frame_img = frame.to_ndarray(format='rgb24')
            break
    container.close()
    return frame_img

def rotate_90cc(xy):
    """Rotate (x, y) by 90 deg CCW -> (y, -x)"""
    return np.stack([-xy[1], xy[0]], axis=0)

def main():
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from alpamayo_r1 import helper
    
    base_dir = "/data01/vla/data_sample_chunk0"
    clip_id = "fa83bcb8-ea31-4dbb-b447-4fb458f5984b"
    
    # 找到视频中间的时间点
    ts_path = f"{base_dir}/camera/camera_front_wide_120fov/{clip_id}.camera_front_wide_120fov.timestamps.parquet"
    df_ts = pd.read_parquet(ts_path)
    ts = df_ts['timestamp'].values
    video_start = ts[0]
    video_end = ts[-1]
    t0_us = video_start + 7_900_000  # t0 = 7.9s
    
    print("=" * 70)
    print(f"创建完整可视化分析")
    print("=" * 70)
    print(f"Clip ID: {clip_id}")
    print(f"时间点: t0 = {t0_us/1e6:.1f}s")
    print()
    
    # 创建输出目录
    output_dir = f"/tmp/scene_analysis_{clip_id[:8]}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/camera_images", exist_ok=True)
    
    # 1. 提取并保存4摄像头图像
    print("1. 提取4摄像头图像...")
    cameras = [
        ("camera_cross_left_120fov", "01_left_cross"),
        ("camera_front_wide_120fov", "02_front_wide"),
        ("camera_cross_right_120fov", "03_right_cross"),
        ("camera_front_tele_30fov", "04_front_tele"),
    ]
    
    camera_images = {}
    for cam_name, label in cameras:
        video_path = f"{base_dir}/camera/{cam_name}/{clip_id}.{cam_name}.mp4"
        ts_cam_path = video_path.replace('.mp4', '.timestamps.parquet')
        
        if os.path.exists(video_path) and os.path.exists(ts_cam_path):
            df_cam = pd.read_parquet(ts_cam_path)
            ts_cam = df_cam['timestamp'].values
            frame_idx = np.argmin(np.abs(ts_cam - t0_us))
            
            frame = extract_frame_single(video_path, frame_idx)
            
            if frame is not None:
                # 保存原始图像
                img = Image.fromarray(frame)
                img.save(f"{output_dir}/camera_images/{label}.png")
                camera_images[label] = frame
                print(f"  ✅ {label}.png saved")
    
    # 2. 加载 ego motion 数据
    print("\n2. 加载 ego motion 数据...")
    ego_path = f"{base_dir}/labels/egomotion/{clip_id}.egomotion.parquet"
    ego_data = load_egomotion(ego_path)
    
    # 计算历史和未来轨迹
    history_timestamps = t0_us + np.arange(-15, 1, 1) * 100000
    future_timestamps = t0_us + np.arange(1, 65, 1) * 100000
    
    # 插值获取历史轨迹
    hist_xyz = []
    for t in history_timestamps:
        idx = np.argmin(np.abs(ego_data[:, 0] - t))
        hist_xyz.append(ego_data[idx, 1:4].tolist())
    
    # 插值获取未来真值
    future_xyz = []
    for t in future_timestamps:
        idx = np.argmin(np.abs(ego_data[:, 0] - t))
        future_xyz.append(ego_data[idx, 1:4].tolist())
    
    # 3. 加载模型进行推理
    print("\n3. 加载模型推理...")
    model = AlpamayoR1.from_pretrained(
        "nvidia/Alpamayo-R1-10B",
        dtype=torch.bfloat16,
        local_files_only=True,
    ).to("cuda")
    processor = helper.get_processor(model.tokenizer)
    
    # 准备输入
    cameras = [
        ("camera_cross_left_120fov", 0),
        ("camera_front_wide_120fov", 1),
        ("camera_cross_right_120fov", 2),
        ("camera_front_tele_30fov", 6),
    ]
    
    image_frames_list = []
    for cam_name, cam_idx in cameras:
        video_path = f"{base_dir}/camera/{cam_name}/{clip_id}.{cam_name}.mp4"
        if os.path.exists(video_path):
            frames = load_video_frames(video_path, num_frames=4)
            image_frames_list.append(frames)
    
    image_frames = torch.stack(image_frames_list, dim=0)
    
    # 转换到本地坐标系
    t0_idx = np.argmin(np.abs(ego_data[:, 0] - t0_us))
    t0_xyz = ego_data[t0_idx, 1:4]
    t0_quat = ego_data[t0_idx, 4:8]
    t0_rot = spt.Rotation.from_quat(t0_quat)
    t0_rot_inv = t0_rot.inv()
    
    hist_local = [t0_rot_inv.apply(np.array(h) - t0_xyz).tolist() for h in hist_xyz]
    future_local = [t0_rot_inv.apply(np.array(f) - t0_xyz).tolist() for f in future_xyz]
    
    # 运行推理
    frames = image_frames.flatten(0, 1)
    messages = helper.create_message(frames)
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        continue_final_message=True, return_dict=True, return_tensors="pt",
    )
    
    ego_history_xyz = torch.tensor(hist_local).float().unsqueeze(0).unsqueeze(0)
    ego_history_rot = torch.eye(3).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(1, 1, 16, 1, 1)
    
    model_inputs = {
        "tokenized_data": {k: v.to("cuda") for k, v in inputs.items()},
        "ego_history_xyz": ego_history_xyz.to("cuda"),
        "ego_history_rot": ego_history_rot.to("cuda"),
    }
    
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs, top_p=0.98, temperature=0.6,
                num_traj_samples=6, max_generation_length=256, return_extra=True,
            )
    
    # 获取预测轨迹和minADE
    pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2]
    gt_xy_local = np.array(future_local)[:, :2]
    
    diff = np.linalg.norm(pred_xy - gt_xy_local[None, ...], axis=2).mean(-1)
    best_idx = diff.argmin()
    best_pred = pred_xy[best_idx]
    
    coc_text = extra["cot"][0][0][0] if "cot" in extra else "N/A"
    min_ade = diff.min()
    
    print(f"  ✅ minADE: {min_ade:.2f}m")
    print(f"  📝 CoC: {coc_text[:60]}...")
    
    # 4. 创建综合可视化图（改进布局）
    print("\n4. 创建可视化图表...")
    
    fig = plt.figure(figsize=(22, 14))
    
    # 使用 GridSpec 创建更灵活的布局
    # 第1行：4个摄像头（各占1/4宽度）
    # 第2行：左边信息面板（占1/3），右边轨迹图（占2/3）
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 1.8])
    
    # 添加总标题（在Figure顶部，不占用网格空间）
    fig.suptitle(f'Scene Analysis: {clip_id[:25]}... | t0={t0_us/1e6:.1f}s | minADE={min_ade:.2f}m', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # 第一行：4个摄像头图像（每列占1.5个grid单元）
    cam_labels = ["01_left_cross", "02_front_wide", "03_right_cross", "04_front_tele"]
    cam_titles = ["Left Cross Camera", "Front Wide Camera", "Right Cross Camera", "Front Tele Camera"]
    for i, (label, title) in enumerate(zip(cam_labels, cam_titles)):
        if label in camera_images:
            # 每个摄像头占 columns i*1.5 : (i+1)*1.5
            col_start = i * 1
            col_end = (i + 1) * 1
            ax = fig.add_subplot(gs[0, col_start:col_end+1])
            ax.imshow(camera_images[label])
            ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
            ax.axis('off')
    
    # 第二行左边：Scene Analysis Report 信息面板（占2列）
    ax_info = fig.add_subplot(gs[1, 0:2])
    ax_info.axis('off')
    
    info_text = f"""Scene Analysis Report
{'='*50}

Clip ID:
{clip_id}

Timestamp: t0 = {t0_us/1e6:.1f}s

Inference Results:
  • minADE: {min_ade:.4f} meters
  • Mean ADE: {diff.mean():.4f} meters  
  • Trajectory samples: 6
  • Best index: {best_idx}

Chain-of-Causation (CoC):
{coc_text}

{'='*50}"""
    
    ax_info.text(0.05, 0.95, info_text, fontsize=10, verticalalignment='top',
                 fontfamily='monospace', linespacing=1.3,
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3, edgecolor='blue'))
    ax_info.set_xlim(0, 1)
    ax_info.set_ylim(0, 1)
    
    # 第二行右边：轨迹对比图（占4列）
    ax_traj = fig.add_subplot(gs[1, 2:6])
    
    # 旋转坐标
    gt_xy_rot = rotate_90cc(gt_xy_local.T)
    
    # 绘制所有预测轨迹
    for i in range(pred_xy.shape[0]):
        pred_xy_rot = rotate_90cc(pred_xy[i].T)
        if i == best_idx:
            ax_traj.plot(*pred_xy_rot, "o-", color='green', linewidth=2.5, markersize=4,
                        label=f"Predicted #{i+1} (Best, minADE={diff[i]:.2f}m)")
        else:
            ax_traj.plot(*pred_xy_rot, "o-", alpha=0.4, color='blue', markersize=3,
                        label=f"Predicted #{i+1}" if i < 2 else "")
    
    # 绘制真实轨迹
    ax_traj.plot(*gt_xy_rot, "r-", linewidth=3, markersize=5, label="Ground Truth")
    
    # 标记当前位置
    ax_traj.scatter(0, 0, c='black', s=250, marker='*', label='Current Position', zorder=10, edgecolors='white', linewidths=1)
    
    # 设置坐标轴范围（固定）
    ax_traj.set_xlim(-30, 30)  # 横轴：-30m 到 +30m
    ax_traj.set_ylim(0, 100)   # 纵轴：0m 到 100m
    
    ax_traj.set_xlabel('X coordinate (meters)', fontsize=12, fontweight='bold')
    ax_traj.set_ylabel('Y coordinate (meters)', fontsize=12, fontweight='bold')
    ax_traj.set_title('Trajectory Comparison (BEV View)', fontsize=13, fontweight='bold', pad=10)
    ax_traj.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax_traj.grid(True, alpha=0.3, linestyle='--')
    ax_traj.set_aspect('equal')
    
    # 添加箭头指示行驶方向
    ax_traj.annotate('', xy=(0, 10), xytext=(0, 2),
                    arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2))
    ax_traj.text(2, 6, 'Forward', fontsize=9, color='darkgreen', fontweight='bold')
    
    # 添加统计信息（放在轨迹图内）
    stats_text = f"""Statistics:
minADE: {min_ade:.4f}m
Mean ADE: {diff.mean():.4f}m
Std ADE: {diff.std():.4f}m"""
    ax_traj.text(0.98, 0.02, stats_text, transform=ax_traj.transAxes, fontsize=9,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为总标题留出空间
    
    # 保存图表
    viz_path = f"{output_dir}/complete_analysis.png"
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"  ✅ 综合可视化图已保存: {viz_path}")
    plt.close()
    
    # 5. 单独保存轨迹对比图
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 绘制所有预测轨迹
    for i in range(pred_xy.shape[0]):
        pred_xy_rot = rotate_90cc(pred_xy[i].T)
        if i == best_idx:
            ax.plot(*pred_xy_rot, "o-", color='green', linewidth=2, 
                   label=f"Predicted #{i+1} (Best, minADE={diff[i]:.2f}m)")
        else:
            ax.plot(*pred_xy_rot, "o-", alpha=0.3, color='blue',
                   label=f"Predicted #{i+1}" if i < 3 else "")
    
    ax.plot(*gt_xy_rot, "r-", linewidth=3, label="Ground Truth")
    ax.scatter(0, 0, c='black', s=300, marker='*', label='Current Position', zorder=10)
    
    # 设置坐标轴范围（固定）
    ax.set_xlim(-30, 30)  # 横轴：-30m 到 +30m
    ax.set_ylim(0, 100)   # 纵轴：0m 到 100m
    
    ax.set_xlabel('X coordinate (meters)', fontsize=14)
    ax.set_ylabel('Y coordinate (meters)', fontsize=14)
    ax.set_title(f'Trajectory Comparison\nClip: {clip_id[:20]}... | minADE: {min_ade:.2f}m', 
                fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    traj_path = f"{output_dir}/trajectory_comparison.png"
    plt.savefig(traj_path, dpi=150, bbox_inches='tight')
    print(f"  ✅ 轨迹对比图已保存: {traj_path}")
    plt.close()
    
    # 6. 保存数据到JSON
    print("\n5. 保存数据到JSON...")
    analysis_data = {
        "clip_id": clip_id,
        "t0_us": int(t0_us),
        "t0_seconds": t0_us / 1e6,
        "minADE": float(min_ade),
        "meanADE": float(diff.mean()),
        "stdADE": float(diff.std()),
        "CoC": coc_text,
        "camera_order": ["left_cross", "front_wide", "right_cross", "front_tele"],
        "history_trajectory_xyz_local": hist_local,
        "future_ground_truth_xyz_local": future_local,
        "predicted_trajectories_xy_local": [pred_xy[i].tolist() for i in range(pred_xy.shape[0])],
        "best_trajectory_index": int(best_idx),
    }
    
    with open(f"{output_dir}/analysis_data.json", "w") as f:
        json.dump(analysis_data, f, indent=2)
    
    # 7. 创建说明文档
    readme = f"""# 场景分析数据包

## 基本信息
- **Clip ID**: {clip_id}
- **时间点**: t0 = {t0_us/1e6:.1f}s (视频内)
- **视频时间范围**: {video_start/1e6:.2f}s ~ {video_end/1e6:.2f}s

## 推理结果
- **minADE**: {min_ade:.4f}m
- **meanADE**: {diff.mean():.4f}m
- **stdADE**: {diff.std():.4f}m
- **最佳轨迹索引**: {best_idx}
- **CoC (Chain-of-Causation)**: 
  > {coc_text}

## 文件结构
```
scene_analysis_{clip_id[:8]}/
├── camera_images/           # 4个摄像头原始图像
│   ├── 01_left_cross.png
│   ├── 02_front_wide.png
│   ├── 03_right_cross.png
│   └── 04_front_tele.png
├── complete_analysis.png    # 综合可视化图（所有内容）
├── trajectory_comparison.png # 轨迹对比图（大图）
├── analysis_data.json       # 完整数据（轨迹坐标、结果）
└── README.md                # 本文件
```

## 数据说明

### history_trajectory (历史轨迹)
- 时间步数: 16步
- 时间跨度: 1.6秒 (t0-1.6s ~ t0)
- 坐标系: 本地坐标系（以t0时刻车辆位置为原点）

### future_ground_truth (未来真值)
- 时间步数: 64步
- 时间跨度: 6.4秒 (t0+0.1s ~ t0+6.4s)
- 坐标系: 本地坐标系

### predicted_trajectories (预测轨迹)
- 样本数: 6条
- 最佳轨迹: 索引 {best_idx} (minADE最小)
- 坐标: XY平面 (Z忽略)

## 可视化说明

**complete_analysis.png** 包含：
1. 第一行: 4个摄像头的图像
2. 第二行: 场景信息面板（Clip ID、minADE、CoC文本）
3. 第三行: BEV视角的轨迹对比图
   - 绿色: 最佳预测轨迹
   - 蓝色: 其他预测轨迹
   - 红色: 真实轨迹
   - 黑色星标: 当前车辆位置

**trajectory_comparison.png** 包含：
- 放大的轨迹对比图
- 更清晰的轨迹细节

## 生成时间
{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    with open(f"{output_dir}/README.md", "w") as f:
        f.write(readme)
    
    print(f"\n" + "=" * 70)
    print(f"数据包创建完成!")
    print(f"位置: {output_dir}")
    print(f"内容:")
    print(f"  - camera_images/: 4个摄像头图像")
    print(f"  - complete_analysis.png: 综合可视化图")
    print(f"  - trajectory_comparison.png: 轨迹对比图")
    print(f"  - analysis_data.json: 轨迹数据和推理结果")
    print(f"  - README.md: 说明文档")
    print("=" * 70)

if __name__ == "__main__":
    main()
