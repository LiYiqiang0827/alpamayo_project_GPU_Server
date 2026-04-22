#!/usr/bin/env python3
"""
Alpamayo R1 可视化脚本 - 生成轨迹预测可视化结果
"""
import os
import glob
import numpy as np
import pandas as pd
import torch
import av
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import Any, List, Tuple
import scipy.spatial.transform as spt
from einops import rearrange

# 设置离线模式
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_video_frame_at_time(video_path: str, target_time_ms: float = 0):
    """从视频加载特定时间点的帧"""
    container = av.open(video_path)
    stream = container.streams.video[0]
    
    # 获取帧
    for frame in container.decode(video=0):
        img = frame.to_ndarray(format='rgb24')
        break  # 只取第一帧
    
    container.close()
    return img


def load_egomotion(ego_path: str):
    """从本地 parquet 加载 ego motion 数据"""
    df = pd.read_parquet(ego_path)
    translations = df[['x', 'y', 'z']].values
    rotations = df[['qx', 'qy', 'qz', 'qw']].values
    timestamps = df['timestamp'].values
    return translations, rotations, timestamps


def interpolate_egomotion(translations, rotations, timestamps, target_timestamps):
    """插值 ego motion"""
    interp_xyz = np.zeros((len(target_timestamps), 3))
    for i in range(3):
        interp_xyz[:, i] = np.interp(target_timestamps, timestamps, translations[:, i])
    
    interp_quat = np.zeros((len(target_timestamps), 4))
    for i in range(4):
        interp_quat[:, i] = np.interp(target_timestamps, timestamps, rotations[:, i])
    
    norms = np.linalg.norm(interp_quat, axis=1, keepdims=True)
    interp_quat = interp_quat / norms
    
    return interp_xyz, interp_quat


def load_local_data(
    base_dir: str,
    clip_id: str,
    t0_us: int = 5_100_000,
    num_history_steps: int = 16,
    num_future_steps: int = 64,
    time_step: float = 0.1,
):
    """加载本地数据"""
    # 加载 ego motion
    ego_path = f"{base_dir}/labels/egomotion/{clip_id}.egomotion.parquet"
    translations, rotations, timestamps = load_egomotion(ego_path)
    
    # 计算轨迹时间戳
    history_offsets_us = np.arange(
        -(num_history_steps - 1) * time_step * 1_000_000,
        time_step * 1_000_000 / 2,
        time_step * 1_000_000,
    ).astype(np.int64)
    history_timestamps = t0_us + history_offsets_us
    
    future_offsets_us = np.arange(
        time_step * 1_000_000,
        (num_future_steps + 0.5) * time_step * 1_000_000,
        time_step * 1_000_000,
    ).astype(np.int64)
    future_timestamps = t0_us + future_offsets_us
    
    # 插值
    ego_history_xyz, ego_history_quat = interpolate_egomotion(
        translations, rotations, timestamps, history_timestamps
    )
    ego_future_xyz, ego_future_quat = interpolate_egomotion(
        translations, rotations, timestamps, future_timestamps
    )
    
    # 转换到本地坐标系
    t0_xyz = ego_history_xyz[-1].copy()
    t0_quat = ego_history_quat[-1].copy()
    t0_rot = spt.Rotation.from_quat(t0_quat)
    t0_rot_inv = t0_rot.inv()
    
    ego_history_xyz_local = t0_rot_inv.apply(ego_history_xyz - t0_xyz)
    ego_future_xyz_local = t0_rot_inv.apply(ego_future_xyz - t0_xyz)
    
    # 加载摄像头帧
    cameras = [
        ("camera_cross_left_120fov", "Left"),
        ("camera_front_wide_120fov", "Front"),
        ("camera_cross_right_120fov", "Right"),
    ]
    
    camera_frames = {}
    for cam_name, label in cameras:
        video_path = f"{base_dir}/camera/{cam_name}/{clip_id}.{cam_name}.mp4"
        if os.path.exists(video_path):
            frame = load_video_frame_at_time(video_path)
            camera_frames[label] = frame
    
    return {
        "ego_history_xyz": ego_history_xyz_local,
        "ego_future_xyz": ego_future_xyz_local,
        "camera_frames": camera_frames,
        "clip_id": clip_id,
    }


def run_inference(model, processor, data, helper):
    """运行推理"""
    from alpamayo_r1 import helper as h
    
    # 准备简化输入（这里需要实际的图像帧）
    # 由于可视化主要关注轨迹，我们先使用已有的数据进行推理
    
    # 创建模拟输入（实际应用中应该加载真实图像）
    batch_size = 1
    num_cameras = 4
    num_frames = 4
    
    # 使用实际的轨迹数据
    ego_history_xyz = torch.from_numpy(data["ego_history_xyz"]).float().unsqueeze(0).unsqueeze(0)
    ego_history_rot = torch.eye(3).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(1, 1, 16, 1, 1)
    
    # 这里简化处理，实际应该加载图像
    # 返回真实轨迹作为基准
    return data["ego_future_xyz"]


def visualize_trajectory(data: dict, pred_xyz: np.ndarray = None, save_path: str = None):
    """可视化轨迹"""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.2])
    
    # 上方：摄像头视图
    camera_labels = ["Left", "Front", "Right"]
    for i, label in enumerate(camera_labels):
        if label in data["camera_frames"]:
            ax = fig.add_subplot(gs[0, i])
            frame = data["camera_frames"][label]
            ax.imshow(frame)
            ax.set_title(f"{label} Camera", fontsize=12, fontweight='bold')
            ax.axis('off')
    
    # 下方：BEV 轨迹图
    ax_bev = fig.add_subplot(gs[1, :])
    
    # 提取历史轨迹 (X, Y) - 本地坐标系
    hist_xy = data["ego_history_xyz"][:, [0, 1]]  # (16, 2)
    gt_future_xy = data["ego_future_xyz"][:, [0, 1]]  # (64, 2)
    
    # 绘制历史轨迹
    ax_bev.plot(hist_xy[:, 0], hist_xy[:, 1], 'b-', linewidth=2, label='History Trajectory', marker='o', markersize=3)
    
    # 绘制真实未来轨迹
    ax_bev.plot(gt_future_xy[:, 0], gt_future_xy[:, 1], 'g-', linewidth=2, label='Ground Truth Future', marker='s', markersize=2)
    
    # 如果有预测轨迹，也绘制
    if pred_xyz is not None:
        pred_xy = pred_xyz[0, 0, :, [0, 1]]  # (64, 2)
        ax_bev.plot(pred_xy[:, 0], pred_xy[:, 1], 'r--', linewidth=2, label='Predicted Future', marker='^', markersize=2)
    
    # 标记当前位置
    ax_bev.scatter(0, 0, c='black', s=200, marker='*', label='Current Position', zorder=10)
    
    # 设置坐标轴
    ax_bev.set_xlabel('X (meters)', fontsize=12)
    ax_bev.set_ylabel('Y (meters)', fontsize=12)
    ax_bev.set_title(f'BEV Trajectory Visualization\nClip: {data["clip_id"][:20]}...', fontsize=14, fontweight='bold')
    ax_bev.legend(loc='best', fontsize=10)
    ax_bev.grid(True, alpha=0.3)
    ax_bev.set_aspect('equal')
    
    # 添加方向箭头
    ax_bev.annotate('', xy=(hist_xy[-1, 0] - hist_xy[-2, 0], hist_xy[-1, 1] - hist_xy[-2, 1]), 
                    xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"可视化结果已保存: {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_multiple_samples(base_dir: str, output_dir: str, num_samples: int = 5):
    """批量可视化多个样本"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取样本列表
    ego_files = glob.glob(f"{base_dir}/labels/egomotion/*.parquet")
    clip_ids = [Path(f).stem.split('.')[0] for f in ego_files[:num_samples]]
    
    print(f"将生成 {len(clip_ids)} 个样本的可视化...")
    
    for i, clip_id in enumerate(clip_ids, 1):
        print(f"[{i}/{len(clip_ids)}] 处理 {clip_id}...")
        
        try:
            # 加载数据
            data = load_local_data(
                base_dir=base_dir,
                clip_id=clip_id,
                t0_us=5_100_000,
            )
            
            # 生成可视化
            save_path = f"{output_dir}/trajectory_{clip_id}.png"
            visualize_trajectory(data, save_path=save_path)
            
        except Exception as e:
            print(f"  失败: {str(e)[:60]}")
    
    print(f"\n所有可视化结果已保存到: {output_dir}")


def create_summary_visualization(results: list, output_path: str):
    """创建汇总统计图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    minADEs = [r["minADE"] for r in results]
    
    # 1. minADE 分布直方图
    axes[0, 0].hist(minADEs, bins=15, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(np.mean(minADEs), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(minADEs):.2f}m')
    axes[0, 0].axvline(np.median(minADEs), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(minADEs):.2f}m')
    axes[0, 0].set_xlabel('minADE (meters)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of minADE')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. minADE 排序图
    sorted_indices = np.argsort(minADEs)
    sorted_minADEs = [minADEs[i] for i in sorted_indices]
    colors = ['green' if m < 3 else 'orange' if m < 6 else 'red' for m in sorted_minADEs]
    axes[0, 1].bar(range(len(minADEs)), sorted_minADEs, color=colors, alpha=0.7)
    axes[0, 1].set_xlabel('Sample Index (sorted)')
    axes[0, 1].set_ylabel('minADE (meters)')
    axes[0, 1].set_title('minADE per Sample (Sorted)')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. 箱线图
    axes[1, 0].boxplot(minADEs, vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.7),
                       medianprops=dict(color='red', linewidth=2))
    axes[1, 0].set_ylabel('minADE (meters)')
    axes[1, 0].set_title('minADE Box Plot')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 统计信息文本
    stats_text = f"""
    Statistics Summary:
    
    Total Samples: {len(minADEs)}
    
    Mean minADE: {np.mean(minADEs):.4f} m
    Median minADE: {np.median(minADEs):.4f} m
    Std Dev: {np.std(minADEs):.4f} m
    
    Min: {np.min(minADEs):.4f} m
    Max: {np.max(minADEs):.4f} m
    
    < 3m (Good): {sum(1 for m in minADEs if m < 3)} samples
    3-6m (Fair): {sum(1 for m in minADEs if 3 <= m < 6)} samples
    > 6m (Poor): {sum(1 for m in minADEs if m >= 6)} samples
    """
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                    fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Summary Statistics')
    
    plt.suptitle('Alpamayo R1 Inference Results Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"汇总图已保存: {output_path}")
    plt.close()


def main():
    print("=" * 60)
    print("Alpamayo R1 可视化生成工具")
    print("=" * 60)
    
    base_dir = "/data01/vla/data_sample_chunk0"
    output_dir = "/data01/vla/visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成 5 个样本的轨迹可视化
    print("\n1. 生成轨迹可视化图...")
    visualize_multiple_samples(base_dir, output_dir, num_samples=5)
    
    # 生成汇总统计图（使用之前的结果数据）
    print("\n2. 生成汇总统计图...")
    # 这里使用之前 20 个样本的结果
    results = [
        {"clip_id": "0c8731a8-...", "minADE": 4.3637},
        {"clip_id": "7e1bc82f-...", "minADE": 1.8776},
        {"clip_id": "0391b4a7-...", "minADE": 4.5161},
        {"clip_id": "dfb6db91-...", "minADE": 1.6945},
        {"clip_id": "4db63321-...", "minADE": 3.9738},
        {"clip_id": "43ada34d-...", "minADE": 0.8434},
        {"clip_id": "7a22a3d6-...", "minADE": 6.6294},
        {"clip_id": "38a3f022-...", "minADE": 3.2862},
        {"clip_id": "d0a0cbf0-...", "minADE": 5.2422},
        {"clip_id": "1f7ef408-...", "minADE": 2.4368},
        {"clip_id": "e9162b7f-...", "minADE": 2.4505},
        {"clip_id": "038678de-...", "minADE": 1.7464},
        {"clip_id": "43c43f34-...", "minADE": 4.0006},
        {"clip_id": "3c16fed6-...", "minADE": 8.1778},
        {"clip_id": "69dfea29-...", "minADE": 5.0372},
        {"clip_id": "4d7920f8-...", "minADE": 0.9229},
        {"clip_id": "fa83bcb8-...", "minADE": 14.5015},
        {"clip_id": "e762aaa3-...", "minADE": 3.9884},
        {"clip_id": "5c8a7587-...", "minADE": 3.7643},
        {"clip_id": "ef4264ed-...", "minADE": 13.6447},
    ]
    
    summary_path = f"{output_dir}/summary_statistics.png"
    create_summary_visualization(results, summary_path)
    
    print("\n" + "=" * 60)
    print("可视化完成!")
    print(f"结果保存在: {output_dir}")
    print("=" * 60)
    print("\n生成的文件:")
    print(f"  - {output_dir}/trajectory_*.png (5个样本的轨迹图)")
    print(f"  - {summary_path} (汇总统计图)")


if __name__ == "__main__":
    main()
