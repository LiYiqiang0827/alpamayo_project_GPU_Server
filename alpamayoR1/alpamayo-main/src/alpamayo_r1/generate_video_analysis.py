#!/usr/bin/env python3
"""
对整个视频进行推理，生成每帧的complete_analysis图，最后合成MP4视频
"""
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import pandas as pd
import numpy as np
import torch
import av
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
from pathlib import Path
from PIL import Image
import scipy.spatial.transform as spt
from einops import rearrange
import warnings
warnings.filterwarnings('ignore')

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

def create_complete_analysis_figure(camera_images, clip_id, t0_us, min_ade, diff, 
                                     pred_xy, gt_xy_local, best_idx, coc_text, 
                                     video_start, video_end):
    """创建单帧的complete_analysis图"""
    fig = plt.figure(figsize=(22, 14))
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 1.8])
    
    # 总标题
    fig.suptitle(f'Scene Analysis: {clip_id[:25]}... | t0={t0_us/1e6:.2f}s | minADE={min_ade:.2f}m', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # 第一行：4个摄像头图像
    cam_labels = ["01_left_cross", "02_front_wide", "03_right_cross", "04_front_tele"]
    cam_titles = ["Left Cross Camera", "Front Wide Camera", "Right Cross Camera", "Front Tele Camera"]
    for i, (label, title) in enumerate(zip(cam_labels, cam_titles)):
        if label in camera_images:
            col_start = i * 1
            col_end = (i + 1) * 1
            ax = fig.add_subplot(gs[0, col_start:col_end+1])
            ax.imshow(camera_images[label])
            ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
            ax.axis('off')
    
    # 第二行左边：信息面板
    ax_info = fig.add_subplot(gs[1, 0:2])
    ax_info.axis('off')
    
    info_text = f"""Scene Analysis Report
{'='*50}

Clip ID:
{clip_id}

Timestamp: t0 = {t0_us/1e6:.2f}s
Video Range: {video_start/1e6:.1f}s ~ {video_end/1e6:.1f}s

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
    
    # 第二行右边：轨迹对比图
    ax_traj = fig.add_subplot(gs[1, 2:6])
    
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
    
    ax_traj.plot(*gt_xy_rot, "r-", linewidth=3, markersize=5, label="Ground Truth")
    ax_traj.scatter(0, 0, c='black', s=250, marker='*', label='Current Position', 
                   zorder=10, edgecolors='white', linewidths=1)
    
    ax_traj.set_xlim(-30, 30)
    ax_traj.set_ylim(0, 100)
    
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
    
    # 添加统计信息
    stats_text = f"""Statistics:
minADE: {min_ade:.4f}m
Mean ADE: {diff.mean():.4f}m
Std ADE: {diff.std():.4f}m"""
    ax_traj.text(0.98, 0.02, stats_text, transform=ax_traj.transAxes, fontsize=9,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    return fig

def process_single_timestamp(args):
    """处理单个时间戳（用于并行处理）"""
    (base_dir, clip_id, t0_us, model, processor, video_start, video_end, 
     output_frames_dir, cameras) = args
    
    from alpamayo_r1 import helper
    
    try:
        # 加载ego motion数据
        ego_path = f"{base_dir}/labels/egomotion/{clip_id}.egomotion.parquet"
        ego_data = load_egomotion(ego_path)
        
        # 计算历史和未来轨迹
        history_timestamps = t0_us + np.arange(-15, 1, 1) * 100000
        future_timestamps = t0_us + np.arange(1, 65, 1) * 100000
        
        hist_xyz = []
        for t in history_timestamps:
            idx = np.argmin(np.abs(ego_data[:, 0] - t))
            hist_xyz.append(ego_data[idx, 1:4].tolist())
        
        future_xyz = []
        for t in future_timestamps:
            idx = np.argmin(np.abs(ego_data[:, 0] - t))
            future_xyz.append(ego_data[idx, 1:4].tolist())
        
        # 提取4摄像头图像
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
                    camera_images[label] = frame
        
        # 准备模型输入
        image_frames_list = []
        for cam_name, cam_idx in [("camera_cross_left_120fov", 0), 
                                   ("camera_front_wide_120fov", 1),
                                   ("camera_cross_right_120fov", 2), 
                                   ("camera_front_tele_30fov", 6)]:
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
        coc_text = extra["cot"][0][0][0] if "cot" in extra else "N/A"
        min_ade = diff.min()
        
        # 创建可视化图
        fig = create_complete_analysis_figure(
            camera_images, clip_id, t0_us, min_ade, diff, 
            pred_xy, gt_xy_local, best_idx, coc_text,
            video_start, video_end
        )
        
        # 保存图片
        frame_filename = f"frame_{t0_us:010d}.png"
        frame_path = f"{output_frames_dir}/{frame_filename}"
        fig.savefig(frame_path, dpi=120, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        return {
            't0_us': t0_us,
            'min_ade': float(min_ade),
            'coc': coc_text,
            'success': True,
            'frame_file': frame_filename
        }
        
    except Exception as e:
        print(f"  ❌ Error at t0={t0_us}: {e}")
        return {'t0_us': t0_us, 'success': False, 'error': str(e)}

def main():
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from alpamayo_r1 import helper
    
    base_dir = "/data01/vla/data_sample_chunk0"
    clip_id = "0c8731a8-9cd4-4f59-9603-165b1ed53e07"  # 换一个表现好的clip
    
    # 获取视频时间范围
    ts_path = f"{base_dir}/camera/camera_front_wide_120fov/{clip_id}.camera_front_wide_120fov.timestamps.parquet"
    df_ts = pd.read_parquet(ts_path)
    ts = df_ts['timestamp'].values
    video_start = ts[0]
    video_end = ts[-1]
    video_duration = (video_end - video_start) / 1e6
    
    print("=" * 70)
    print(f"完整视频推理分析")
    print("=" * 70)
    print(f"Clip ID: {clip_id}")
    print(f"视频时长: {video_duration:.1f}s")
    print(f"视频帧数: {len(ts)} frames @ 30 FPS")
    print()
    
    # 创建输出目录
    output_base = f"/data01/vla/video_analysis_{clip_id[:8]}"
    frames_dir = f"{output_base}/frames"
    os.makedirs(frames_dir, exist_ok=True)
    
    # 计算采样时间点（每0.3秒一帧，更流畅）
    sample_interval = 0.3  # 秒
    num_frames = int(video_duration / sample_interval) + 1
    t0_list = [video_start + int(i * sample_interval * 1e6) for i in range(num_frames)]
    # 过滤超出范围的
    t0_list = [t for t in t0_list if t <= video_end - 6.5e6]  # 留出6.5秒用于未来轨迹
    
    print(f"将处理 {len(t0_list)} 个时间点 (每{sample_interval}s采样)")
    print(f"输出目录: {output_base}")
    print()
    
    # 加载模型（只加载一次）
    print("加载模型...")
    model = AlpamayoR1.from_pretrained(
        "nvidia/Alpamayo-R1-10B",
        dtype=torch.bfloat16,
        local_files_only=True,
    ).to("cuda")
    processor = helper.get_processor(model.tokenizer)
    print("模型加载完成!\n")
    
    # 摄像头配置
    cameras = [
        ("camera_cross_left_120fov", "01_left_cross"),
        ("camera_front_wide_120fov", "02_front_wide"),
        ("camera_cross_right_120fov", "03_right_cross"),
        ("camera_front_tele_30fov", "04_front_tele"),
    ]
    
    # 处理每个时间点
    results = []
    for i, t0_us in enumerate(t0_list):
        print(f"处理帧 {i+1}/{len(t0_list)}: t0={t0_us/1e6:.2f}s ...", end=' ')
        
        result = process_single_timestamp(
            (base_dir, clip_id, t0_us, model, processor, video_start, video_end,
             frames_dir, cameras)
        )
        
        if result['success']:
            print(f"✅ minADE={result['min_ade']:.2f}m")
        else:
            print(f"❌ Failed")
        
        results.append(result)
    
    # 保存结果摘要
    summary = {
        'clip_id': clip_id,
        'video_duration': float(video_duration),
        'num_frames_processed': len(t0_list),
        'sample_interval': sample_interval,
        'results': [{k: (int(v) if isinstance(v, (np.int64, np.int32)) else 
                      float(v) if isinstance(v, (np.float64, np.float32)) else v)
                    for k, v in r.items()} for r in results]
    }
    
    with open(f"{output_base}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 70)
    print(f"图片生成完成!")
    print(f"共生成 {len([r for r in results if r['success']])} 帧")
    print(f"帧图片位置: {frames_dir}")
    print("=" * 70)
    
    # 合成视频
    print("\n正在合成MP4视频...")
    output_video = f"{output_base}/{clip_id}_result.mp4"
    
    # 使用av库合成视频
    import av
    import glob
    from PIL import Image
    
    frame_files = sorted(glob.glob(f"{frames_dir}/frame_*.png"))
    if len(frame_files) > 0:
        first_frame = Image.open(frame_files[0])
        width, height = first_frame.size
        
        container = av.open(output_video, mode='w')
        stream = container.add_stream('mpeg4', rate=3)  # 3 FPS
        stream.width = width
        stream.height = height
        stream.pix_fmt = 'yuv420p'
        
        for i, frame_file in enumerate(frame_files):
            img = Image.open(frame_file).convert('RGB')
            frame_array = np.array(img)
            frame = av.VideoFrame.from_ndarray(frame_array, format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)
            if (i+1) % 10 == 0:
                print(f"  已写入 {i+1}/{len(frame_files)} 帧")
        
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        
        video_size = os.path.getsize(output_video) / (1024 * 1024)
        print(f"\n✅ 视频合成成功!")
        print(f"视频文件: {output_video}")
        print(f"视频大小: {video_size:.1f} MB")
        print(f"视频时长: ~{len(frame_files)/3:.1f}s (3 FPS)")
    else:
        print(f"\n❌ 没有找到帧文件")
    
    print("=" * 70)

if __name__ == "__main__":
    main()
