#!/usr/bin/env python3
"""
完整视频推理 - 对单个clip的所有时间点进行推理
"""
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import pandas as pd
import numpy as np
import torch
import av
from pathlib import Path
import scipy.spatial.transform as spt
from einops import rearrange
import matplotlib.pyplot as plt

def load_video_frames(video_path: str, num_frames: int = 4):
    """从本地视频文件加载帧"""
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
    translations = df[['x', 'y', 'z']].values
    rotations = df[['qx', 'qy', 'qz', 'qw']].values
    timestamps = df['timestamp'].values
    return translations, rotations, timestamps

def interpolate_egomotion(translations, rotations, timestamps, target_timestamps):
    interp_xyz = np.zeros((len(target_timestamps), 3))
    for i in range(3):
        interp_xyz[:, i] = np.interp(target_timestamps, timestamps, translations[:, i])
    
    interp_quat = np.zeros((len(target_timestamps), 4))
    for i in range(4):
        interp_quat[:, i] = np.interp(target_timestamps, timestamps, rotations[:, i])
    
    norms = np.linalg.norm(interp_quat, axis=1, keepdims=True)
    interp_quat = interp_quat / norms
    
    return interp_xyz, interp_quat

def load_data_at_timestamp(base_dir: str, clip_id: str, t0_us: int):
    """加载指定时间戳的数据"""
    ego_path = f"{base_dir}/labels/egomotion/{clip_id}.egomotion.parquet"
    translations, rotations, timestamps = load_egomotion(ego_path)
    
    # 检查t0是否在有效范围
    if t0_us < timestamps[0] + 1600000 or t0_us > timestamps[-1] - 6400000:
        return None  # 需要至少1.6s历史和6.4s未来
    
    # 计算时间戳
    history_timestamps = t0_us + np.arange(-15, 1, 1) * 100000
    future_timestamps = t0_us + np.arange(1, 65, 1) * 100000
    
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
    ego_history_rot_local = (t0_rot_inv * spt.Rotation.from_quat(ego_history_quat)).as_matrix()
    
    # 加载摄像头图像
    cameras = [
        ("camera_cross_left_120fov", 0),
        ("camera_front_wide_120fov", 1),
        ("camera_cross_right_120fov", 2),
        ("camera_front_tele_30fov", 6),
    ]
    
    image_frames_list = []
    camera_indices_list = []
    
    for cam_name, cam_idx in cameras:
        video_path = f"{base_dir}/camera/{cam_name}/{clip_id}.{cam_name}.mp4"
        if os.path.exists(video_path):
            try:
                frames = load_video_frames(video_path, num_frames=4)
                image_frames_list.append(frames)
                camera_indices_list.append(cam_idx)
            except Exception as e:
                print(f"  Warning: Failed to load {cam_name}: {e}")
    
    if not image_frames_list:
        return None
    
    image_frames = torch.stack(image_frames_list, dim=0)
    camera_indices = torch.tensor(camera_indices_list, dtype=torch.int64)
    
    sort_order = torch.argsort(camera_indices)
    image_frames = image_frames[sort_order]
    camera_indices = camera_indices[sort_order]
    
    return {
        "image_frames": image_frames,
        "camera_indices": camera_indices,
        "ego_history_xyz": torch.from_numpy(ego_history_xyz_local).float().unsqueeze(0).unsqueeze(0),
        "ego_history_rot": torch.from_numpy(ego_history_rot_local).float().unsqueeze(0).unsqueeze(0),
        "ego_future_xyz": torch.from_numpy(ego_future_xyz_local).float().unsqueeze(0).unsqueeze(0),
        "t0_us": t0_us,
    }

def run_inference(model, processor, data, helper, num_traj_samples=6):
    """运行推理"""
    frames = data["image_frames"].flatten(0, 1)
    messages = helper.create_message(frames)
    
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    
    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    
    model_inputs = helper.to_device(model_inputs, "cuda")
    
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs,
                top_p=0.98,
                temperature=0.6,
                num_traj_samples=num_traj_samples,
                max_generation_length=256,
                return_extra=True,
            )
    
    gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
    pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
    diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
    min_ade = diff.min()
    
    coc_text = extra["cot"][0][0][0] if "cot" in extra else "N/A"
    
    return min_ade, coc_text

def visualize_results(results, clip_id, save_path):
    """可视化推理结果"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    times = [r["time_sec"] for r in results]
    minADEs = [r["minADE"] for r in results]
    
    # 上：minADE 随时间变化
    axes[0].plot(times, minADEs, 'b-o', linewidth=2, markersize=6)
    axes[0].axhline(y=np.mean(minADEs), color='r', linestyle='--', label=f'Mean: {np.mean(minADEs):.2f}m')
    axes[0].fill_between(times, minADEs, alpha=0.3)
    axes[0].set_xlabel('Time (seconds)', fontsize=12)
    axes[0].set_ylabel('minADE (meters)', fontsize=12)
    axes[0].set_title(f'Complete Video Inference: {clip_id[:20]}...\nAverage minADE: {np.mean(minADEs):.2f}m', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 下：minADE 分布直方图
    axes[1].hist(minADEs, bins=15, color='skyblue', edgecolor='black', alpha=0.7)
    axes[1].axvline(np.mean(minADEs), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(minADEs):.2f}m')
    axes[1].axvline(np.median(minADEs), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(minADEs):.2f}m')
    axes[1].set_xlabel('minADE (meters)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('minADE Distribution', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"可视化结果已保存: {save_path}")
    plt.close()

def main():
    print("=" * 70)
    print("完整视频推理 - 单个clip所有时间点")
    print("=" * 70)
    
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from alpamayo_r1 import helper
    
    print("\n加载模型...")
    model = AlpamayoR1.from_pretrained(
        "nvidia/Alpamayo-R1-10B",
        dtype=torch.bfloat16,
        local_files_only=True,
    ).to("cuda")
    processor = helper.get_processor(model.tokenizer)
    print("模型加载完成!")
    
    base_dir = "/data01/vla/data_sample_chunk0"
    clip_id = "0c8731a8-9cd4-4f59-9603-165b1ed53e07"  # 选择一个样本
    
    # 获取可用时间范围
    ego_path = f"{base_dir}/labels/egomotion/{clip_id}.egomotion.parquet"
    ego = pd.read_parquet(ego_path)
    ego_ts = ego['timestamp'].values
    
    # 有效时间范围：需要1.6s历史和6.4s未来
    valid_start = ego_ts[0] + 1600000 + 300000  # +3秒缓冲
    valid_end = ego_ts[-1] - 6400000 - 300000   # -3秒缓冲
    
    print(f"\n视频: {clip_id}")
    print(f"Ego Motion 时间范围: {ego_ts[0]/1e6:.2f}s to {ego_ts[-1]/1e6:.2f}s")
    print(f"有效推理时间范围: {valid_start/1e6:.2f}s to {valid_end/1e6:.2f}s")
    
    # 每1秒采样一个时间点
    sample_interval = 1000000  # 1秒 = 1,000,000 us
    timestamps = np.arange(valid_start, valid_end, sample_interval)
    
    print(f"\n将推理 {len(timestamps)} 个时间点...")
    print(f"采样间隔: {sample_interval/1e6:.1f}秒")
    print()
    
    results = []
    for i, t0_us in enumerate(timestamps):
        time_sec = (t0_us - ego_ts[0]) / 1e6  # 相对开始时间
        print(f"[{i+1:2d}/{len(timestamps)}] t={time_sec:.1f}s...", end=" ")
        
        try:
            data = load_data_at_timestamp(base_dir, clip_id, t0_us)
            if data is None:
                print("❌ 时间戳超出范围")
                continue
            
            min_ade, coc = run_inference(model, processor, data, helper, num_traj_samples=6)
            
            results.append({
                "time_sec": time_sec,
                "t0_us": t0_us,
                "minADE": min_ade,
                "CoC": coc,
            })
            
            print(f"✅ minADE={min_ade:.2f}m")
            
        except Exception as e:
            print(f"❌ {str(e)[:50]}")
        
        # 每5个清理显存
        if (i + 1) % 5 == 0:
            torch.cuda.empty_cache()
    
    # 输出汇总
    print("\n" + "=" * 70)
    print("完整视频推理结果汇总")
    print("=" * 70)
    
    if results:
        minADEs = [r["minADE"] for r in results]
        print(f"\n成功推理: {len(results)}/{len(timestamps)} 个时间点")
        print(f"平均 minADE: {np.mean(minADEs):.4f} 米")
        print(f"中位数 minADE: {np.median(minADEs):.4f} 米")
        print(f"标准差: {np.std(minADEs):.4f} 米")
        print(f"范围: [{np.min(minADEs):.4f}, {np.max(minADEs):.4f}] 米")
        
        print("\n各时间点结果:")
        print("-" * 70)
        for r in results:
            print(f"t={r['time_sec']:5.1f}s | minADE={r['minADE']:6.2f}m | {r['CoC'][:50]}...")
    
    # 生成可视化
    if results:
        output_dir = "/data01/vla/visualizations"
        os.makedirs(output_dir, exist_ok=True)
        viz_path = f"{output_dir}/complete_video_{clip_id}.png"
        visualize_results(results, clip_id, viz_path)
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
