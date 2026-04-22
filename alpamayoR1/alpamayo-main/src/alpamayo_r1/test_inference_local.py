#!/usr/bin/env python3
"""
本地数据推理脚本 - 直接使用本地VLA数据，无需网络连接
"""
import os
import glob
import numpy as np
import pandas as pd
import torch
import av
from pathlib import Path
from typing import Any
import scipy.spatial.transform as spt
from einops import rearrange

# 设置离线模式
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

def load_video_frames(video_path: str, num_frames: int = 4, t0_us: int = None):
    """从本地视频文件加载帧"""
    container = av.open(video_path)
    
    # 获取视频信息
    stream = container.streams.video[0]
    total_frames = stream.frames
    
    # 均匀采样帧
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for i, frame in enumerate(container.decode(video=0)):
        if i in indices:
            # 转换为 numpy array (H, W, C)
            img = frame.to_ndarray(format='rgb24')
            frames.append(img)
        if len(frames) >= num_frames:
            break
    
    container.close()
    
    # 转换为 tensor (T, C, H, W)
    frames = np.stack(frames)  # (T, H, W, C)
    frames = rearrange(frames, "t h w c -> t c h w")
    return torch.from_numpy(frames).float()

def load_egomotion(ego_path: str):
    """从本地 parquet 加载 ego motion 数据"""
    df = pd.read_parquet(ego_path)
    
    # 提取位置和旋转
    translations = df[['x', 'y', 'z']].values  # (N, 3)
    rotations = df[['qx', 'qy', 'qz', 'qw']].values  # (N, 4) 四元数
    timestamps = df['timestamp'].values  # (N,)
    
    return translations, rotations, timestamps

def interpolate_egomotion(translations, rotations, timestamps, target_timestamps):
    """插值 ego motion 到目标时间戳"""
    # 位置线性插值
    interp_xyz = np.zeros((len(target_timestamps), 3))
    for i in range(3):
        interp_xyz[:, i] = np.interp(target_timestamps, timestamps, translations[:, i])
    
    # 四元数球形插值（简化为线性插值后归一化）
    interp_quat = np.zeros((len(target_timestamps), 4))
    for i in range(4):
        interp_quat[:, i] = np.interp(target_timestamps, timestamps, rotations[:, i])
    
    # 归一化四元数
    norms = np.linalg.norm(interp_quat, axis=1, keepdims=True)
    interp_quat = interp_quat / norms
    
    return interp_xyz, interp_quat

def load_local_dataset(
    base_dir: str = "/data01/vla/data_sample_chunk0",
    clip_id: str = None,
    t0_us: int = 5_100_000,
    num_history_steps: int = 16,
    num_future_steps: int = 64,
    time_step: float = 0.1,
    num_frames: int = 4,
):
    """从本地加载数据"""
    
    if clip_id is None:
        # 自动选择一个 clip
        ego_files = glob.glob(f"{base_dir}/labels/egomotion/*.parquet")
        clip_id = Path(ego_files[0]).stem.split('.')[0]
    
    print(f"Loading local data for clip_id: {clip_id}")
    
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
    
    # 插值 ego motion
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
    ego_future_rot_local = (t0_rot_inv * spt.Rotation.from_quat(ego_future_quat)).as_matrix()
    
    # 加载摄像头数据
    cameras = [
        "camera_cross_left_120fov",
        "camera_front_wide_120fov", 
        "camera_cross_right_120fov",
        "camera_front_tele_30fov",
    ]
    
    camera_indices_map = {
        "camera_cross_left_120fov": 0,
        "camera_front_wide_120fov": 1,
        "camera_cross_right_120fov": 2,
        "camera_front_tele_30fov": 6,
    }
    
    image_frames_list = []
    camera_indices_list = []
    
    for cam_name in cameras:
        video_path = f"{base_dir}/camera/{cam_name}/{clip_id}.{cam_name}.mp4"
        if os.path.exists(video_path):
            frames = load_video_frames(video_path, num_frames=num_frames, t0_us=t0_us)
            image_frames_list.append(frames)
            camera_indices_list.append(camera_indices_map[cam_name])
        else:
            print(f"Warning: {video_path} not found")
    
    if not image_frames_list:
        raise FileNotFoundError("No camera data found!")
    
    # 堆叠数据
    image_frames = torch.stack(image_frames_list, dim=0)  # (N_cameras, T, C, H, W)
    camera_indices = torch.tensor(camera_indices_list, dtype=torch.int64)
    
    # 排序
    sort_order = torch.argsort(camera_indices)
    image_frames = image_frames[sort_order]
    camera_indices = camera_indices[sort_order]
    
    # 转换为 batch 格式
    ego_history_xyz_tensor = (
        torch.from_numpy(ego_history_xyz_local).float().unsqueeze(0).unsqueeze(0)
    )
    ego_history_rot_tensor = (
        torch.from_numpy(ego_history_rot_local).float().unsqueeze(0).unsqueeze(0)
    )
    ego_future_xyz_tensor = (
        torch.from_numpy(ego_future_xyz_local).float().unsqueeze(0).unsqueeze(0)
    )
    ego_future_rot_tensor = (
        torch.from_numpy(ego_future_rot_local).float().unsqueeze(0).unsqueeze(0)
    )
    
    return {
        "image_frames": image_frames,
        "camera_indices": camera_indices,
        "ego_history_xyz": ego_history_xyz_tensor,
        "ego_history_rot": ego_history_rot_tensor,
        "ego_future_xyz": ego_future_xyz_tensor,
        "ego_future_rot": ego_future_rot_tensor,
        "clip_id": clip_id,
    }


def run_inference(model, processor, data, helper):
    """运行单次推理"""
    # 准备输入
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
    
    # 运行推理
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs,
                top_p=0.98,
                temperature=0.6,
                num_traj_samples=1,
                max_generation_length=256,
                return_extra=True,
            )
    
    # 计算 minADE
    gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
    pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
    diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
    min_ade = diff.min()
    
    coc_text = extra["cot"][0][0][0] if "cot" in extra else "N/A"
    
    return min_ade, coc_text


def main():
    import glob
    from pathlib import Path
    
    print("=" * 60)
    print("Alpamayo R1 批量本地推理脚本")
    print("=" * 60)
    
    # 加载模型
    print("\n加载模型...")
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from alpamayo_r1 import helper
    
    model = AlpamayoR1.from_pretrained(
        "nvidia/Alpamayo-R1-10B",
        dtype=torch.bfloat16,
        local_files_only=True,
    ).to("cuda")
    
    processor = helper.get_processor(model.tokenizer)
    print("模型加载完成!")
    
    # 获取所有可用的 clip_id
    ego_files = glob.glob("/data01/vla/data_sample_chunk0/labels/egomotion/*.parquet")
    # 提取 clip_id (只取前 20 个)
    clip_ids = [Path(f).stem.split('.')[0] for f in ego_files[:20]]
    
    print(f"\n准备测试 {len(clip_ids)} 个样本...")
    print(f"样本列表: {clip_ids[:5]}... (前5个)")
    
    # 批量推理
    results = []
    failed_clips = []
    
    for i, clip_id in enumerate(clip_ids, 1):
        print(f"\n[{i}/{len(clip_ids)}] 处理 {clip_id}...", end=" ")
        
        try:
            # 加载数据
            data = load_local_dataset(
                base_dir="/data01/vla/data_sample_chunk0",
                clip_id=clip_id,
                t0_us=5_100_000,
            )
            
            # 运行推理
            min_ade, coc_text = run_inference(model, processor, data, helper)
            
            results.append({
                "clip_id": clip_id,
                "minADE": min_ade,
                "CoC": coc_text[:100] + "..." if len(coc_text) > 100 else coc_text,
            })
            
            print(f"✅ minADE={min_ade:.2f}m")
            
        except Exception as e:
            print(f"❌ 失败: {str(e)[:50]}")
            failed_clips.append((clip_id, str(e)))
        
        # 每 5 个样本清理一次显存
        if i % 5 == 0:
            torch.cuda.empty_cache()
    
    # 输出汇总结果
    print("\n" + "=" * 60)
    print("批量推理结果汇总")
    print("=" * 60)
    
    if results:
        minADEs = [r["minADE"] for r in results]
        print(f"\n成功样本: {len(results)}/{len(clip_ids)}")
        print(f"平均 minADE: {np.mean(minADEs):.4f} 米")
        print(f"minADE 中位数: {np.median(minADEs):.4f} 米")
        print(f"minADE 标准差: {np.std(minADEs):.4f} 米")
        print(f"minADE 范围: [{np.min(minADEs):.4f}, {np.max(minADEs):.4f}] 米")
        
        print("\n各样本结果:")
        print("-" * 60)
        for r in results:
            print(f"{r['clip_id'][:20]:20s} minADE: {r['minADE']:.4f}m")
        
        print("\n第一个样本的 Chain-of-Causation:")
        print("-" * 60)
        print(results[0]["CoC"])
    
    if failed_clips:
        print(f"\n失败样本 ({len(failed_clips)} 个):")
        print("-" * 60)
        for clip_id, error in failed_clips:
            print(f"  {clip_id}: {error[:60]}...")
    
    print("\n" + "=" * 60)
    print("批量推理完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
