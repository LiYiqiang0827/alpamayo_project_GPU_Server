#!/usr/bin/env python3
"""
改进的方法 - 对齐官方实现的关键细节
"""
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import pandas as pd
import numpy as np
import torch
import av
import scipy.spatial.transform as spt
from einops import rearrange
import warnings
warnings.filterwarnings('ignore')

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper

def rotate_90cc(xy):
    return np.stack([-xy[1], xy[0]], axis=0)

def load_egomotion(ego_path: str):
    df = pd.read_parquet(ego_path)
    return df[['timestamp', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']].values

def extract_frames_at_timestamps(video_path: str, ts_path: str, target_timestamps: np.ndarray):
    """从视频中提取指定时间戳的帧 - 严格对齐官方方法"""
    df_ts = pd.read_parquet(ts_path)
    video_ts = df_ts['timestamp'].values
    
    container = av.open(video_path)
    frames_list = []
    
    for target_ts in target_timestamps:
        # 找到最接近的帧索引
        frame_idx = np.argmin(np.abs(video_ts - target_ts))
        
        # 解码到该帧
        container.seek(0)
        frame_count = 0
        frame_img = None
        for frame in container.decode(video=0):
            if frame_count == frame_idx:
                frame_img = frame.to_ndarray(format='rgb24')
                break
            frame_count += 1
        
        if frame_img is not None:
            frames_list.append(frame_img)
        else:
            # 如果失败，用零填充
            frames_list.append(np.zeros((224, 224, 3), dtype=np.uint8))
    
    container.close()
    
    frames = np.stack(frames_list)
    frames = rearrange(frames, "t h w c -> t c h w")
    return torch.from_numpy(frames).float()

def improved_inference(base_dir, clip_id, t0_us, model, processor):
    """改进的方法 - 对齐官方实现"""
    
    # 1. 计算图像时间戳 - 官方: [t0-0.3s, t0-0.2s, t0-0.1s, t0]
    num_frames = 4
    time_step = 0.1
    image_timestamps = np.array(
        [t0_us - (num_frames - 1 - i) * int(time_step * 1_000_000) for i in range(num_frames)],
        dtype=np.int64,
    )
    print(f"  图像时间戳: {image_timestamps}")
    print(f"  相对t0: {(image_timestamps - t0_us)/1e6}")
    
    # 2. 计算ego_history时间戳 - 官方从 t0-1.5s 到 t0
    num_history_steps = 16
    history_offsets_us = np.arange(
        -(num_history_steps - 1) * time_step * 1_000_000,
        time_step * 1_000_000 / 2,
        time_step * 1_000_000,
    ).astype(np.int64)
    history_timestamps = t0_us + history_offsets_us
    print(f"  历史轨迹时间戳范围: {(history_timestamps[0] - t0_us)/1e6:.2f}s ~ {(history_timestamps[-1] - t0_us)/1e6:.2f}s")
    
    # 3. 计算ego_future时间戳 - t0+0.1s 到 t0+6.4s
    num_future_steps = 64
    future_offsets_us = np.arange(
        time_step * 1_000_000,
        (num_future_steps + 0.5) * time_step * 1_000_000,
        time_step * 1_000_000,
    ).astype(np.int64)
    future_timestamps = t0_us + future_offsets_us
    
    # 4. 加载ego motion数据
    ego_path = f"{base_dir}/labels/egomotion/{clip_id}.egomotion.parquet"
    ego_data = load_egomotion(ego_path)
    
    # 5. 插值获取历史和未来的ego pose
    def interpolate_egomotion(ego_data, target_timestamps):
        xyz_list = []
        quat_list = []
        for t in target_timestamps:
            idx = np.argmin(np.abs(ego_data[:, 0] - t))
            xyz_list.append(ego_data[idx, 1:4])
            quat_list.append(ego_data[idx, 4:8])
        return np.array(xyz_list), np.array(quat_list)
    
    ego_history_xyz, ego_history_quat = interpolate_egomotion(ego_data, history_timestamps)
    ego_future_xyz, ego_future_quat = interpolate_egomotion(ego_data, future_timestamps)
    
    # 6. 转换到本地坐标系（以t0为原点）
    t0_idx = np.argmin(np.abs(ego_data[:, 0] - t0_us))
    t0_xyz = ego_data[t0_idx, 1:4]
    t0_quat = ego_data[t0_idx, 4:8]
    t0_rot = spt.Rotation.from_quat(t0_quat)
    t0_rot_inv = t0_rot.inv()
    
    # 转换位置
    ego_history_xyz_local = t0_rot_inv.apply(ego_history_xyz - t0_xyz)
    ego_future_xyz_local = t0_rot_inv.apply(ego_future_xyz - t0_xyz)
    
    # 转换旋转 - 关键改进！官方使用完整的旋转矩阵
    ego_history_rot_local = (t0_rot_inv * spt.Rotation.from_quat(ego_history_quat)).as_matrix()
    ego_future_rot_local = (t0_rot_inv * spt.Rotation.from_quat(ego_future_quat)).as_matrix()
    
    print(f"  t0位置: {t0_xyz}")
    print(f"  历史轨迹本地坐标范围: X={ego_history_xyz_local[:,0].min():.2f}~{ego_history_xyz_local[:,0].max():.2f}, "
          f"Y={ego_history_xyz_local[:,1].min():.2f}~{ego_history_xyz_local[:,1].max():.2f}")
    
    # 7. 加载4摄像头图像 - 使用严格对齐的时间戳
    camera_features = [
        ("camera_cross_left_120fov", 0),
        ("camera_front_wide_120fov", 1),
        ("camera_cross_right_120fov", 2),
        ("camera_front_tele_30fov", 6),
    ]
    
    image_frames_list = []
    camera_indices_list = []
    
    for cam_name, cam_idx in camera_features:
        video_path = f"{base_dir}/camera/{cam_name}/{clip_id}.{cam_name}.mp4"
        ts_path = video_path.replace('.mp4', '.timestamps.parquet')
        
        if os.path.exists(video_path) and os.path.exists(ts_path):
            frames = extract_frames_at_timestamps(video_path, ts_path, image_timestamps)
            image_frames_list.append(frames)
            camera_indices_list.append(cam_idx)
    
    # 按camera_idx排序 [0, 1, 2, 6]
    sort_order = np.argsort(camera_indices_list)
    image_frames = torch.stack([image_frames_list[i] for i in sort_order], dim=0)
    
    print(f"  图像张量形状: {image_frames.shape}")
    
    # 8. 准备模型输入
    frames = image_frames.flatten(0, 1)
    messages = helper.create_message(frames)
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        continue_final_message=True, return_dict=True, return_tensors="pt",
    )
    
    # 使用完整的旋转矩阵（关键改进！）
    ego_history_xyz_tensor = torch.from_numpy(ego_history_xyz_local).float().unsqueeze(0).unsqueeze(0)
    ego_history_rot_tensor = torch.from_numpy(ego_history_rot_local).float().unsqueeze(0).unsqueeze(0)
    
    model_inputs = {
        "tokenized_data": {k: v.to("cuda") for k, v in inputs.items()},
        "ego_history_xyz": ego_history_xyz_tensor.to("cuda"),
        "ego_history_rot": ego_history_rot_tensor.to("cuda"),
    }
    
    # 9. 运行推理
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs, top_p=0.98, temperature=0.6,
                num_traj_samples=6, max_generation_length=256, return_extra=True,
            )
    
    # 10. 计算minADE
    pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2]
    gt_xy_local = ego_future_xyz_local[:, :2]
    
    diff = np.linalg.norm(pred_xy - gt_xy_local[None, ...], axis=2).mean(-1)
    min_ade = diff.min()
    coc = extra["cot"][0][0][0] if "cot" in extra else "N/A"
    
    return min_ade, coc, pred_xy, gt_xy_local

def old_inference(base_dir, clip_id, t0_us, model, processor):
    """我们的旧方法 - 用于对比"""
    
    ego_path = f"{base_dir}/labels/egomotion/{clip_id}.egomotion.parquet"
    ego_data = load_egomotion(ego_path)
    
    # 旧方法的时间戳计算
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
    
    # 旧方法的图像加载 - 均匀采样
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
    
    image_frames_list = []
    camera_order = [
        ("camera_cross_left_120fov", 0),
        ("camera_front_wide_120fov", 1),
        ("camera_cross_right_120fov", 2),
        ("camera_front_tele_30fov", 6),
    ]
    for cam_name, cam_idx in camera_order:
        video_path = f"{base_dir}/camera/{cam_name}/{clip_id}.{cam_name}.mp4"
        if os.path.exists(video_path):
            frames = load_video_frames(video_path, num_frames=4)
            image_frames_list.append(frames)
    
    image_frames = torch.stack(image_frames_list, dim=0)
    
    # 旧方法的坐标转换
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
    
    # 旧方法：使用单位矩阵而不是真实的旋转矩阵！
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
    
    pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2]
    gt_xy_local = np.array(future_local)[:, :2]
    
    diff = np.linalg.norm(pred_xy - gt_xy_local[None, ...], axis=2).mean(-1)
    min_ade = diff.min()
    coc = extra["cot"][0][0][0] if "cot" in extra else "N/A"
    
    return min_ade, coc, pred_xy, gt_xy_local

def main():
    base_dir = "/data01/vla/data_sample_chunk0"
    clip_id = "fa83bcb8-ea31-4dbb-b447-4fb458f5984b"
    t0_us = 7_800_000
    
    print("=" * 70)
    print("对比测试: 旧方法 vs 改进方法")
    print("=" * 70)
    print(f"Clip ID: {clip_id}")
    print(f"t0: {t0_us/1e6:.2f}s")
    print()
    
    # 加载模型
    print("加载模型...")
    model = AlpamayoR1.from_pretrained(
        "nvidia/Alpamayo-R1-10B",
        dtype=torch.bfloat16,
        local_files_only=True,
    ).to("cuda")
    processor = helper.get_processor(model.tokenizer)
    print("模型加载完成!\n")
    
    # 方法1: 旧方法
    print("=" * 70)
    print("方法1: 旧方法 (均匀采样图像 + 单位旋转矩阵)")
    print("=" * 70)
    min_ade_1, coc_1, pred_1, gt_1 = old_inference(base_dir, clip_id, t0_us, model, processor)
    print(f"\n✅ minADE: {min_ade_1:.4f}m")
    print(f"📝 CoC: {coc_1[:80]}...")
    print()
    
    # 方法2: 改进方法
    print("=" * 70)
    print("方法2: 改进方法 (严格时间戳对齐 + 真实旋转矩阵)")
    print("=" * 70)
    min_ade_2, coc_2, pred_2, gt_2 = improved_inference(base_dir, clip_id, t0_us, model, processor)
    print(f"\n✅ minADE: {min_ade_2:.4f}m")
    print(f"📝 CoC: {coc_2[:80]}...")
    print()
    
    # 对比
    print("=" * 70)
    print("对比结果")
    print("=" * 70)
    print(f"旧方法:   {min_ade_1:.4f}m")
    print(f"改进方法: {min_ade_2:.4f}m")
    print(f"差异:     {abs(min_ade_1 - min_ade_2):.4f}m ({abs(min_ade_1 - min_ade_2)/min_ade_1*100:.1f}%)")
    
    if min_ade_2 < min_ade_1:
        print(f"\n🎉 改进方法更好！提升: {((min_ade_1 - min_ade_2)/min_ade_1*100):.1f}%")
    elif min_ade_1 < min_ade_2:
        print(f"\n⚠️  旧方法更好")
    else:
        print(f"\n⚡ 两种方法结果相近")
    
    print("\n" + "=" * 70)
    print("关键改进点:")
    print("1. 图像时间戳严格对齐 [t0-0.3s, t0-0.2s, t0-0.1s, t0]")
    print("2. 使用真实的ego_history_rot矩阵（不是单位矩阵）")
    print("3. 摄像头按索引排序 [0, 1, 2, 6]")
    print("=" * 70)

if __name__ == "__main__":
    main()
