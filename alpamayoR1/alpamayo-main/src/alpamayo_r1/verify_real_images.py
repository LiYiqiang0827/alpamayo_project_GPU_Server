#!/usr/bin/env python3
"""
真实图像 + num_traj_samples=6 验证脚本
"""
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import glob
import numpy as np
import pandas as pd
import torch
import av
from pathlib import Path
import scipy.spatial.transform as spt
from einops import rearrange

def load_video_frames(video_path: str, num_frames: int = 4):
    """从本地视频文件加载帧"""
    container = av.open(video_path)
    stream = container.streams.video[0]
    total_frames = stream.frames if stream.frames else 100
    
    # 均匀采样帧
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for i, frame in enumerate(container.decode(video=0)):
        if i in indices:
            img = frame.to_ndarray(format='rgb24')
            frames.append(img)
        if len(frames) >= num_frames:
            break
    
    container.close()
    
    # 确保有足够帧
    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
    
    # 转换为 tensor (T, C, H, W)
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

def load_local_data_with_images(base_dir: str, clip_id: str, t0_us: int = 5_100_000):
    """加载本地数据（含真实图像）"""
    
    # 加载 ego motion
    ego_path = f"{base_dir}/labels/egomotion/{clip_id}.egomotion.parquet"
    translations, rotations, timestamps = load_egomotion(ego_path)
    
    # 计算轨迹时间戳
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
        raise FileNotFoundError("No camera data found!")
    
    # 堆叠数据
    image_frames = torch.stack(image_frames_list, dim=0)  # (N_cameras, T, C, H, W)
    camera_indices = torch.tensor(camera_indices_list, dtype=torch.int64)
    
    # 排序
    sort_order = torch.argsort(camera_indices)
    image_frames = image_frames[sort_order]
    camera_indices = camera_indices[sort_order]
    
    return {
        "image_frames": image_frames,
        "camera_indices": camera_indices,
        "ego_history_xyz": torch.from_numpy(ego_history_xyz_local).float().unsqueeze(0).unsqueeze(0),
        "ego_history_rot": torch.from_numpy(ego_history_rot_local).float().unsqueeze(0).unsqueeze(0),
        "ego_future_xyz": torch.from_numpy(ego_future_xyz_local).float().unsqueeze(0).unsqueeze(0),
        "clip_id": clip_id,
    }

def run_inference_real(model, processor, data, helper, num_traj_samples=6):
    """使用真实图像运行推理"""
    
    # 创建消息
    frames = data["image_frames"].flatten(0, 1)  # (N_cameras * T, C, H, W)
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
                num_traj_samples=num_traj_samples,  # 6条轨迹
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
    print("=" * 70)
    print("真实图像 + num_traj_samples=6 验证")
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
    
    # 选择表现差异大的样本进行验证
    test_samples = [
        ("43ada34d-283f-46b9-ae38-cd41c576b889", "最佳 (之前0.42m)"),
        ("fa83bcb8-ea31-4dbb-b447-4fb458f5984b", "最差 (之前38.48m)"),
        ("ef4264ed-0fd2-4a64-9831-87e1aae28407", "良好 (之前0.52m)"),
    ]
    
    print(f"\n验证 {len(test_samples)} 个样本...")
    print("对比: num_traj_samples=1 (之前) vs num_traj_samples=6 (现在)")
    print("输入: 真实图像 vs 虚拟图像\n")
    
    results = []
    for clip_id, desc in test_samples:
        print(f"【{desc}】")
        print(f"  Clip: {clip_id}")
        
        try:
            # 加载真实数据
            data = load_local_data_with_images(base_dir, clip_id)
            print(f"  图像: {data['image_frames'].shape}")
            
            # 运行推理 (num_traj_samples=6)
            min_ade, coc = run_inference_real(model, processor, data, helper, num_traj_samples=6)
            
            results.append({
                "clip_id": clip_id,
                "description": desc,
                "minADE_6samples": min_ade,
                "CoC": coc,
            })
            
            print(f"  ✅ minADE (6 samples): {min_ade:.2f}m")
            print(f"  📝 CoC: {coc[:80]}...")
            print()
            
        except Exception as e:
            print(f"  ❌ 失败: {str(e)[:60]}\n")
        
        # 清理显存
        torch.cuda.empty_cache()
    
    # 输出对比
    print("=" * 70)
    print("结果对比")
    print("=" * 70)
    print(f"{'样本':<30} {'之前(1样)':<12} {'现在(6样)':<12} {'提升':<10}")
    print("-" * 70)
    
    previous_results = {
        "43ada34d-283f-46b9-ae38-cd41c576b889": 0.42,
        "fa83bcb8-ea31-4dbb-b447-4fb458f5984b": 38.48,
        "ef4264ed-0fd2-4a64-9831-87e1aae28407": 0.52,
    }
    
    for r in results:
        prev = previous_results.get(r["clip_id"][:20], 0)
        now = r["minADE_6samples"]
        improvement = ((prev - now) / prev * 100) if prev > 0 else 0
        print(f"{r['clip_id'][:28]:<30} {prev:<12.2f} {now:<12.2f} {improvement:+.1f}%")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
