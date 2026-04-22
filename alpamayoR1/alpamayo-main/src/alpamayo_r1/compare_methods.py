#!/usr/bin/env python3
"""
对比我们的方法和官方load_physical_aiavdataset的差异
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
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset

def rotate_90cc(xy):
    return np.stack([-xy[1], xy[0]], axis=0)

def load_egomotion(ego_path: str):
    df = pd.read_parquet(ego_path)
    return df[['timestamp', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']].values

def our_method_inference(base_dir, clip_id, t0_us, model, processor):
    """我们的原始方法"""
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
    
    # 加载4摄像头视频帧
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
    
    pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2]
    gt_xy_local = np.array(future_local)[:, :2]
    
    diff = np.linalg.norm(pred_xy - gt_xy_local[None, ...], axis=2).mean(-1)
    min_ade = diff.min()
    coc = extra["cot"][0][0][0] if "cot" in extra else "N/A"
    
    return min_ade, coc, pred_xy, gt_xy_local

def official_method_inference(clip_id, t0_us, model, processor):
    """使用官方load_physical_aiavdataset方法"""
    import physical_ai_av
    
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface(revision="2ae73f49ffd2b5db43b404201beb7b92889f7afc")
    
    data = load_physical_aiavdataset(
        clip_id=clip_id,
        t0_us=t0_us,
        avdi=avdi,
        maybe_stream=False,  # 使用本地数据
    )
    
    messages = helper.create_message(data["image_frames"].flatten(0, 1))
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        continue_final_message=True, return_dict=True, return_tensors="pt",
    )
    
    model_inputs = {
        "tokenized_data": {k: v.to("cuda") for k, v in inputs.items()},
        "ego_history_xyz": data["ego_history_xyz"].to("cuda"),
        "ego_history_rot": data["ego_history_rot"].to("cuda"),
    }
    
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs, top_p=0.98, temperature=0.6,
                num_traj_samples=6, max_generation_length=256, return_extra=True,
            )
    
    pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2]
    gt_xy = data["ego_future_xyz"].cpu().numpy()[0, 0, :, :2]
    
    diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=2).mean(-1)
    min_ade = diff.min()
    coc = extra["cot"][0][0][0] if "cot" in extra else "N/A"
    
    return min_ade, coc, pred_xy, gt_xy

def main():
    base_dir = "/data01/vla/data_sample_chunk0"
    clip_id = "fa83bcb8-ea31-4dbb-b447-4fb458f5984b"
    t0_us = 7_800_000
    
    print("=" * 70)
    print("对比测试: 我们的方法 vs 官方方法")
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
    
    # 方法1: 我们的方法
    print("-" * 70)
    print("方法1: 我们的原始方法")
    print("-" * 70)
    min_ade_1, coc_1, pred_1, gt_1 = our_method_inference(base_dir, clip_id, t0_us, model, processor)
    print(f"minADE: {min_ade_1:.4f}m")
    print(f"CoC: {coc_1[:80]}...")
    print()
    
    # 方法2: 官方方法
    print("-" * 70)
    print("方法2: 官方load_physical_aiavdataset方法")
    print("-" * 70)
    try:
        min_ade_2, coc_2, pred_2, gt_2 = official_method_inference(clip_id, t0_us, model, processor)
        print(f"minADE: {min_ade_2:.4f}m")
        print(f"CoC: {coc_2[:80]}...")
        print()
        
        # 对比
        print("=" * 70)
        print("对比结果")
        print("=" * 70)
        print(f"我们的方法:   {min_ade_1:.4f}m")
        print(f"官方方法:     {min_ade_2:.4f}m")
        print(f"差异:         {abs(min_ade_1 - min_ade_2):.4f}m")
        
        if min_ade_2 < min_ade_1:
            print(f"\n✅ 官方方法更好，改进: {((min_ade_1 - min_ade_2)/min_ade_1*100):.1f}%")
        elif min_ade_1 < min_ade_2:
            print(f"\n✅ 我们的方法更好")
        else:
            print(f"\n⚡ 两种方法结果相近")
            
    except Exception as e:
        print(f"官方方法失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
