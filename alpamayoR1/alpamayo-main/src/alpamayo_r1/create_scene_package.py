#!/usr/bin/env python3
"""
创建完整的场景分析数据包 - 包含图像、真值、推理结果
"""
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import pandas as pd
import numpy as np
import torch
import av
import json
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
    print(f"创建完整分析数据包")
    print("=" * 70)
    print(f"Clip ID: {clip_id}")
    print(f"时间点: t0 = {t0_us/1e6:.1f}s")
    print()
    
    # 创建输出目录
    output_dir = f"/tmp/scene_analysis_{clip_id[:8]}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    
    # 1. 提取4摄像头图像
    print("1. 提取4摄像头图像...")
    cameras = [
        ("camera_cross_left_120fov", "01_left_cross"),
        ("camera_front_wide_120fov", "02_front_wide"),
        ("camera_cross_right_120fov", "03_right_cross"),
        ("camera_front_tele_30fov", "04_front_tele"),
    ]
    
    for cam_name, label in cameras:
        video_path = f"{base_dir}/camera/{cam_name}/{clip_id}.{cam_name}.mp4"
        ts_cam_path = video_path.replace('.mp4', '.timestamps.parquet')
        
        if os.path.exists(video_path) and os.path.exists(ts_cam_path):
            df_cam = pd.read_parquet(ts_cam_path)
            ts_cam = df_cam['timestamp'].values
            frame_idx = np.argmin(np.abs(ts_cam - t0_us))
            
            frame = extract_frame_single(video_path, frame_idx)
            if frame is not None:
                img = Image.fromarray(frame)
                img.save(f"{output_dir}/images/{label}.png")
                print(f"  ✅ {label}.png")
    
    # 2. 加载并保存ego motion真值
    print("\n2. 加载ego motion真值...")
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
    
    # 获取预测轨迹（最佳的一条）
    pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2]
    gt_xy_local = np.array(future_local)[:, :2]
    
    diff = np.linalg.norm(pred_xy - gt_xy_local[None, ...], axis=2).mean(axis=1)
    best_idx = diff.argmin()
    best_pred = pred_xy[best_idx].tolist()
    
    coc_text = extra["cot"][0][0][0] if "cot" in extra else "N/A"
    min_ade = diff.min()
    
    print(f"  ✅ minADE: {min_ade:.2f}m")
    print(f"  📝 CoC: {coc_text[:60]}...")
    
    # 4. 保存所有数据到JSON
    print("\n4. 保存数据到JSON...")
    analysis_data = {
        "clip_id": clip_id,
        "t0_us": int(t0_us),
        "t0_seconds": t0_us / 1e6,
        "minADE": float(min_ade),
        "CoC": coc_text,
        "camera_order": ["left_cross", "front_wide", "right_cross", "front_tele"],
        "history_trajectory": {
            "time_steps": 16,
            "duration_seconds": 1.6,
            "xyz_local": hist_local,
        },
        "future_ground_truth": {
            "time_steps": 64,
            "duration_seconds": 6.4,
            "xyz_local": future_local,
        },
        "predicted_trajectory": {
            "best_sample_index": int(best_idx),
            "xy_local": best_pred,
        },
    }
    
    with open(f"{output_dir}/analysis_data.json", "w") as f:
        json.dump(analysis_data, f, indent=2)
    
    # 5. 创建说明文档
    readme = f"""# 场景分析数据包

## 基本信息
- **Clip ID**: {clip_id}
- **时间点**: t0 = {t0_us/1e6:.1f}s (视频内)
- **视频时间范围**: {video_start/1e6:.2f}s ~ {video_end/1e6:.2f}s

## 推理结果
- **minADE**: {min_ade:.2f}m
- **CoC (Chain-of-Causation)**: 
  > {coc_text}

## 文件夹结构
```
scene_analysis_{clip_id[:8]}/
├── images/
│   ├── 01_left_cross.png    # 左交叉摄像头
│   ├── 02_front_wide.png    # 前广角摄像头
│   ├── 03_right_cross.png   # 右交叉摄像头
│   └── 04_front_tele.png    # 前长焦摄像头
├── analysis_data.json       # 完整数据（轨迹、真值、预测）
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

### predicted_trajectory (预测轨迹)
- 样本数: 6条
- 选择: minADE最小的那条
- 维度: XY平面 (Z忽略)

## 生成时间
{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    with open(f"{output_dir}/README.md", "w") as f:
        f.write(readme)
    
    print(f"\n" + "=" * 70)
    print(f"数据包创建完成!")
    print(f"位置: {output_dir}")
    print(f"内容:")
    print(f"  - images/: 4个摄像头图像")
    print(f"  - analysis_data.json: 轨迹数据和推理结果")
    print(f"  - README.md: 说明文档")
    print("=" * 70)

if __name__ == "__main__":
    main()
