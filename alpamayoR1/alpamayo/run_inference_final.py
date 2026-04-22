#!/usr/bin/env python3
"""
推理脚本 - 使用新的严格预处理索引
简化版：使用之前成功的 local 推理方式
"""
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import sys
sys.path.insert(0, '/home/user/mikelee/alpamayo-main/src')

import numpy as np
import pandas as pd
import torch
from pathlib import Path

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper
import alpamayo_r1.load_local_dataset as load_local

# 配置
clip_id = '01d3588e-bca7-4a18-8e74-c6cfe9e996db'
num_frames = 100

infer_dir = f'/data01/vla/data/data_sample_chunk0/infer/{clip_id}'

# 加载索引
index_path = f'{infer_dir}/data/inference_index_strict.csv'
index_df = pd.read_csv(index_path)

print(f"Loaded index: {len(index_df)} frames")
print(f"Will process: {min(num_frames, len(index_df))} frames\n")

# 加载模型
print("Loading model...")
model = AlpamayoR1.from_pretrained(
    "nvidia/Alpamayo-R1-10B",
    dtype=torch.bfloat16,
    local_files_only=True,
).to("cuda")
processor = helper.get_processor(model.tokenizer)
print("Model loaded!\n")

results = []
num_to_process = min(num_frames, len(index_df))

for i in range(num_to_process):
    row = index_df.iloc[i]
    frame_id = int(row['frame_id'])
    ego_idx = int(row['ego_idx'])
    
    print(f"[{i+1}/{num_to_process}] Frame {frame_id} (ego_idx={ego_idx})", end=" ")
    
    try:
        # 使用 load_local 加载数据
        data = load_local.load_local_physical_aiavdataset(
            clip_id=clip_id,
            t0_us=int(row['ego_ts']),
            chunk_id=0,
            num_history_steps=16,
            num_future_steps=64,
            num_frames=4,
        )
        
        # 推理
        with torch.no_grad():
            pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                data=data,
                top_p=0.98,
                temperature=0.6,
                num_traj_samples=1,
            )
        
        # 计算 ADE
        gt_xy = data['ego_future_xyz'][0, 0, :, :2].cpu().numpy()
        pred_xy = pred_xyz[0, 0, :, :2].cpu().numpy()
        diff = np.linalg.norm(pred_xy - gt_xy, axis=1)
        ade = diff.mean()
        print(f"ADE: {ade:.3f}m")
        results.append({'frame_id': frame_id, 'ego_idx': ego_idx, 'ade': ade})
        
    except Exception as e:
        print(f"Error: {e}")
        continue

# 总结
if results:
    ades = [r['ade'] for r in results]
    print(f"\n{'='*50}")
    print(f"Processed {len(results)} frames")
    print(f"Mean ADE: {np.mean(ades):.3f}m")
    print(f"Median ADE: {np.median(ades):.3f}m")
    print(f"Min/Max: {np.min(ades):.3f}m / {np.max(ades):.3f}m")
    
    df = pd.DataFrame(results)
    df.to_csv(f'{infer_dir}/inference_results_strict.csv', index=False)
    print(f"\nSaved: {infer_dir}/inference_results_strict.csv")
