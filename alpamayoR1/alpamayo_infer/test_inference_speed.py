#!/usr/bin/env python3
"""测试推理速度"""
import os
import sys
import time
import copy

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
import numpy as np
import pandas as pd
from PIL import Image
from einops import rearrange

sys.path.insert(0, '/home/user/mikelee/alpamayo-main/src')
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper
from scipy.spatial.transform import Rotation as R

MODEL_PATH = "/data01/vla/models--nvidia--Alpamayo-R1-10B/snapshots/22fab1399111f50b52bfbe5d8b809f39bd4c2fe1"
CAMERA_ORDER = ['camera_cross_left_120fov','camera_front_wide_120fov','camera_cross_right_120fov','camera_front_tele_30fov']

def load_data(row, data_dir):
    """加载单帧数据"""
    # 加载图像
    images = []
    for cam in CAMERA_ORDER:
        for t in range(4):
            frame_idx = int(row[f'{cam}_f{t}_idx'])
            img = Image.open(f"{data_dir}/camera_images/{cam}/{frame_idx:06d}.jpg").convert('RGB')
            images.append(np.array(img))
    images = rearrange(np.stack(images), '(c t) h w ch -> c t ch h w', c=4, t=4)
    
    # 加载egomotion
    frame_id = int(row['frame_id'])
    history = np.load(f"{data_dir}/egomotion/frame_{frame_id:06d}_history.npy", allow_pickle=False)
    future = np.load(f"{data_dir}/egomotion/frame_{frame_id:06d}_future_gt.npy", allow_pickle=False)
    
    hist_xyz_world, hist_quat = history[:,5:8], history[:,1:5]
    t0_rot_inv = R.from_quat(hist_quat[-1]).inv()
    hist_xyz = t0_rot_inv.apply(hist_xyz_world - hist_xyz_world[-1])
    hist_rot = (t0_rot_inv * R.from_quat(hist_quat)).as_matrix()
    
    image_frames = torch.from_numpy(images).float()
    hist_xyz_t = torch.from_numpy(hist_xyz).float().unsqueeze(0).unsqueeze(0)
    hist_rot_t = torch.from_numpy(hist_rot).float().unsqueeze(0).unsqueeze(0)
    
    return image_frames, hist_xyz_t, hist_rot_t

def main():
    CLIP_ID = "01d3588e-bca7-4a18-8e74-c6cfe9e996db"
    DATA_DIR = f'/data01/vla/data/data_sample_chunk0/infer/{CLIP_ID}/data'
    
    print("="*70)
    print("推理速度测试")
    print("="*70)
    
    # 加载数据
    print("\n1. 加载数据...")
    index_df = pd.read_csv(f'{DATA_DIR}/inference_index_strict.csv')
    row = index_df.iloc[0]
    image_frames, hist_xyz, hist_rot = load_data(row, DATA_DIR)
    print(f"   图像 shape: {image_frames.shape}")
    print(f"   历史轨迹 shape: {hist_xyz.shape}")
    
    # 加载模型
    print("\n2. 加载模型...")
    start = time.time()
    print("   2.1 开始 from_pretrained...")
    model = AlpamayoR1.from_pretrained(MODEL_PATH, dtype=torch.bfloat16)
    print("   2.2 from_pretrained 完成，移动到 cuda...")
    model = model.to("cuda")
    print("   2.3 获取 processor...")
    processor = helper.get_processor(model.tokenizer)
    print(f"   模型加载时间: {time.time()-start:.1f}s")
    print(f"   模型 dtype: {model.dtype}")
    print(f"   注意力实现: {model.vlm.config._attn_implementation}")
    
    # 准备输入
    print("\n3. 准备输入...")
    messages = helper.create_message(image_frames.flatten(0, 1))
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        continue_final_message=True, return_dict=True, return_tensors="pt",
    )
    
    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": hist_xyz,
        "ego_history_rot": hist_rot,
    }
    model_inputs = helper.to_device(model_inputs, "cuda")
    torch.cuda.manual_seed_all(42)
    
    # 预热
    print("\n4. 预热 (3次)...")
    for i in range(3):
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _ = model.sample_trajectories_from_data_with_vlm_rollout(
                    data=copy.deepcopy(model_inputs),
                    top_p=0.98, temperature=0.6,
                    num_traj_samples=1,
                    max_generation_length=256,
                    return_extra=True
                )
        torch.cuda.synchronize()
        print(f"   预热 {i+1}/3 完成")
    
    # 正式测试
    print("\n5. 正式测试 (5次)...")
    times = []
    for i in range(5):
        torch.cuda.synchronize()
        start = time.time()
        
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                    data=copy.deepcopy(model_inputs),
                    top_p=0.98, temperature=0.6,
                    num_traj_samples=1,
                    max_generation_length=256,
                    return_extra=True
                )
        
        torch.cuda.synchronize()
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"   测试 {i+1}/5: {elapsed*1000:.1f} ms")
    
    print(f"\n6. 结果统计:")
    print(f"   平均: {np.mean(times)*1000:.1f} ms")
    print(f"   中位数: {np.median(times)*1000:.1f} ms")
    print(f"   最小: {np.min(times)*1000:.1f} ms")
    print(f"   最大: {np.max(times)*1000:.1f} ms")

if __name__ == '__main__':
    main()
