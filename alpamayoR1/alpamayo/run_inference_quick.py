#!/usr/bin/env python3
"""
快速推理脚本 - 使用新的严格预处理索引
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
from PIL import Image
import glob

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

def run_quick_inference(clip_id='01d3588e-bca7-4a18-8e74-c6cfe9e996db', num_frames=100):
    base_dir = '/data01/vla/data/data_sample_chunk0'
    infer_dir = f'{base_dir}/infer/{clip_id}'
    
    # Load index
    index_path = f'{infer_dir}/data/inference_index_strict.csv'
    index_df = pd.read_csv(index_path)
    
    print(f"Loaded index: {len(index_df)} frames available")
    print(f"Will process: {min(num_frames, len(index_df))} frames\n")
    
    # Load model
    print("Loading model...")
    model = AlpamayoR1.from_pretrained(
        "nvidia/Alpamayo-R1-10B",
        dtype=torch.bfloat16,
        local_files_only=True,
    ).to("cuda")
    
    # Get processor  
    from alpamayo_r1 import helper
    processor = helper.get_processor(model.tokenizer)
    print("Model loaded!\n")
    
    # Camera names
    cameras = ['camera_cross_left_120fov', 'camera_front_wide_120fov', 
               'camera_cross_right_120fov', 'camera_front_tele_30fov']
    
    results = []
    num_to_process = min(num_frames, len(index_df))
    
    for i in range(num_to_process):
        row = index_df.iloc[i]
        frame_id = int(row['frame_id'])
        
        print(f"[{i+1}/{num_to_process}] Frame {frame_id}", end=" ")
        
        # Load images (use f3 - current time)
        images = []
        for cam in cameras:
            idx_col = f'{cam}_f3_idx'
            frame_idx = int(row[idx_col])
            img_path = f'{infer_dir}/data/camera_images/{cam}/{frame_idx:06d}.jpg'
            img = Image.open(img_path)
            images.append(img)
        
        # Load egomotion history
        hist_path = f'{infer_dir}/data/egomotion/frame_{frame_id:06d}_history.npy'
        history = np.load(hist_path)
        
        # Extract xyz and quaternions
        ego_xyz = history[:, 4:7]  # x, y, z
        ego_quat = history[:, 0:4]  # qx, qy, qz, qw
        
        # Convert quaternions to rotation matrices
        from scipy.spatial.transform import Rotation as R
        ego_rot = R.from_quat(ego_quat).as_matrix()  # (16, 3, 3)
        
        # Convert to tensors (add trajectory group dimension)
        ego_xyz_tensor = torch.from_numpy(ego_xyz).float().unsqueeze(0).unsqueeze(0).to("cuda")  # (1, 1, 16, 3)
        ego_rot_tensor = torch.from_numpy(ego_rot).float().unsqueeze(0).unsqueeze(0).to("cuda")  # (1, 1, 16, 3, 3)
        
        # Load ground truth
        future_path = f'{infer_dir}/data/egomotion/frame_{frame_id:06d}_future_gt.npy'
        future = np.load(future_path)
        gt_xy = future[:, 1:3]  # x, y
        
        # Inference
        text = "<image>" * 4 + " Predict the future trajectory."
        inputs = processor(text=[text], images=images, return_tensors="pt")
        
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to("cuda")
        
        # Prepare model inputs
        model_inputs = {
            **inputs,
            'ego_history_xyz': ego_xyz_tensor,
            'ego_history_rot': ego_rot_tensor,
        }
        
        with torch.no_grad():
            pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs,
                top_p=0.98,
                temperature=0.6,
                num_traj_samples=1,
            )
        
        # Calculate ADE
        pred_xy = pred_xyz[0, 0, :, :2].cpu().numpy()
        diff = np.linalg.norm(pred_xy - gt_xy, axis=1)
        ade = diff.mean()
        print(f"ADE: {ade:.3f}m")
        results.append({'frame_id': frame_id, 'ade': ade})
    
    # Summary
    if results:
        ades = [r['ade'] for r in results]
        print(f"\n{'='*50}")
        print(f"Processed {len(results)} frames")
        print(f"Mean ADE: {np.mean(ades):.3f}m")
        print(f"Median ADE: {np.median(ades):.3f}m")
        print(f"Min/Max: {np.min(ades):.3f}m / {np.max(ades):.3f}m")
        
        # Save results
        df = pd.DataFrame(results)
        df.to_csv(f'{infer_dir}/inference_results_strict_100.csv', index=False)
        print(f"\nResults saved to: {infer_dir}/inference_results_strict_100.csv")
    
    return results

if __name__ == '__main__':
    run_quick_inference()
