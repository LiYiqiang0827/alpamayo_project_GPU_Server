#!/usr/bin/env python3
"""
Run inference on preprocessed data using strict timestamp alignment
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# Add Alpamayo path
sys.path.insert(0, '/home/user/mikelee/alpamayo-main/src')

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from transformers import AutoProcessor

def load_image_frame(clip_path, cam_name, frame_idx):
    """Load a single image frame"""
    # Image files are named like: 000044.jpg (not including camera name)
    img_path = clip_path / 'data' / 'camera_images' / cam_name / f'{frame_idx:06d}.jpg'
    if not img_path.exists():
        print(f"  Image not found: {img_path}")
        return None
    img = Image.open(img_path)
    return img

def run_inference_strict(clip_id, data_root, num_frames=100):
    """Run inference on preprocessed data"""
    
    # Paths
    infer_path = Path(data_root) / 'infer' / clip_id
    data_path = infer_path / 'data'
    
    # Load index
    index_df = pd.read_csv(data_path / 'inference_index_strict.csv')
    
    print(f"Total available frames: {len(index_df)}")
    print(f"Will process: {min(num_frames, len(index_df))} frames")
    
    # Load model
    print("\nLoading model...")
    # Use local cache paths
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    
    model_cache = '/data01/vla/models--nvidia--Alpamayo-R1-10B/snapshots/22fab1399111f50b52bfbe5d8b809f39bd4c2fe1'
    
    model = AlpamayoR1.from_pretrained(
        model_cache,
        dtype=torch.bfloat16,
        local_files_only=True,
    ).to("cuda")
    
    # Load processor from local cache
    from transformers import AutoProcessor
    processor_cache = os.path.expanduser('~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/*')
    import glob
    processor_cache = glob.glob(processor_cache)[0]
    
    processor = AutoProcessor.from_pretrained(
        processor_cache,
        min_pixels=256*28*28,
        max_pixels=1280*28*28,
        local_files_only=True,
    )
    processor.tokenizer = model.tokenizer
    
    # Results storage
    results = []
    
    # Process frames
    num_to_process = min(num_frames, len(index_df))
    
    for i in range(num_to_process):
        row = index_df.iloc[i]
        frame_id = int(row['frame_id'])
        ego_idx = int(row['ego_idx'])
        
        print(f"\n[{i+1}/{num_to_process}] Processing frame {frame_id} (ego_idx={ego_idx})")
        
        # Load camera images
        images = []
        cam_names = ['camera_cross_left_120fov', 'camera_front_wide_120fov',
                     'camera_cross_right_120fov', 'camera_front_tele_30fov']
        
        img_load_success = True
        for j, cam in enumerate(cam_names):
            # Load f0, f1, f2, f3 for this camera
            cam_imgs = []
            for k in range(4):
                idx_col = f'{cam}_f{k}_idx'
                if idx_col not in row:
                    img_load_success = False
                    break
                frame_idx = int(row[idx_col])
                img = load_image_frame(infer_path, cam, frame_idx)
                if img is None:
                    img_load_success = False
                    break
                cam_imgs.append(img)
            
            if not img_load_success:
                break
            
            # Use f3 (current time) as representative image
            images.append(cam_imgs[3])
        
        if not img_load_success:
            print(f"  WARNING: Failed to load images for frame {frame_id}")
            continue
        
        # Load egomotion history
        history_path = data_path / 'egomotion' / f'frame_{frame_id:06d}_history.npy'
        if not history_path.exists():
            print(f"  WARNING: History not found: {history_path}")
            continue
        
        history_data = np.load(history_path)
        # Extract xyz from history (columns 4,5,6: x,y,z)
        ego_history_xyz = torch.from_numpy(history_data[:, 4:7]).float().unsqueeze(0)
        
        # Load ground truth future
        future_path = data_path / 'egomotion' / f'frame_{frame_id:06d}_future_gt.npy'
        if not future_path.exists():
            print(f"  WARNING: Future GT not found: {future_path}")
            continue
        
        future_gt = np.load(future_path)
        gt_xy = future_gt[:, 1:3]  # x, y columns
        
        # Prepare inputs
        text_prompt = "<image>" * 4 + " Predict the future trajectory."
        
        inputs = processor(
            text=[text_prompt],
            images=images,
            return_tensors="pt"
        )
        
        # Move to device
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(model.device)
        
        ego_history_xyz = ego_history_xyz.to(model.device)
        
        # Run inference
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                ego_history_xyz=ego_history_xyz,
                max_new_tokens=512,
                num_traj_samples=1,
            )
        
        # Decode trajectory
        pred_text = processor.batch_decode(outputs['sequences'], skip_special_tokens=True)[0]
        
        # Get predicted trajectory
        if 'pred_xyz' in outputs:
            pred_xyz = outputs['pred_xyz']
            pred_xy = pred_xyz[0, 0, :, :2].cpu().numpy()  # [64, 2]
            diff = np.linalg.norm(pred_xy - gt_xy, axis=1)
            ade = diff.mean()
        else:
            ade = float('nan')
        
        print(f"  minADE: {ade:.3f}m")
        
        results.append({
            'frame_id': frame_id,
            'ego_idx': ego_idx,
            'ade': ade,
            'pred_text': pred_text[:100]  # First 100 chars
        })
    
    # Summary
    print("\n" + "="*50)
    print(f"Inference complete! Processed {len(results)} frames")
    
    if len(results) > 0:
        ades = [r['ade'] for r in results]
        print(f"ADE stats:")
        print(f"  Mean: {np.mean(ades):.3f}m")
        print(f"  Median: {np.median(ades):.3f}m")
        print(f"  Min: {np.min(ades):.3f}m")
        print(f"  Max: {np.max(ades):.3f}m")
        
        # Save results
        results_df = pd.DataFrame(results)
        output_file = infer_path / 'inference_results_first100.csv'
        results_df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
    
    return results

if __name__ == '__main__':
    clip_id = '01d3588e-bca7-4a18-8e74-c6cfe9e996db'
    data_root = '/data01/vla/data/data_sample_chunk0'
    
    results = run_inference_strict(clip_id, data_root, num_frames=100)
