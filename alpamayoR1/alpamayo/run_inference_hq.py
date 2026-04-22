#!/usr/bin/env python3
"""
连续推理脚本 - 使用高质量索引
"""
import os
import sys
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import copy
import numpy as np
import pandas as pd
import torch
import json
import time
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from einops import rearrange

sys.path.insert(0, '/home/user/mikelee/alpamayo-main/src')
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper

MODEL_PATH = "/data01/vla/models--nvidia--Alpamayo-R1-10B/snapshots/22fab1399111f50b52bfbe5d8b809f39bd4c2fe1"
CAMERA_ORDER = [
    'camera_cross_left_120fov',
    'camera_front_wide_120fov', 
    'camera_cross_right_120fov',
    'camera_front_tele_30fov'
]

cam_cols = {
    'camera_cross_left_120fov': ['cam_left_f0', 'cam_left_f1', 'cam_left_f2', 'cam_left_f3'],
    'camera_front_wide_120fov': ['cam_front_f0', 'cam_front_f1', 'cam_front_f2', 'cam_front_f3'],
    'camera_cross_right_120fov': ['cam_right_f0', 'cam_right_f1', 'cam_right_f2', 'cam_right_f3'],
    'camera_front_tele_30fov': ['cam_tele_f0', 'cam_tele_f1', 'cam_tele_f2', 'cam_tele_f3'],
}

def load_images_for_frame(row, data_dir):
    images = []
    for cam in CAMERA_ORDER:
        for col in cam_cols[cam]:
            img_path = f'{data_dir}/{row[col]}'
            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img)
            images.append(img_array)
    images = np.stack(images, axis=0)
    images = rearrange(images, '(c t) h w ch -> c t ch h w', c=4, t=4)
    return torch.from_numpy(images).float()

def load_egomotion_for_frame(row, data_dir):
    prefix = row['ego_file_prefix']
    ego_dir = f'{data_dir}/egomotion'
    hist_local = np.load(f'{ego_dir}/{prefix}_history_local.npy', allow_pickle=True).item()
    future_gt = np.load(f'{ego_dir}/{prefix}_future_gt.npy', allow_pickle=True).item()
    
    hist_xyz = torch.from_numpy(hist_local['xyz']).float().unsqueeze(0).unsqueeze(0)
    hist_rot = torch.from_numpy(hist_local['rotation_matrix']).float().unsqueeze(0).unsqueeze(0)
    future_xyz = torch.from_numpy(future_gt['xyz']).float().unsqueeze(0).unsqueeze(0)
    return hist_xyz, hist_rot, future_xyz

def run_inference(model, processor, row, data_dir, num_traj, top_p, temp, max_len):
    image_frames = load_images_for_frame(row, data_dir)
    hist_xyz, hist_rot, future_xyz = load_egomotion_for_frame(row, data_dir)
    
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
    
    torch.cuda.synchronize()
    inference_start = time.time()
    
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=copy.deepcopy(model_inputs),
            top_p=top_p, temperature=temp,
            num_traj_samples=num_traj,
            max_generation_length=max_len,
            return_extra=True,
        )
    
    torch.cuda.synchronize()
    inference_time = time.time() - inference_start
    
    # 计算minADE
    gt_xy = future_xyz.cpu().numpy()[0, 0, :, :2]
    
    if num_traj > 1:
        min_ade = float('inf')
        best_traj = 0
        for t in range(pred_xyz.shape[2]):
            pred_t = pred_xyz.cpu().numpy()[0, 0, t, :, :2]
            diff_t = np.linalg.norm(pred_t - gt_xy, axis=1)
            ade_t = diff_t.mean()
            if ade_t < min_ade:
                min_ade = ade_t
                best_traj = t
        pred_xyz_np = pred_xyz.cpu().numpy()[0, 0, best_traj, :, :2]
    else:
        pred_xyz_np = pred_xyz.cpu().numpy()[0, 0, 0, :, :2]
        diff = np.linalg.norm(pred_xyz_np - gt_xy, axis=1)
        min_ade = diff.mean()
    
    coc_texts = extra.get("cot", [[[]]])[0]
    if isinstance(coc_texts, np.ndarray):
        coc_texts = coc_texts.tolist()
    
    return {
        'pred_xyz': pred_xyz_np,
        'gt_xy': gt_xy,
        'minADE': float(min_ade),
        'coc_texts': coc_texts,
        'inference_time_ms': round(inference_time * 1000, 1),
    }

def main():
    parser = argparse.ArgumentParser(description='批量推理 (高质量索引)')
    parser.add_argument('--clip', type=str, required=True)
    parser.add_argument('--chunk', type=int, default=0)
    parser.add_argument('--step', type=int, default=10)
    parser.add_argument('--traj', type=int, default=1)
    args = parser.parse_args()
    
    CLIP_ID = args.clip
    CHUNK = args.chunk
    STEP = args.step
    NUM_TRAJ = args.traj
    
    BASE_DIR = f'/data01/vla/data/data_sample_chunk{CHUNK}'
    INFER_DIR = f'{BASE_DIR}/infer/{CLIP_ID}'
    DATA_DIR = f'{INFER_DIR}/data'
    RESULT_DIR = f'{INFER_DIR}/result'
    
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    print(f'=== 连续推理 ({CLIP_ID}) - 使用高质量索引 ===\n')
    print(f'配置: 每{STEP}帧推理一次, {NUM_TRAJ}条轨迹')
    
    print('\n1. 加载模型...')
    model = AlpamayoR1.from_pretrained(MODEL_PATH, dtype=torch.bfloat16).to("cuda")
    processor = helper.get_processor(model.tokenizer)
    print('   模型加载完成!')
    
    print('\n2. 加载高质量索引...')
    index_df = pd.read_csv(f'{DATA_DIR}/inference_index_high_quality.csv')
    index_df['ego_file_prefix'] = index_df['infer_idx'].apply(lambda x: f'ego_{x:06d}')
    
    total_frames = len(index_df)
    print(f'   高质量索引: {total_frames} 帧')
    
    target_indices = list(range(0, total_frames, STEP))
    print(f'   本次推理帧数: {len(target_indices)} (每{STEP}帧一次)')
    
    print('\n3. 开始连续推理...')
    results = []
    inference_times = []
    
    for idx in tqdm(target_indices, desc='推理进度'):
        row = index_df.iloc[idx]
        infer_idx = int(row['infer_idx'])
        
        try:
            result = run_inference(model, processor, row, DATA_DIR, NUM_TRAJ, 0.98, 0.6, 256)
            inference_times.append(result['inference_time_ms'])
            
            np.save(f'{RESULT_DIR}/pred_{infer_idx:06d}.npy', result['pred_xyz'])
            
            results.append({
                'infer_idx': infer_idx,
                't0_timestamp': row['t0_timestamp'],
                'minADE': result['minADE'],
                'inference_time_ms': result['inference_time_ms'],
                'coc_text': json.dumps(result['coc_texts']),
            })
        except Exception as e:
            print(f'\n   错误: Frame {infer_idx}: {e}')
            results.append({'infer_idx': infer_idx, 'error': str(e)})
    
    print('\n4. 保存结果...')
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{RESULT_DIR}/continuous_inference_results.csv', index=False)
    
    print(f'\n=== 结果统计 ===')
    print(f'成功: {len([r for r in results if r.get("minADE")])}/{len(results)}')
    
    if inference_times:
        print(f'\nGPU纯推理时间 ({NUM_TRAJ}条轨迹):')
        print(f'  均值: {np.mean(inference_times):.1f} ms')
        print(f'  中位数: {np.median(inference_times):.1f} ms')
        
        ades = [r['minADE'] for r in results if r.get('minADE')]
        print(f'\nminADE统计:')
        print(f'  均值: {np.mean(ades):.2f} m')
        print(f'  中位数: {np.median(ades):.2f} m')
        print(f'  最小: {np.min(ades):.2f} m')
        print(f'  最大: {np.max(ades):.2f} m')
    
    print(f'\n结果保存至: {RESULT_DIR}/')

if __name__ == '__main__':
    main()
