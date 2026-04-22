#!/usr/bin/env python3
"""
连续推理脚本 - 每10帧推理一次
"""
import os
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

import sys
sys.path.insert(0, '/home/user/mikelee/alpamayo-main/src')
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper

# 配置
CLIP_ID = '01d3588e-bca7-4a18-8e74-c6cfe9e996db'
BASE_DIR = '/data01/vla/data/data_sample_chunk0'
INFER_DIR = f'{BASE_DIR}/infer/{CLIP_ID}'
DATA_DIR = f'{INFER_DIR}/data'
RESULT_DIR = f'{INFER_DIR}/result'

MODEL_PATH = "/data01/vla/models--nvidia--Alpamayo-R1-10B/snapshots/22fab1399111f50b52bfbe5d8b809f39bd4c2fe1"
NUM_TRAJ_SAMPLES = 1  # 1条轨迹
TOP_P = 0.98
TEMPERATURE = 0.6
MAX_GENERATION_LENGTH = 256

# 每10帧推理一次
STEP = 5
START_FRAME = 0

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

def run_inference(model, processor, row, data_dir):
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
            top_p=TOP_P, temperature=TEMPERATURE,
            num_traj_samples=NUM_TRAJ_SAMPLES,
            max_generation_length=MAX_GENERATION_LENGTH,
            return_extra=True,
        )
    
    torch.cuda.synchronize()
    inference_time = time.time() - inference_start
    
    pred_xyz_np = pred_xyz.cpu().numpy()[0, 0, 0, :, :2]  # (1, 64, 2) -> (64, 2)
    gt_xy = future_xyz.cpu().numpy()[0, 0, :, :2]
    
    diff = np.linalg.norm(pred_xyz_np - gt_xy, axis=1)
    ade = diff.mean()
    
    coc_texts = extra.get("cot", [[[]]])[0]
    if isinstance(coc_texts, np.ndarray):
        coc_texts = coc_texts.tolist()
    
    return {
        'pred_xyz': pred_xyz_np,
        'gt_xy': gt_xy,
        'ade': float(ade),
        'coc_texts': coc_texts,
        'inference_time_ms': round(inference_time * 1000, 1),
    }

def main():
    print('=== 连续推理: 1条轨迹, 每10帧一次 ===\n')
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    print('1. 加载模型...')
    model = AlpamayoR1.from_pretrained(MODEL_PATH, dtype=torch.bfloat16).to("cuda")
    processor = helper.get_processor(model.tokenizer)
    print('   模型加载完成!')
    
    print('\n2. 加载索引...')
    index_df = pd.read_csv(f'{DATA_DIR}/inference_index_high_quality.csv')
    total_frames = len(index_df)
    print(f'   总帧数: {total_frames}')
    
    # 计算要推理的帧数 (每10帧一次)
    target_indices = list(range(START_FRAME, total_frames, STEP))
    print(f'   本次推理帧数: {len(target_indices)} (每{STEP}帧一次)')
    
    print('\n3. 开始连续推理...')
    results = []
    inference_times = []
    
    for idx in tqdm(target_indices, desc='推理进度'):
        row = index_df.iloc[idx]
        infer_idx = int(row['infer_idx'])
        
        try:
            result = run_inference(model, processor, row, DATA_DIR)
            inference_times.append(result['inference_time_ms'])
            
            np.save(f'{RESULT_DIR}/pred_{infer_idx:06d}.npy', result['pred_xyz'])
            
            results.append({
                'infer_idx': infer_idx,
                't0_timestamp': row['t0_timestamp'],
                'ade': result['ade'],
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
    print(f'成功: {len([r for r in results if r.get("ade")])}/{len(results)}')
    
    if inference_times:
        print(f'\nGPU纯推理时间 (1条轨迹):')
        print(f'  均值: {np.mean(inference_times):.1f} ms')
        print(f'  中位数: {np.median(inference_times):.1f} ms')
        print(f'  范围: [{np.min(inference_times):.1f}, {np.max(inference_times):.1f}] ms')
        
        ades = [r['ade'] for r in results if r.get('ade')]
        print(f'\nADE统计:')
        print(f'  均值: {np.mean(ades):.2f} m')
        print(f'  中位数: {np.median(ades):.2f} m')
    
    print(f'\n结果保存至: {RESULT_DIR}/')

if __name__ == '__main__':
    main()
