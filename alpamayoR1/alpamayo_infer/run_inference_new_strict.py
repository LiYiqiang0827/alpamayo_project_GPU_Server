#!/usr/bin/env python3
"""
推理脚本 - 使用新的严格预处理索引 (基于 run_inference.py)
"""
import os
import sys
import argparse

# 必须先解析参数，再设置GPU，最后import torch
parser = argparse.ArgumentParser(description='批量推理 - 使用新严格预处理索引')
parser.add_argument('--clip', type=str, required=True, help='Clip ID')
parser.add_argument('--num_frames', type=int, default=100, help='推理帧数')
parser.add_argument('--traj', type=int, default=1, help='轨迹数量 (1/3/6)')
parser.add_argument('--gpu', type=int, default=0, help='使用的GPU ID (默认: 0)')
args = parser.parse_args()

# 设置使用的GPU（必须在import torch之前）
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
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

def load_images_for_frame_strict(row, data_dir):
    """从新的严格索引格式加载图片"""
    images = []
    for cam in CAMERA_ORDER:
        # 新格式: {cam}_f0_idx, {cam}_f1_idx, {cam}_f2_idx, {cam}_f3_idx
        for t in range(4):
            idx_col = f'{cam}_f{t}_idx'
            frame_idx = int(row[idx_col])
            img_path = f'{data_dir}/camera_images/{cam}/{frame_idx:06d}.jpg'
            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img)
            images.append(img_array)
    
    images = np.stack(images, axis=0)  # (16, H, W, 3)
    images = rearrange(images, '(c t) h w ch -> c t ch h w', c=4, t=4)
    return torch.from_numpy(images).float()

def load_egomotion_for_frame_strict(row, data_dir):
    """从新的严格索引格式加载egomotion，并转换到局部坐标系"""
    frame_id = int(row['frame_id'])
    ego_dir = f'{data_dir}/egomotion'
    
    # 新格式: frame_{id:06d}_history.npy (numpy array directly)
    history = np.load(f'{ego_dir}/frame_{frame_id:06d}_history.npy', allow_pickle=False)
    future = np.load(f'{ego_dir}/frame_{frame_id:06d}_future_gt.npy', allow_pickle=False)
    
    # history: (16, 11) - columns: timestamp, qx, qy, qz, qw, x, y, z, vx, vy, vz
    # future: (64, 4) - columns: timestamp, x, y, z
    
    # 提取历史 xyz 和 quaternions (世界坐标系)
    # history columns: timestamp, qx, qy, qz, qw, x, y, z, vx, vy, vz
    hist_xyz_world = history[:, 5:8]  # x, y, z (columns 5,6,7)
    hist_quat = history[:, 1:5]  # qx, qy, qz, qw (columns 1,2,3,4)
    
    # 提取未来 xyz (世界坐标系)
    # future columns: timestamp, x, y, z
    future_xyz_world = future[:, 1:4]  # x, y, z (columns 1,2,3)
    
    # 获取 t0 时刻的位姿（历史最后一个点）
    t0_xyz = hist_xyz_world[-1].copy()
    t0_quat = hist_quat[-1].copy()
    
    # Convert quaternions to rotation matrices
    from scipy.spatial.transform import Rotation as R
    hist_rot = R.from_quat(hist_quat).as_matrix()  # (16, 3, 3)
    t0_rot = R.from_quat(t0_quat)
    t0_rot_inv = t0_rot.inv()
    
    # 将历史轨迹转换到局部坐标系（相对于 t0）
    hist_xyz_local = t0_rot_inv.apply(hist_xyz_world - t0_xyz)  # (16, 3)
    hist_rot_local = (t0_rot_inv * R.from_quat(hist_quat)).as_matrix()  # (16, 3, 3)
    
    # 将未来轨迹转换到局部坐标系
    # 局部坐标 = R_t0_inv × (世界坐标 - t0_xyz)
    future_xyz_local = t0_rot_inv.apply(future_xyz_world - t0_xyz)  # (64, 3)
    
    # Add dimensions for model: (batch=1, traj_group=1, steps, dim)
    hist_xyz_t = torch.from_numpy(hist_xyz_local).float().unsqueeze(0).unsqueeze(0)
    hist_rot_t = torch.from_numpy(hist_rot_local).float().unsqueeze(0).unsqueeze(0)
    future_xyz_t = torch.from_numpy(future_xyz_local).float().unsqueeze(0).unsqueeze(0)
    
    return hist_xyz_t, hist_rot_t, future_xyz_t

def run_inference(model, processor, row, data_dir, num_traj, top_p, temp, max_len):
    image_frames = load_images_for_frame_strict(row, data_dir)
    hist_xyz, hist_rot, future_xyz = load_egomotion_for_frame_strict(row, data_dir)
    
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
    
    pred_xyz_np = pred_xyz.cpu().numpy()[0, 0, 0, :, :2]
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
    CLIP_ID = args.clip
    NUM_FRAMES = args.num_frames
    NUM_TRAJ = args.traj
    GPU_ID = args.gpu
    
    BASE_DIR = '/data01/vla/data/data_sample_chunk0'
    INFER_DIR = f'{BASE_DIR}/infer/{CLIP_ID}'
    DATA_DIR = f'{INFER_DIR}/data'
    RESULT_DIR = f'{INFER_DIR}/result_strict'
    
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    print(f'=== 推理 ({CLIP_ID}) - 使用新严格预处理索引 ===\n')
    print(f'配置: {NUM_FRAMES}帧, {NUM_TRAJ}条轨迹, GPU {GPU_ID}')
    
    print('\n1. 加载模型...')
    model = AlpamayoR1.from_pretrained(MODEL_PATH, dtype=torch.bfloat16).to("cuda")
    processor = helper.get_processor(model.tokenizer)
    print('   模型加载完成!')
    
    print('\n2. 加载新严格预处理索引...')
    index_df = pd.read_csv(f'{DATA_DIR}/inference_index_strict.csv')
    
    total_frames = len(index_df)
    print(f'   总帧数: {total_frames}')
    
    target_frames = min(NUM_FRAMES, total_frames)
    print(f'   本次推理帧数: {target_frames}')
    
    print('\n3. 开始推理...')
    results = []
    inference_times = []
    
    for idx in tqdm(range(target_frames), desc='推理进度'):
        row = index_df.iloc[idx]
        frame_id = int(row['frame_id'])
        
        try:
            result = run_inference(model, processor, row, DATA_DIR, NUM_TRAJ, 0.98, 0.6, 256)
            inference_times.append(result['inference_time_ms'])
            
            np.save(f'{RESULT_DIR}/pred_{frame_id:06d}.npy', result['pred_xyz'])
            
            results.append({
                'frame_id': frame_id,
                'ego_idx': int(row['ego_idx']),
                'ade': result['ade'],
                'inference_time_ms': result['inference_time_ms'],
                'coc_text': json.dumps(result['coc_texts']),
            })
        except Exception as e:
            print(f'\n   错误: Frame {frame_id}: {e}')
            import traceback
            traceback.print_exc()
            results.append({'frame_id': frame_id, 'error': str(e)})
    
    print('\n4. 保存结果...')
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{RESULT_DIR}/inference_results_strict.csv', index=False)
    
    print(f'\n=== 结果统计 ===')
    print(f'成功: {len([r for r in results if r.get("ade")])}/{len(results)}')
    
    if inference_times:
        print(f'\nGPU纯推理时间 ({NUM_TRAJ}条轨迹):')
        print(f'  均值: {np.mean(inference_times):.1f} ms')
        print(f'  中位数: {np.median(inference_times):.1f} ms')
        
        ades = [r['ade'] for r in results if r.get('ade')]
        print(f'\nADE统计:')
        print(f'  均值: {np.mean(ades):.2f} m')
        print(f'  中位数: {np.median(ades):.2f} m')
    
    print(f'\n结果保存至: {RESULT_DIR}/')

if __name__ == '__main__':
    main()
