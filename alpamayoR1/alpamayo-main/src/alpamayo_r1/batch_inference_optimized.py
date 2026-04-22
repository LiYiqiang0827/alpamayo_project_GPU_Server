#!/usr/bin/env python3
"""
Step 3 优化版: 批量推理脚本
- 3条轨迹预测
- 坐标轴固定范围
- 多线程IO
- 批量后处理画图
- 计时功能
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import copy
import numpy as np
import pandas as pd
import torch
import json
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Alpamayo 导入
import sys
sys.path.insert(0, '/home/user/mikelee/alpamayo-main/src')
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper
from einops import rearrange

# ==================== 配置 ====================
CLIP_ID = '01d3588e-bca7-4a18-8e74-c6cfe9e996db'
BASE_DIR = '/data01/vla/data/data_sample_chunk0'
INFER_DIR = f'{BASE_DIR}/infer/{CLIP_ID}'
DATA_DIR = f'{INFER_DIR}/data'
RESULT_DIR = f'{INFER_DIR}/result'

# 模型参数
MODEL_PATH = "/data01/vla/models--nvidia--Alpamayo-R1-10B/snapshots/22fab1399111f50b52bfbe5d8b809f39bd4c2fe1"
NUM_TRAJ_SAMPLES = 1        # 6条轨迹 (对比测试)
TOP_P = 0.98
TEMPERATURE = 0.6
MAX_GENERATION_LENGTH = 256

# 推理帧数设置
MAX_FRAMES = 10             # 限制推理帧数
START_FRAME = 500           # 从Frame 500开始

# 相机顺序
CAMERA_ORDER = [
    'camera_cross_left_120fov',
    'camera_front_wide_120fov', 
    'camera_cross_right_120fov',
    'camera_front_tele_30fov'
]

# 列名映射
cam_cols = {
    'camera_cross_left_120fov': ['cam_left_f0', 'cam_left_f1', 'cam_left_f2', 'cam_left_f3'],
    'camera_front_wide_120fov': ['cam_front_f0', 'cam_front_f1', 'cam_front_f2', 'cam_front_f3'],
    'camera_cross_right_120fov': ['cam_right_f0', 'cam_right_f1', 'cam_right_f2', 'cam_right_f3'],
    'camera_front_tele_30fov': ['cam_tele_f0', 'cam_tele_f1', 'cam_tele_f2', 'cam_tele_f3'],
}

# ==================== 工具函数 ====================
def rotate_90cc(xy):
    """逆时针旋转90度: (x, y) -> (-y, x)"""
    return np.stack([-xy[1], xy[0]], axis=0)

def load_images_for_frame(row, data_dir):
    """加载一个推理帧的16张图片"""
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
    """加载egomotion数据"""
    prefix = row['ego_file_prefix']
    ego_dir = f'{data_dir}/egomotion'
    
    hist_local = np.load(f'{ego_dir}/{prefix}_history_local.npy', allow_pickle=True).item()
    future_gt = np.load(f'{ego_dir}/{prefix}_future_gt.npy', allow_pickle=True).item()
    
    hist_xyz = torch.from_numpy(hist_local['xyz']).float().unsqueeze(0).unsqueeze(0)
    hist_rot = torch.from_numpy(hist_local['rotation_matrix']).float().unsqueeze(0).unsqueeze(0)
    future_xyz = torch.from_numpy(future_gt['xyz']).float().unsqueeze(0).unsqueeze(0)
    
    return hist_xyz, hist_rot, future_xyz

# ==================== 推理函数 ====================
def run_inference(model, processor, row, data_dir):
    """对单个推理帧运行推理"""
    
    # 加载数据
    image_frames = load_images_for_frame(row, data_dir)
    hist_xyz, hist_rot, future_xyz = load_egomotion_for_frame(row, data_dir)
    
    # 构建模型输入
    messages = helper.create_message(image_frames.flatten(0, 1))
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
        "ego_history_xyz": hist_xyz,
        "ego_history_rot": hist_rot,
    }
    model_inputs = helper.to_device(model_inputs, "cuda")
    
    torch.cuda.manual_seed_all(42)
    
    # 计时开始
    torch.cuda.synchronize()
    inference_start = time.time()
    
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=copy.deepcopy(model_inputs),
            top_p=TOP_P,
            temperature=TEMPERATURE,
            num_traj_samples=NUM_TRAJ_SAMPLES,
            max_generation_length=MAX_GENERATION_LENGTH,
            return_extra=True,
        )
    
    torch.cuda.synchronize()
    inference_time = time.time() - inference_start
    
    # 提取结果
    pred_xyz_np = pred_xyz.cpu().numpy()[0, 0, :, :, :2]  # (3, 64, 2)
    gt_xy = future_xyz.cpu().numpy()[0, 0, :, :2]
    
    # 计算 minADE
    diff = np.linalg.norm(pred_xyz_np - gt_xy[None, ...], axis=2)
    ade_per_traj = diff.mean(axis=1)
    min_ade = float(ade_per_traj.min())
    best_traj_idx = int(ade_per_traj.argmin())
    
    # CoC文本
    coc_texts = extra.get("cot", [[[]]])[0]
    if isinstance(coc_texts, np.ndarray):
        coc_texts = coc_texts.tolist()
    
    return {
        'pred_xyz': pred_xyz_np,
        'gt_xy': gt_xy,
        'min_ade': min_ade,
        'best_traj_idx': best_traj_idx,
        'ade_all': [float(x) for x in ade_per_traj],
        'coc_texts': coc_texts,
        'inference_time': inference_time,
    }

# ==================== 批量画图函数 ====================
def plot_trajectory_batch(args):
    """批量画单个轨迹图 (用于多线程)"""
    infer_idx, pred_xyz, gt_xy, result_dir = args
    
    plt.figure(figsize=(10, 10))
    
    # 3条预测轨迹
    colors = plt.cm.tab10(np.linspace(0, 1, NUM_TRAJ_SAMPLES))
    for i in range(pred_xyz.shape[0]):
        pred_xy = pred_xyz[i].T
        pred_xy_rot = rotate_90cc(pred_xy)
        plt.plot(pred_xy_rot[0], pred_xy_rot[1], "o-", 
                color=colors[i], label=f"Predicted #{i+1}", markersize=2)
    
    # 真值轨迹
    gt_xy_rot = rotate_90cc(gt_xy.T)
    plt.plot(gt_xy_rot[0], gt_xy_rot[1], "r-", label="Ground Truth", linewidth=2)
    
    # 固定坐标轴范围
    plt.xlim(-30, 30)
    plt.ylim(0, 100)
    
    plt.xlabel("X (meters, left-right)", fontsize=12)
    plt.ylabel("Y (meters, forward)", fontsize=12)
    plt.title(f"Trajectory Prediction - Frame {infer_idx:06d}", fontsize=14)
    plt.legend(loc="upper left", fontsize=9)
    plt.grid(True, alpha=0.3)
    
    plt.savefig(f'{result_dir}/trajectory_{infer_idx:06d}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return infer_idx

# ==================== IO线程函数 ====================
def save_npy_async(args):
    """异步保存npy文件"""
    pred_xyz, path = args
    np.save(path, pred_xyz)
    return path

def load_images_async(args):
    """异步加载图片"""
    row, data_dir = args
    return load_images_for_frame(row, data_dir)

# ==================== 主函数 ====================
def main():
    print('=== Step 3 优化版: 批量推理 ===\n')
    print(f'配置: {NUM_TRAJ_SAMPLES}条轨迹, 从Frame {START_FRAME}开始, 共{MAX_FRAMES}帧\n')
    
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    # 1. 加载模型
    print('1. 加载模型...')
    model_load_start = time.time()
    model = AlpamayoR1.from_pretrained(MODEL_PATH, dtype=torch.bfloat16).to("cuda")
    processor = helper.get_processor(model.tokenizer)
    print(f'   模型加载完成! 耗时: {time.time() - model_load_start:.1f}s')
    
    # 2. 加载索引
    print('\n2. 加载推理索引表...')
    index_df = pd.read_csv(f'{DATA_DIR}/inference_index_v2.csv')
    print(f'   总推理帧数: {len(index_df)}')
    
    start_idx = START_FRAME
    num_frames = min(MAX_FRAMES, len(index_df) - start_idx)
    print(f'   本次推理帧数: {num_frames} (从索引{start_idx}开始)')
    
    # 3. 批量推理
    print('\n3. 开始批量推理...')
    results = []
    inference_times = []
    
    for idx in tqdm(range(start_idx, start_idx + num_frames), desc='推理进度'):
        row = index_df.iloc[idx]
        infer_idx = int(row['infer_idx'])
        
        try:
            result = run_inference(model, processor, row, DATA_DIR)
            inference_times.append(result['inference_time'])
            
            # 立即保存npy (简单直接)
            np.save(f'{RESULT_DIR}/pred_{infer_idx:06d}.npy', result['pred_xyz'])
            
            results.append({
                'infer_idx': infer_idx,
                't0_timestamp': row['t0_timestamp'],
                'min_ade': result['min_ade'],
                'best_traj_idx': result['best_traj_idx'],
                'ade_all': json.dumps(result['ade_all']),
                'coc_text': json.dumps(result['coc_texts']),
                'inference_time_ms': round(result['inference_time'] * 1000, 1),
                'pred_file': f'pred_{infer_idx:06d}.npy',
                'plot_file': f'trajectory_{infer_idx:06d}.png',
            })
            
        except Exception as e:
            print(f'\n   错误: Frame {infer_idx} 推理失败: {e}')
            results.append({
                'infer_idx': infer_idx,
                't0_timestamp': row['t0_timestamp'],
                'min_ade': None,
                'error': str(e),
            })
    
    # 4. 批量画图 (多线程)
    print('\n4. 批量画图 (多线程)...')
    plot_args = []
    for r in results:
        if r.get('min_ade') is not None:
            pred_xyz = np.load(f'{RESULT_DIR}/{r["pred_file"]}')
            # 加载gt_xy
            row = index_df[index_df['infer_idx'] == r['infer_idx']].iloc[0]
            _, _, future_xyz = load_egomotion_for_frame(row, DATA_DIR)
            gt_xy = future_xyz.cpu().numpy()[0, 0, :, :2]
            plot_args.append((r['infer_idx'], pred_xyz, gt_xy, RESULT_DIR))
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(tqdm(executor.map(plot_trajectory_batch, plot_args), 
                  total=len(plot_args), desc='画图进度'))
    
    # 5. 保存结果CSV
    print('\n5. 保存结果...')
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('infer_idx')
    results_df.to_csv(f'{RESULT_DIR}/inference_results.csv', index=False)
    
    # 统计
    success_count = results_df['min_ade'].notna().sum()
    print(f'   成功: {success_count}/{len(results_df)}')
    
    if success_count > 0:
        min_ades = results_df['min_ade'].dropna()
        print(f'   minADE 均值: {min_ades.mean():.2f} m')
        print(f'   minADE 中位数: {min_ades.median():.2f} m')
        print(f'   minADE 范围: [{min_ades.min():.2f}, {min_ades.max():.2f}] m')
        
        # 推理时间统计
        times = results_df['inference_time_ms'].dropna()
        print(f'\n   GPU纯推理时间统计:')
        print(f'     均值: {times.mean():.1f} ms')
        print(f'     中位数: {times.median():.1f} ms')
        print(f'     范围: [{times.min():.1f}, {times.max():.1f}] ms')
    
    print(f'\n结果保存至: {RESULT_DIR}/')
    print('  - inference_results.csv (CoC + minADE + 推理时间)')
    print('  - pred_*.npy (3条轨迹预测)')
    print('  - trajectory_*.png (轨迹可视化)')

if __name__ == '__main__':
    main()
