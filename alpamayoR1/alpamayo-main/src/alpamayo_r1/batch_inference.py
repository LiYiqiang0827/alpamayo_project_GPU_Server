#!/usr/bin/env python3
"""
Step 3: 批量推理脚本
加载预处理数据，运行 Alpamayo-R1 推理，保存结果
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
import matplotlib
matplotlib.use('Agg')  # 无头模式
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

# Alpamayo 导入
import sys
sys.path.insert(0, '/home/user/mikelee/alpamayo-main/src')
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper
from einops import rearrange

# 配置
CLIP_ID = '01d3588e-bca7-4a18-8e74-c6cfe9e996db'
BASE_DIR = '/data01/vla/data/data_sample_chunk0'
INFER_DIR = f'{BASE_DIR}/infer/{CLIP_ID}'
DATA_DIR = f'{INFER_DIR}/data'
RESULT_DIR = f'{INFER_DIR}/result'

# 模型参数
MODEL_PATH = "/data01/vla/models--nvidia--Alpamayo-R1-10B/snapshots/22fab1399111f50b52bfbe5d8b809f39bd4c2fe1"

# 模型参数
NUM_TRAJ_SAMPLES = 6  # 6条轨迹
TOP_P = 0.98
TEMPERATURE = 0.6
MAX_GENERATION_LENGTH = 256
MAX_FRAMES = 10  # 限制推理帧数 (用于测试)
START_FRAME = 500  # 从Frame 500开始（避免边界问题）

# 相机顺序 (与模型训练时一致)
CAMERA_ORDER = [
    'camera_cross_left_120fov',
    'camera_front_wide_120fov', 
    'camera_cross_right_120fov',
    'camera_front_tele_30fov'
]

def rotate_90cc(xy):
    """逆时针旋转90度: (x, y) -> (-y, x)"""
    return np.stack([-xy[1], xy[0]], axis=0)

def load_images_for_frame(row, data_dir):
    """加载一个推理帧的16张图片 (4相机 × 4帧)"""
    images = []
    
    # 相机列名映射
    cam_cols = {
        'camera_cross_left_120fov': ['cam_left_f0', 'cam_left_f1', 'cam_left_f2', 'cam_left_f3'],
        'camera_front_wide_120fov': ['cam_front_f0', 'cam_front_f1', 'cam_front_f2', 'cam_front_f3'],
        'camera_cross_right_120fov': ['cam_right_f0', 'cam_right_f1', 'cam_right_f2', 'cam_right_f3'],
        'camera_front_tele_30fov': ['cam_tele_f0', 'cam_tele_f1', 'cam_tele_f2', 'cam_tele_f3'],
    }
    
    # 按相机顺序加载
    for cam in CAMERA_ORDER:
        for f in range(4):  # f0, f1, f2, f3
            col_name = cam_cols[cam][f]
            img_path = f'{data_dir}/{row[col_name]}'
            
            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img)  # (H, W, 3)
            images.append(img_array)
    
    # 转换为 tensor: (16, H, W, 3) -> (4, 4, 3, H, W)
    images = np.stack(images, axis=0)  # (16, H, W, 3)
    images = rearrange(images, '(c t) h w ch -> c t ch h w', c=4, t=4)
    return torch.from_numpy(images).float()

def load_egomotion_for_frame(row, data_dir):
    """加载一个推理帧的egomotion数据"""
    prefix = row['ego_file_prefix']
    ego_dir = f'{data_dir}/egomotion'
    
    # 加载局部坐标历史
    hist_local = np.load(f'{ego_dir}/{prefix}_history_local.npy', allow_pickle=True).item()
    
    # 加载未来真值
    future_gt = np.load(f'{ego_dir}/{prefix}_future_gt.npy', allow_pickle=True).item()
    
    # 转换为 torch tensor，添加batch维度
    hist_xyz = torch.from_numpy(hist_local['xyz']).float().unsqueeze(0).unsqueeze(0)  # (1, 1, 16, 3)
    hist_rot = torch.from_numpy(hist_local['rotation_matrix']).float().unsqueeze(0).unsqueeze(0)  # (1, 1, 16, 3, 3)
    
    future_xyz = torch.from_numpy(future_gt['xyz']).float().unsqueeze(0).unsqueeze(0)  # (1, 1, 64, 3)
    
    return hist_xyz, hist_rot, future_xyz

def run_inference(model, processor, row, data_dir):
    """对单个推理帧运行推理"""
    
    # 加载数据
    image_frames = load_images_for_frame(row, data_dir)  # (4, 4, 3, H, W)
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
    
    # 设置随机种子（可选，用于可重复性）
    torch.cuda.manual_seed_all(42)
    
    # 推理
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=copy.deepcopy(model_inputs),
            top_p=TOP_P,
            temperature=TEMPERATURE,
            num_traj_samples=NUM_TRAJ_SAMPLES,
            max_generation_length=MAX_GENERATION_LENGTH,
            return_extra=True,
        )
    
    # 提取结果
    # pred_xyz: (batch=1, n_traj_sets=1, num_traj_samples=6, 64, 3)
    pred_xyz_np = pred_xyz.cpu().numpy()[0, 0, :, :, :2]  # (6, 64, 2) - xy坐标
    
    # 真值
    gt_xy = future_xyz.cpu().numpy()[0, 0, :, :2]  # (64, 2)
    
    # 计算 minADE
    # pred_xy: (6, 64, 2), gt_xy: (64, 2)
    diff = np.linalg.norm(pred_xyz_np - gt_xy[None, ...], axis=2)  # (6, 64)
    ade_per_traj = diff.mean(axis=1)  # (6,) - 每条轨迹的ADE
    min_ade = float(ade_per_traj.min())  # 最小ADE - 转为Python float
    best_traj_idx = int(ade_per_traj.argmin())  # 最佳轨迹索引 - 转为Python int
    
    # CoC文本 - 确保可JSON序列化
    coc_texts = extra.get("cot", [[[]]])[0]  # 获取第一条batch的CoC
    if isinstance(coc_texts, np.ndarray):
        coc_texts = coc_texts.tolist()
    
    return {
        'pred_xyz': pred_xyz_np,  # (6, 64, 2)
        'gt_xy': gt_xy,  # (64, 2)
        'min_ade': min_ade,
        'best_traj_idx': best_traj_idx,
        'ade_all': [float(x) for x in ade_per_traj],  # 转为Python float列表
        'coc_texts': coc_texts,
    }

def plot_trajectories(pred_xyz, gt_xy, infer_idx, result_dir):
    """绘制轨迹图"""
    plt.figure(figsize=(10, 10))
    
    # 绘制6条预测轨迹
    colors = plt.cm.tab10(np.linspace(0, 1, 6))
    for i in range(pred_xyz.shape[0]):
        pred_xy = pred_xyz[i].T  # (2, 64)
        pred_xy_rot = rotate_90cc(pred_xy)
        plt.plot(pred_xy_rot[0], pred_xy_rot[1], "o-", 
                color=colors[i], label=f"Predicted #{i+1}", markersize=2)
    
    # 绘制真值轨迹
    gt_xy_rot = rotate_90cc(gt_xy.T)
    plt.plot(gt_xy_rot[0], gt_xy_rot[1], "r-", 
            label="Ground Truth", linewidth=2)
    
    # 设置坐标轴
    plt.xlim(-30, 30)  # X: 左右 -30m ~ +30m
    plt.ylim(0, 100)   # Y: 前方 0m ~ 100m
    
    plt.xlabel("X (meters, left-right)", fontsize=12)
    plt.ylabel("Y (meters, forward)", fontsize=12)
    plt.title(f"Trajectory Prediction - Frame {infer_idx:06d}", fontsize=14)
    plt.legend(loc="upper left", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # 保存
    plt.savefig(f'{result_dir}/trajectory_{infer_idx:06d}.png', 
                dpi=150, bbox_inches='tight')
    plt.close()

def main():
    print('=== Step 3: 批量推理 ===\n')
    
    # 创建结果目录
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    # 1. 加载模型
    print('1. 加载模型...')
    model = AlpamayoR1.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16
    ).to("cuda")
    processor = helper.get_processor(model.tokenizer)
    print('   模型加载完成！')
    
    # 2. 加载索引表
    print('\n2. 加载推理索引表...')
    index_df = pd.read_csv(f'{DATA_DIR}/inference_index_v2.csv')
    print(f'   总推理帧数: {len(index_df)}')
    
    # 限制推理帧数
    start_idx = START_FRAME
    num_frames = min(MAX_FRAMES, len(index_df) - start_idx)
    print(f'   本次推理帧数: {num_frames} (从索引{start_idx}开始)')
    
    # 3. 批量推理
    print('\n3. 开始批量推理...')
    results = []
    
    for idx in tqdm(range(start_idx, start_idx + num_frames), desc='推理进度'):
        row = index_df.iloc[idx]
        infer_idx = int(row['infer_idx'])
        
        try:
            # 运行推理
            result = run_inference(model, processor, row, DATA_DIR)
            
            # 保存轨迹预测
            np.save(f'{RESULT_DIR}/pred_{infer_idx:06d}.npy', result['pred_xyz'])
            
            # 绘制轨迹图
            plot_trajectories(result['pred_xyz'], result['gt_xy'], infer_idx, RESULT_DIR)
            
            # 记录结果
            results.append({
                'infer_idx': infer_idx,
                't0_timestamp': row['t0_timestamp'],
                'min_ade': float(result['min_ade']),
                'best_traj_idx': int(result['best_traj_idx']),
                'ade_all': json.dumps([float(x) for x in result['ade_all']]),
                'coc_text': json.dumps(result['coc_texts']),
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
    
    # 4. 保存结果CSV
    print('\n4. 保存结果...')
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
    
    print(f'\n结果保存至: {RESULT_DIR}/')
    print('  - inference_results.csv (CoC + minADE)')
    print('  - pred_*.npy (6条轨迹预测)')
    print('  - trajectory_*.png (轨迹可视化)')

if __name__ == '__main__':
    main()
