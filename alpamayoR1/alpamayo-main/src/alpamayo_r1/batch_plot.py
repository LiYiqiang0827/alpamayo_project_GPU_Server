#!/usr/bin/env python3
"""
批量画图脚本 - 从推理结果生成轨迹图
多线程并行处理
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import json

# 配置
RESULT_DIR = '/data01/vla/data/data_sample_chunk0/infer/01d3588e-bca7-4a18-8e74-c6cfe9e996db/result'
DATA_DIR = '/data01/vla/data/data_sample_chunk0/infer/01d3588e-bca7-4a18-8e74-c6cfe9e996db/data'
PLOT_DIR = f'{RESULT_DIR}/plots'

# 坐标轴范围
X_MIN, X_MAX = -30, 30  # 左右 -30m ~ +30m
Y_MIN, Y_MAX = 0, 100   # 前方 0m ~ 100m

os.makedirs(PLOT_DIR, exist_ok=True)

def rotate_90cc(xy):
    """逆时针旋转90度: (x, y) -> (-y, x)"""
    return np.stack([-xy[1], xy[0]], axis=0)

def load_gt_xy(infer_idx):
    """从egomotion加载真值轨迹"""
    # 从CSV找到对应的ego_file_prefix
    csv_path = f'{RESULT_DIR}/continuous_inference_results.csv'
    df = pd.read_csv(csv_path)
    row = df[df['infer_idx'] == infer_idx].iloc[0]
    
    # 构造ego文件名
    ego_filename = f'ego_{infer_idx:06d}'
    future_path = f'{DATA_DIR}/egomotion/{ego_filename}_future_gt.npy'
    
    future = np.load(future_path, allow_pickle=True).item()
    gt_xy = future['xyz'][:, :2]  # (64, 2)
    return gt_xy

def plot_single_trajectory(args):
    """画单个轨迹图"""
    pred_file, plot_dir = args
    
    # 解析infer_idx
    infer_idx = int(pred_file.split('_')[1].split('.')[0])
    
    try:
        # 加载预测轨迹
        pred_path = f'{RESULT_DIR}/{pred_file}'
        pred_xyz = np.load(pred_path)  # (64, 2) for 1-traj or (N, 64, 2) for multi-traj
        
        # 处理维度
        if pred_xyz.ndim == 2:
            pred_xyz = pred_xyz[np.newaxis, :, :]  # (1, 64, 2)
        
        # 加载真值
        gt_xy = load_gt_xy(infer_idx)
        
        # 画图
        plt.figure(figsize=(10, 10))
        
        # 预测轨迹 (支持多条)
        colors = plt.cm.tab10(np.linspace(0, 1, pred_xyz.shape[0]))
        for i in range(pred_xyz.shape[0]):
            pred_xy = pred_xyz[i].T
            pred_xy_rot = rotate_90cc(pred_xy)
            plt.plot(pred_xy_rot[0], pred_xy_rot[1], "o-", 
                    color=colors[i], label=f"Predicted #{i+1}", markersize=2, linewidth=1.5)
        
        # 真值轨迹
        gt_xy_rot = rotate_90cc(gt_xy.T)
        plt.plot(gt_xy_rot[0], gt_xy_rot[1], "r-", label="Ground Truth", linewidth=2.5)
        
        # 固定坐标轴范围
        plt.xlim(X_MIN, X_MAX)
        plt.ylim(Y_MIN, Y_MAX)
        
        plt.xlabel("X (meters, left-right)", fontsize=12)
        plt.ylabel("Y (meters, forward)", fontsize=12)
        plt.title(f"Trajectory Prediction - Frame {infer_idx:06d}", fontsize=14)
        plt.legend(loc="upper left", fontsize=9)
        plt.grid(True, alpha=0.3)
        
        # 保存
        plt.savefig(f'{plot_dir}/trajectory_{infer_idx:06d}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        return infer_idx, True, None
        
    except Exception as e:
        plt.close()
        return infer_idx, False, str(e)

def main():
    print('=== 批量画图 ===\n')
    
    # 获取所有pred文件
    pred_files = sorted([f for f in os.listdir(RESULT_DIR) if f.startswith('pred_') and f.endswith('.npy')])
    print(f'找到 {len(pred_files)} 个预测文件')
    print(f'画图保存至: {PLOT_DIR}/')
    
    # 准备参数
    plot_args = [(f, PLOT_DIR) for f in pred_files]
    
    # 多线程批量画图
    print('\n开始画图 (多线程)...')
    success_count = 0
    failed = []
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(tqdm(
            executor.map(plot_single_trajectory, plot_args),
            total=len(plot_args),
            desc='画图进度'
        ))
    
    for infer_idx, success, error in results:
        if success:
            success_count += 1
        else:
            failed.append((infer_idx, error))
    
    print(f'\n完成: {success_count}/{len(pred_files)}')
    
    if failed:
        print(f'失败: {len(failed)}')
        for idx, err in failed[:5]:
            print(f'  Frame {idx}: {err}')
    
    print(f'\n图片保存至: {PLOT_DIR}/')
    
    # 统计文件大小
    plot_files = [f for f in os.listdir(PLOT_DIR) if f.endswith('.png')]
    total_size = sum(os.path.getsize(f'{PLOT_DIR}/{f}') for f in plot_files) / (1024*1024)
    print(f'总大小: {total_size:.1f} MB ({len(plot_files)} 张图)')

if __name__ == '__main__':
    main()
