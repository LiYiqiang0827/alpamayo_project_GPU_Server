#!/usr/bin/env python3
"""
mADE 计算脚本 - Alpamayo 1.5 推理后处理

用法:
    python post_made_cal.py --infer_result infer_result_20260424_010530 --parallel 30
    python post_made_cal.py --infer_result /data01/mikelee/infer_result/infer_result_20260424_010530 --parallel 30

输入:
    推理结果目录 (infer_result_XXXXXXXX_XXXXXX 格式)
    自动扩展为 /data01/mikelee/infer_result/infer_result_XXXXXXXX_XXXXXX

输出:
    post_made_all.csv — 包含原始 infer_results_all.csv 所有列 + mADE 列
"""

import os
import sys
import argparse
import glob
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp


# ===================== 配置 =====================
INFER_RESULT_BASE = "/data01/mikelee/infer_result"
DATA_BASE = "/data01/mikelee/data"
CAMERA_ORDER = [
    'camera_cross_left_120fov',
    'camera_front_wide_120fov',
    'camera_cross_right_120fov',
    'camera_front_tele_30fov',
]


# ===================== 路径构建 =====================

def resolve_infer_result_path(infer_result_arg: str) -> str:
    """将推理结果参数扩展为完整路径"""
    if infer_result_arg.startswith("/"):
        return infer_result_arg
    if infer_result_arg.startswith("infer_result_"):
        return os.path.join(INFER_RESULT_BASE, infer_result_arg)
    # 可能是裸名称
    return os.path.join(INFER_RESULT_BASE, f"infer_result_{infer_result_arg}")


def get_pred_traj_path(infer_result_dir: str, chunk_id: int, clip_name: str, frame_number: int) -> str:
    """构建 pred_traj.npy 路径"""
    return os.path.join(
        infer_result_dir, "pred_traj",
        f"chunk{chunk_id:04d}_{clip_name}_{frame_number:06d}_predtraj.npy"
    )


def get_gt_paths(data_base: str, chunk_id: int, clip_name: str, frame_number: int):
    """构建历史和真值轨迹文件路径"""
    base = os.path.join(data_base, f"data_sample_chunk{chunk_id}", "infer", clip_name, "data", "egomotion")
    history_path = os.path.join(base, f"frame_{frame_number:06d}_history.npy")
    future_path = os.path.join(base, f"frame_{frame_number:06d}_future_gt.npy")
    return history_path, future_path


# ===================== 坐标转换 =====================

def world_to_local(history: np.ndarray, future: np.ndarray) -> np.ndarray:
    """
    将真值轨迹从世界坐标系转换到局部坐标系（t0 车身参考系）

    Args:
        history: (16, 11) 世界坐标历史轨迹 [timestamp, qx, qy, qz, qw, x, y, z, vx, vy, vz]
        future:  (64, 4) 世界坐标未来轨迹 [timestamp, x, y, z]

    Returns:
        gt_xy: (64, 2) 局部坐标系下的真值 xy
    """
    # t0 时刻的位置（历史最后一个点）
    t0_xyz = history[-1, 5:8]       # (x, y, z)
    t0_quat = history[-1, 1:5]      # (qx, qy, qz, qw)

    # 构建旋转：从世界坐标转换到 t0 局部坐标
    t0_rot = R.from_quat(t0_quat)
    t0_rot_inv = t0_rot.inv()

    # 未来真值：世界坐标 → 局部坐标
    future_xyz_world = future[:, 1:4]                          # (64, 3)
    future_xyz_local = t0_rot_inv.apply(future_xyz_world - t0_xyz)  # (64, 3)
    gt_xy = future_xyz_local[:, :2]                            # (64, 2)

    return gt_xy


# ===================== mADE 计算 =====================

def compute_ade(pred_xy: np.ndarray, gt_xy: np.ndarray) -> float:
    """
    计算单帧 ADE

    Args:
        pred_xy: (64, 2) 预测轨迹局部坐标
        gt_xy:   (64, 2) 真值轨迹局部坐标

    Returns:
        ADE (米)
    """
    diff = np.linalg.norm(pred_xy - gt_xy, axis=1)   # (64,)
    ade = diff.mean()
    return float(ade)


def process_single_frame(args: tuple) -> Dict[str, Any]:
    """
    处理单个帧的 mADE 计算

    Returns:
        dict with 'chunk_id', 'clip_name', 'frame_number', 'ade', 'error'
    """
    chunk_id, clip_name, frame_number, infer_result_dir, data_base = args

    try:
        # 路径
        pred_path = get_pred_traj_path(infer_result_dir, chunk_id, clip_name, frame_number)
        history_path, future_path = get_gt_paths(data_base, chunk_id, clip_name, frame_number)

        # 检查文件存在
        if not os.path.exists(pred_path):
            return {
                'chunk_id': chunk_id, 'clip_name': clip_name, 'frame_number': frame_number,
                'ade': np.nan, 'error': f"pred_traj not found: {pred_path}"
            }
        if not os.path.exists(history_path):
            return {
                'chunk_id': chunk_id, 'clip_name': clip_name, 'frame_number': frame_number,
                'ade': np.nan, 'error': f"history not found: {history_path}"
            }
        if not os.path.exists(future_path):
            return {
                'chunk_id': chunk_id, 'clip_name': clip_name, 'frame_number': frame_number,
                'ade': np.nan, 'error': f"future_gt not found: {future_path}"
            }

        # 加载数据
        pred_xy = np.load(pred_path)          # (64, 2)
        history = np.load(history_path)       # (16, 11)
        future = np.load(future_path)         # (64, 4)

        # 坐标转换
        gt_xy = world_to_local(history, future)   # (64, 2)

        # 计算 ADE
        ade = compute_ade(pred_xy, gt_xy)

        return {
            'chunk_id': chunk_id, 'clip_name': clip_name, 'frame_number': frame_number,
            'ade': ade, 'error': None
        }

    except Exception as e:
        return {
            'chunk_id': chunk_id, 'clip_name': clip_name, 'frame_number': frame_number,
            'ade': np.nan, 'error': str(e)
        }


def process_frames_parallel(frames_df: pd.DataFrame, infer_result_dir: str,
                            data_base: str, parallel: int, chunk_id: int = None) -> List[Dict]:
    """
    多进程处理帧列表

    Args:
        frames_df: 包含 chunk_id, clip_name, frame_number 列的 DataFrame
        infer_result_dir: 推理结果根目录
        data_base: 数据根目录
        parallel: 并行进程数
        chunk_id: 当前处理的 chunk_id（用于进度显示）

    Returns:
        结果字典列表
    """
    # 构建任务参数
    args_list = [
        (int(row['chunk_id']), str(row['clip_name']), int(row['frame_number']), infer_result_dir, data_base)
        for _, row in frames_df.iterrows()
    ]

    results = []
    with ProcessPoolExecutor(max_workers=parallel) as executor:
        futures = {executor.submit(process_single_frame, args): args for args in args_list}

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

            # 进度显示（每100个更新一次）
            if len(results) % 500 == 0 or len(results) == len(args_list):
                print(f"  [Chunk {'?' if chunk_id is None else chunk_id}] Progress: {len(results)}/{len(args_list)}")

    return results


# ===================== 主流程 =====================

def main():
    parser = argparse.ArgumentParser(description='mADE 计算脚本 - Alpamayo 1.5 推理后处理')
    parser.add_argument('--infer_result', type=str, required=True,
                        help='推理结果名称或路径, 如 infer_result_20260424_010530')
    parser.add_argument('--parallel', type=int, default=30,
                        help='并行进程数 (默认 30)')
    parser.add_argument('--stat_enable', type=bool, default=True,
                        help='计算完成后自动调用 post_made_stat.py 进行统计 (默认 True)')
    args = parser.parse_args()

    # 解析路径
    infer_result_dir = resolve_infer_result_path(args.infer_result)
    if not os.path.exists(infer_result_dir):
        print(f"错误: 推理结果目录不存在: {infer_result_dir}")
        sys.exit(1)

    csv_path = os.path.join(infer_result_dir, "infer_results_all.csv")
    if not os.path.exists(csv_path):
        print(f"错误: infer_results_all.csv 不存在: {csv_path}")
        sys.exit(1)

    output_csv = os.path.join(infer_result_dir, "post_made_all.csv")

    print(f"{'=' * 60}")
    print(f"mADE 计算 - Alpamayo 1.5 推理后处理")
    print(f"{'=' * 60}")
    print(f"推理结果目录: {infer_result_dir}")
    print(f"进程数: {args.parallel}")
    print(f"输出文件: {output_csv}")
    print()

    # Step 1: 读取 infer_results_all.csv
    print(f"[1/4] 读取 infer_results_all.csv ...")
    df = pd.read_csv(csv_path)
    total_rows = len(df)
    print(f"  总帧数: {total_rows}")

    # Step 2: 按 chunk_id 分组，多进程处理
    print(f"[2/4] 计算 mADE (并行 {args.parallel} 进程) ...")
    all_results = []

    # 按 chunk_id 分组，每组单独多进程处理
    if 'chunk_id' in df.columns:
        chunk_ids = df['chunk_id'].unique()
        print(f"  Chunk 数: {len(chunk_ids)}")

        for cid in sorted(chunk_ids):
            chunk_df = df[df['chunk_id'] == cid]
            print(f"\n  处理 Chunk {cid}: {len(chunk_df)} 帧")
            chunk_results = process_frames_parallel(
                chunk_df, infer_result_dir, DATA_BASE, args.parallel, chunk_id=cid
            )
            all_results.extend(chunk_results)
    else:
        # 没有 chunk_id 列则全局处理
        all_results = process_frames_parallel(df, infer_result_dir, DATA_BASE, args.parallel)

    # 构建结果 DataFrame
    results_df = pd.DataFrame(all_results)

    # Step 3: 合并结果
    print(f"\n[3/4] 合并结果 ...")
    # 确保按相同顺序合并
    results_df = results_df.sort_values(
        by=['chunk_id', 'clip_name', 'frame_number']
    ).reset_index(drop=True)

    # 将 ade/error 合并回原 DataFrame
    # 构建一个 lookup 字典用于合并
    ade_lookup = {
        (row['chunk_id'], row['clip_name'], row['frame_number']): row['ade']
        for _, row in results_df.iterrows()
    }
    error_lookup = {
        (row['chunk_id'], row['clip_name'], row['frame_number']): row['error']
        for _, row in results_df.iterrows()
    }

    df = df.sort_values(by=['chunk_id', 'clip_name', 'frame_number']).reset_index(drop=True)
    df['mADE'] = df.apply(
        lambda r: ade_lookup.get((r['chunk_id'], r['clip_name'], r['frame_number']), np.nan),
        axis=1
    )
    df['error'] = df.apply(
        lambda r: error_lookup.get((r['chunk_id'], r['clip_name'], r['frame_number']), 'unknown'),
        axis=1
    )

    # Step 4: 保存结果
    print(f"[4/4] 保存结果到 {output_csv} ...")
    df.to_csv(output_csv, index=False)

    # 统计输出
    valid_ades = df['mADE'].dropna()
    print(f"\n{'=' * 60}")
    print(f"计算完成!")
    print(f"  总帧数: {total_rows}")
    print(f"  有效帧数: {len(valid_ades)}")
    print(f"  失败帧数: {total_rows - len(valid_ades)}")
    if len(valid_ades) > 0:
        print(f"\n  mADE 统计 (米):")
        print(f"    Mean:  {valid_ades.mean():.4f}")
        print(f"    Median: {valid_ades.median():.4f}")
        print(f"    Min:   {valid_ades.min():.4f}")
        print(f"    Max:   {valid_ades.max():.4f}")
        print(f"    Std:   {valid_ades.std():.4f}")
    print(f"{'=' * 60}")

    # Step 5: 自动调用 post_made_stat.py（如果启用）
    if args.stat_enable:
        print(f"\n[5/5] 自动调用 post_made_stat.py 进行统计 ...")
        stat_script = os.path.join(os.path.dirname(__file__), "post_made_stat.py")
        if os.path.exists(stat_script):
            import subprocess
            cmd = [
                sys.executable, stat_script,
                '--infer_result', infer_result_dir
            ]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                print(result.stdout)
                if result.returncode != 0:
                    print(f"警告: post_made_stat.py 执行失败: {result.stderr}")
            except Exception as e:
                print(f"警告: 调用 post_made_stat.py 时出错: {e}")
        else:
            print(f"警告: 未找到 post_made_stat.py: {stat_script}")


if __name__ == '__main__':
    main()
