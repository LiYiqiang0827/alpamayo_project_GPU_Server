#!/usr/bin/env python3
"""
post_video_gen.py - Alpamayo 1.5 推理结果视频生成工具
直接渲染轨迹图+相机画面→MP4，无中间文件
"""

import os
import sys
import argparse
import multiprocessing as mp
from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import av
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# ============ 全局配置 ============
INFER_RESULT_BASE = "/data01/mikelee/infer_result"
DATA_BASE = "/data01/mikelee/data/"
OUTPUT_BASE = "/data01/mikelee/infer_result"

X_MIN, X_MAX = 0.0, 60.0   # Forward axis (固定)

# ============ 工具函数 ============

def world_to_local(history: np.ndarray, future: np.ndarray) -> np.ndarray:
    t0_xyz = history[-1, 6:9]
    t0_rot = R.from_quat(history[-1, 3:7])
    t0_rot_inv = t0_rot.inv()
    future_world = future[:, 1:4]
    future_local = t0_rot_inv.apply(future_world - t0_xyz)
    return future_local[:, :2]


def get_pred_traj_path(infer_result_dir: str, chunk_id: int, clip_name: str, frame_number: int) -> str:
    return os.path.join(infer_result_dir, "pred_traj",
                        f"chunk{chunk_id:04d}_{clip_name}_{frame_number:06d}_predtraj.npy")


def get_data_paths(data_base: str, chunk_id: int, clip_name: str, frame_number: int):
    base = os.path.join(data_base, f"data_sample_chunk{chunk_id}", "infer", clip_name, "data")
    history_path = os.path.join(base, f"frame_{frame_number:06d}_history.npy")
    future_path = os.path.join(base, f"frame_{frame_number:06d}_future_gt.npy")
    return history_path, future_path, base


def compute_global_lateral_range(infer_result_dir: str, df: pd.DataFrame, sample_n: int = 30) -> tuple:
    """采样估算全局lateral范围，返回 (lat_min, lat_max)"""
    from scipy.spatial.transform import Rotation as R

    def _world_to_local(h, f):
        t0_xyz = h[-1, 6:9]
        t0_rot = R.from_quat(h[-1, 3:7])
        t0_rot_inv = t0_rot.inv()
        fw = f[:, 1:4]
        return t0_rot_inv.apply(fw - t0_xyz)[:, :2]

    clips = df.groupby(['chunk_id', 'clip_name']).size().reset_index()
    if len(clips) > sample_n:
        clips = clips.sample(sample_n, random_state=42)

    lat_mins, lat_maxs = [], []
    for _, row in clips.iterrows():
        chunk_id, clip_name = int(row['chunk_id']), row['clip_name']
        frame_number = int(df[(df['chunk_id']==chunk_id) & (df['clip_name']==clip_name)].iloc[0]['frame_number'])
        pred_path = get_pred_traj_path(infer_result_dir, chunk_id, clip_name, frame_number)
        hp, fp, _ = get_data_paths(DATA_BASE, chunk_id, clip_name, frame_number)
        if not os.path.exists(pred_path) or not os.path.exists(hp):
            continue
        try:
            pred = np.load(pred_path)
            if pred.ndim == 3:
                pred = pred[0]
            h, f = np.load(hp), np.load(fp)
            gt = _world_to_local(h, f)
            gt = np.vstack([np.array([[0, 0]]), gt])[:64]
            all_lat = np.concatenate([gt[:, 1].ravel(), pred[:, 1].ravel()])
            lat_mins.append(all_lat.min())
            lat_maxs.append(all_lat.max())
        except Exception:
            continue

    if not lat_mins:
        return -15.0, 70.0
    # 用5th和95th percentile，避免极端值
    global_min = np.percentile(lat_mins, 10)
    global_max = np.percentile(lat_maxs, 90)
    margin = (global_max - global_min) * 0.1
    return float(global_min - margin), float(global_max + margin)


def create_trajectory_plot(pred_xy: np.ndarray, gt_xy: np.ndarray,
                           frame_id: int, lat_min: float, lat_max: float) -> Image.Image:
    """渲染单帧轨迹图，返回 PIL Image"""
    if pred_xy.ndim > 2:
        pred_xy = pred_xy.reshape(-1, 2)
    if gt_xy.ndim > 2:
        gt_xy = gt_xy.reshape(-1, 2)

    fig, ax = plt.subplots(figsize=(5.4, 10.8), dpi=100)

    ax.plot(gt_xy[:, 1], gt_xy[:, 0], "r-", label="GT", linewidth=2.5)
    ax.plot(pred_xy[:, 1], pred_xy[:, 0], "-", color="#00BCD4",
            label="Pred", linewidth=1.5)
    ax.plot(0, 0, "k*", markersize=15, label="Ego (t0)")

    ax.set_xlim(lat_min, lat_max)
    ax.set_ylim(X_MIN, X_MAX)
    ax.set_xlabel("Lateral (m) - Right", fontsize=10)
    ax.set_ylabel("Forward (m)", fontsize=10)
    ax.set_title(f"Frame {frame_id:06d}", fontsize=12)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.tight_layout()

    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    img_array = img_array[:, :, :3]
    plt.close(fig)
    return Image.fromarray(img_array)


def process_single_clip(args, infer_result_dir: str, output_dir: str, lat_min: float, lat_max: float):
    """处理单个clip，生成视频"""
    chunk_id, clip_name = args
    df = pd.read_csv(f"{infer_result_dir}/infer_results_all.csv")
    frames_df = df[(df['chunk_id'] == chunk_id) & (df['clip_name'] == clip_name)].sort_values('frame_number')
    if frames_df.empty:
        return {'clip_key': f'chunk{chunk_id:04d}_{clip_name}', 'num_frames': 0, 'status': 'empty'}

    video_path = f"{output_dir}/chunk{chunk_id:04d}_{clip_name}_combined_video.mp4"
    index_dict = df.set_index('frame_number')

    try:
        container = av.open(video_path, mode='w')
        stream = container.add_stream('libx264', rate=10)
        stream.width = 1280
        stream.height = 1800
        stream.pix_fmt = 'yuv420p'
        stream.options = {'crf': '23', 'preset': 'medium'}

        for _, row in frames_df.iterrows():
            frame_number = int(row['frame_number'])
            image_path = row['image_path']

            # 加载并检查图像
            if not os.path.exists(image_path):
                img = Image.new('RGB', (1280, 1280), (50, 50, 50))
            else:
                img = Image.open(image_path).convert('RGB')
                if img.size != (1280, 1280):
                    img = img.resize((1280, 1280), Image.LANCZOS)

            # 计算轨迹
            pred_path = get_pred_traj_path(infer_result_dir, chunk_id, clip_name, frame_number)
            hp, fp, _ = get_data_paths(DATA_BASE, chunk_id, clip_name, frame_number)

            if os.path.exists(pred_path) and os.path.exists(hp) and os.path.exists(fp):
                try:
                    pred = np.load(pred_path)
                    if pred.ndim == 3:
                        pred = pred[0]
                    h, f = np.load(hp), np.load(fp)
                    gt = world_to_local(h, f)
                    gt = np.vstack([np.array([[0, 0]]), gt])[:64]
                    traj_img = create_trajectory_plot(pred, gt, frame_number, lat_min, lat_max)
                except Exception:
                    traj_img = Image.new('RGB', (540, 1080), (40, 40, 40))
            else:
                traj_img = Image.new('RGB', (540, 1080), (40, 40, 40))

            # 16宫格 → 540x1080
            grid_img = img.resize((540, 1080), Image.LANCZOS)

            # Info面板
            info = Image.new('RGB', (740, 40), (40, 40, 40))
            draw = ImageDraw.Draw(info)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
            except Exception:
                font = ImageFont.load_default()
            ade_val = index_dict.loc[frame_number, 'ade'] if frame_number in index_dict.index else 0.0
            draw.text((10, 10), f"ADE: {ade_val:.2f}m", font=font, fill=(200, 200, 200))

            # 拼接：左侧540x1080(grid) + 右侧540x1080(traj) + 底部740x40(info) + 底部右540x40(留空)
            top = Image.new('RGB', (540, 1080), (40, 40, 40))
            combined_w = 540 + 540
            combined_h = 1080
            combined = Image.new('RGB', (combined_w, combined_h), (30, 30, 30))
            combined.paste(grid_img, (0, 0))
            combined.paste(traj_img, (540, 0))

            # 底部info拼到右侧
            bottom_info = Image.new('RGB', (combined_w, 40), (40, 40, 40))
            bottom_info.paste(info, (540, 0))
            final_h = combined_h + 40
            final = Image.new('RGB', (combined_w, final_h), (30, 30, 30))
            final.paste(combined, (0, 0))
            final.paste(bottom_info, (0, combined_h))

            # 扩展到1280宽度
            final_full = Image.new('RGB', (1280, 1800), (30, 30, 30))
            offset_x = (1280 - combined_w) // 2
            final_full.paste(final, (offset_x, 0))

            arr = np.array(final_full.convert('RGB'), dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(arr, format='rgb24')
            for pkt in stream.encode(frame):
                container.mux(pkt)

        for pkt in stream.encode():
            container.mux(pkt)
        container.close()

        size_mb = os.path.getsize(video_path) / 1024 / 1024
        return {
            'clip_key': f'chunk{chunk_id:04d}_{clip_name}',
            'num_frames': len(frames_df),
            'video_size_mb': size_mb,
            'status': 'success'
        }
    except Exception as e:
        return {
            'clip_key': f'chunk{chunk_id:04d}_{clip_name}',
            'num_frames': 0,
            'video_size_mb': 0,
            'status': 'error',
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--infer_result', required=True)
    parser.add_argument('--parallel', type=int, default=30)
    parser.add_argument('--sample_n', type=int, default=30, help='全局坐标轴范围采样帧数')
    args = parser.parse_args()

    infer_result_dir = f"{INFER_RESULT_BASE}/{args.infer_result}"
    output_dir = f"{infer_result_dir}/combined_videos"
    os.makedirs(output_dir, exist_ok=True)

    print(f"[1/5] 读取 {infer_result_dir}/infer_results_all.csv ...")
    df = pd.read_csv(f"{infer_result_dir}/infer_results_all.csv")
    total_frames = len(df)
    print(f"  总帧数: {total_frames}")

    print(f"[2/5] 按 (chunk, clip) 分组 ...")
    clip_groups = df.groupby(['chunk_id', 'clip_name']).size().reset_index()[['chunk_id', 'clip_name']]
    clip_list = list(clip_groups.itertuples(index=False, name=None))
    print(f"  Clip 数: {len(clip_list)}")

    print(f"[3/5] 估算全局 lateral 坐标范围 (采样{args.sample_n}个clip) ...")
    lat_min, lat_max = compute_global_lateral_range(infer_result_dir, df, sample_n=args.sample_n)
    print(f"  Global lateral 范围: {lat_min:.1f} ~ {lat_max:.1f} m")

    print(f"[4/5] 分配 clips 到 {args.parallel} 个进程 ...")
    with mp.Pool(args.parallel) as pool:
        worker = partial(process_single_clip,
                         infer_result_dir=infer_result_dir,
                         output_dir=output_dir,
                         lat_min=lat_min,
                         lat_max=lat_max)
        results = []
        for r in pool.imap(worker, clip_list):
            results.append(r)
            done = len(results)
            if done % 30 == 0 or done == len(clip_list):
                print(f"  进度: {done}/{len(clip_list)}")

    print(f"[5/5] 汇总结果 ...")
    success = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] == 'error')
    empty_clips = sum(1 for r in results if r['status'] == 'empty')
    total_video_frames = sum(r['num_frames'] for r in results)
    total_size = sum(r.get('video_size_mb', 0) for r in results)

    print(f"\n============================================================")
    print(f"处理完成!")
    print(f"  Clip 总数: {len(clip_list)}")
    print(f"  成功: {success}, 失败: {failed}, 空: {empty_clips}")
    print(f"  总帧数: {total_video_frames}")
    print(f"  视频总大小: {total_size:.2f} GB")
    print(f"  输出目录: {output_dir}/")
    print(f"============================================================")


if __name__ == '__main__':
    main()
