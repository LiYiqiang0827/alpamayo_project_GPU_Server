#!/usr/bin/env python3
"""
post_video_gen.py - 推理结果批量视频生成工具

从 infer_results_all.csv 读取推理结果，按 (chunk_id, clip_name) 分组，
渲染每帧的组合画面并直接编码为 MP4 视频。

用法:
    python3 post_video_gen.py --infer_result infer_result_20260424_010530
    python3 post_video_gen.py --infer_result /data01/mikelee/infer_result/infer_result_20260424_010530 --parallel 30
"""

import os
import sys
import json
import argparse
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import av
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
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

# 视频编码参数
VIDEO_FPS = 10
VIDEO_CRF = 23
VIDEO_PRESET = 'medium'

# 绘图参数
X_MIN, X_MAX = -15, 15
Y_MIN, Y_MAX = 0, 60
# 全局横向(lateral)坐标范围 - 所有帧共用，保证坐标轴一致性
# 基于数据采样估算: 0~59m lateral范围，取5th~95th percentile + 5% margin
GLOBAL_LAT_MIN, GLOBAL_LAT_MAX = -5.0, 70.0
GRID_W, GRID_H = 320, 180
INFO_W, BOTTOM_H = 740, 1080
TRAJ_W = 540
CANVAS_W = 1280
CANVAS_GRID_H = 720
CANVAS_TOTAL_H = 1800


# ===================== 路径解析 =====================

def resolve_infer_result_path(infer_result_arg: str) -> str:
    if infer_result_arg.startswith("/"):
        return infer_result_arg
    if infer_result_arg.startswith("infer_result_"):
        return os.path.join(INFER_RESULT_BASE, infer_result_arg)
    return os.path.join(INFER_RESULT_BASE, f"infer_result_{infer_result_arg}")


def get_pred_traj_path(infer_result_dir: str, chunk_id: int, clip_name: str, frame_number: int) -> str:
    return os.path.join(
        infer_result_dir, "pred_traj",
        f"chunk{chunk_id:04d}_{clip_name}_{frame_number:06d}_predtraj.npy"
    )


def get_data_paths(data_base: str, chunk_id: int, clip_name: str, frame_number: int):
    base = os.path.join(data_base, f"data_sample_chunk{chunk_id}", "infer", clip_name, "data")
    ego_dir = os.path.join(base, "egomotion")
    history_path = os.path.join(ego_dir, f"frame_{frame_number:06d}_history.npy")
    future_path = os.path.join(ego_dir, f"frame_{frame_number:06d}_future_gt.npy")
    index_path = os.path.join(base, "inference_index_strict.csv")
    return history_path, future_path, index_path


# ===================== 坐标转换 =====================

def world_to_local(history: np.ndarray, future: np.ndarray) -> np.ndarray:
    t0_xyz = history[-1, 5:8]
    t0_quat = history[-1, 1:5]
    t0_rot = R.from_quat(t0_quat)
    t0_rot_inv = t0_rot.inv()
    future_xyz_world = future[:, 1:4]
    future_xyz_local = t0_rot_inv.apply(future_xyz_world - t0_xyz)
    return future_xyz_local[:, :2]


def compute_ade(pred_xy: np.ndarray, gt_xy: np.ndarray) -> float:
    diff = np.linalg.norm(pred_xy - gt_xy, axis=1)
    return float(diff.mean())


# ===================== 帧渲染 =====================

def load_font(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except:
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
        except:
            return ImageFont.load_default()


def load_images_for_frame(index_row: pd.Series, data_dir: str) -> List[Image.Image]:
    images = []
    for cam in CAMERA_ORDER:
        for t in range(4):
            idx_col = f'{cam}_f{t}_idx'
            frame_idx = int(index_row[idx_col])
            img_path = os.path.join(data_dir, "camera_images", cam, f"{frame_idx:06d}_small.jpg")
            if not os.path.exists(img_path):
                img_path = os.path.join(data_dir, "camera_images", cam, f"{frame_idx:06d}.jpg")
            try:
                img = Image.open(img_path).convert('RGB')
            except:
                img = Image.new('RGB', (GRID_W, GRID_H), color=(128, 128, 128))
            img = img.resize((GRID_W, GRID_H))
            images.append(img)
    return images


def create_16_grid(images: List[Image.Image]) -> Image.Image:
    grid_w, grid_h = GRID_W * 4, GRID_H * 4
    grid = Image.new('RGB', (grid_w, grid_h), color=(0, 0, 0))
    idx = 0
    for cam_idx in range(4):
        for time_idx in range(4):
            grid.paste(images[idx], (time_idx * GRID_W, cam_idx * GRID_H))
            idx += 1
    return grid


def create_trajectory_plot(pred_xy: np.ndarray, gt_xy: np.ndarray, frame_id: int) -> Image.Image:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # 使用全局固定横向坐标范围 (所有帧统一，保证坐标轴一致)
    if pred_xy.ndim > 2:
        pred_xy = pred_xy.reshape(-1, 2)
    if gt_xy.ndim > 2:
        gt_xy = gt_xy.reshape(-1, 2)
    lat_y_min = GLOBAL_LAT_MIN
    lat_y_max = GLOBAL_LAT_MAX
    # Forward axis fixed 0~60m
    x_min, x_max = 0.0, 60.0

    fig, ax = plt.subplots(figsize=(5.4, 10.8), dpi=100)

    ax.plot(gt_xy[:, 1], gt_xy[:, 0], "r-", label="GT", linewidth=2.5)
    ax.plot(pred_xy[:, 1], pred_xy[:, 0], "-", color="#00BCD4",
            label="Pred", linewidth=1.5)
    ax.plot(0, 0, "k*", markersize=15, label="Ego (t0)")

    ax.set_xlim(lat_y_min, lat_y_max)
    ax.set_ylim(x_min, x_max)
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
    traj_img = Image.fromarray(img_array)
    plt.close(fig)
    return traj_img


def wrap_text(draw: ImageDraw.Draw, text: str, max_w: int, font: ImageFont.FreeTypeFont) -> List[str]:
    if not text:
        return []
    lines, current = [], ""
    for word in str(text).split():
        test = current + " " + word if current else word
        try:
            bbox = draw.textbbox((0, 0), test, font=font)
            if bbox[2] <= max_w:
                current = test
            else:
                if current:
                    lines.append(current)
                current = word
        except:
            if len(test) * 10 <= max_w:
                current = test
            else:
                if current:
                    lines.append(current)
                current = word
    if current:
        lines.append(current)
    return lines[:10]


def create_info_panel(frame_id: int, ade: float, inference_time_ms: float,
                      cot_text: str, width: int, height: int) -> Image.Image:
    panel = Image.new('RGB', (width, height), color=(40, 40, 40))
    draw = ImageDraw.Draw(panel)

    font_large = load_font(28)
    font_medium = load_font(22)
    font_small = load_font(18)

    cot_lines = wrap_text(draw, cot_text, width - 40, font_small)

    y = 30
    draw.text((20, y), f"Frame: {frame_id}", fill=(255, 255, 0), font=font_large); y += 50
    draw.text((20, y), f"ADE: {ade:.3f} m", fill=(0, 255, 0), font=font_medium); y += 45
    draw.text((20, y), f"Time: {inference_time_ms:.0f} ms", fill=(200, 200, 200), font=font_medium); y += 60
    draw.text((20, y), "CoC Reasoning:", fill=(100, 200, 255), font=font_medium); y += 40

    for line in cot_lines:
        draw.text((20, y), line, fill=(255, 255, 255), font=font_small); y += 28

    return panel


def render_combined_frame(frame_id: int, ade: float, inference_time_ms: float,
                          cot_text: str, images: List[Image.Image],
                          pred_xy: np.ndarray, gt_xy: np.ndarray) -> Image.Image:
    grid = create_16_grid(images)
    traj_img = create_trajectory_plot(pred_xy, gt_xy, frame_id)
    traj_img = traj_img.resize((TRAJ_W, BOTTOM_H))
    info_panel = create_info_panel(frame_id, ade, inference_time_ms, cot_text, INFO_W, BOTTOM_H)

    bottom = Image.new('RGB', (CANVAS_W, BOTTOM_H))
    bottom.paste(info_panel, (0, 0))
    bottom.paste(traj_img, (INFO_W, 0))

    frame = Image.new('RGB', (CANVAS_W, CANVAS_TOTAL_H))
    frame.paste(grid, (0, 0))
    frame.paste(bottom, (0, CANVAS_GRID_H))

    return frame


# ===================== 视频写入 =====================

def create_video_writer(output_path: str, fps: int):
    container = av.open(output_path, mode='w')
    stream = container.add_stream('libx264', rate=fps)
    stream.width = CANVAS_W
    stream.height = CANVAS_TOTAL_H
    stream.pix_fmt = 'yuv420p'
    stream.options = {'crf': str(VIDEO_CRF), 'preset': VIDEO_PRESET}
    return container, stream


def write_video_frame(container, stream, frame_img: Image.Image):
    img_array = np.array(frame_img.convert('RGB'))
    av_frame = av.VideoFrame.from_ndarray(img_array, format='rgb24')
    for packet in stream.encode(av_frame):
        container.mux(packet)


def finish_video(container, stream):
    for packet in stream.encode():
        container.mux(packet)
    container.close()


# ===================== Clip 分组与分配 =====================

def group_clips(df: pd.DataFrame) -> Dict[Tuple[int, str], pd.DataFrame]:
    groups = {}
    for (chunk_id, clip_name), sub_df in df.groupby(['chunk_id', 'clip_name']):
        sub_df = sub_df.sort_values('frame_number').reset_index(drop=True)
        groups[(chunk_id, clip_name)] = sub_df
    return groups


def assign_clips_to_workers(clip_groups: Dict[Tuple[int, str], pd.DataFrame],
                            num_workers: int) -> List[List[Tuple[int, str]]]:
    clip_keys = sorted(clip_groups.keys())
    assignments = [[] for _ in range(num_workers)]
    for i, key in enumerate(clip_keys):
        assignments[i % num_workers].append(key)
    return assignments


# ===================== 单 Clip 处理 =====================

def process_single_clip(args: Tuple) -> Dict:
    (chunk_id, clip_name), frames_df, infer_result_dir, data_base, index_cache = args

    clip_key_str = f"chunk{chunk_id:04d}_{clip_name}"
    output_dir = os.path.join(infer_result_dir, "combined_videos")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{clip_key_str}_combined_video.mp4")

    try:
        # 预加载该 clip 的索引表
        if (chunk_id, clip_name) not in index_cache:
            _, _, index_path = get_data_paths(data_base, chunk_id, clip_name, 0)
            if not os.path.exists(index_path):
                return {'clip_key': clip_key_str, 'num_frames': 0,
                        'status': 'error', 'error': f'index not found: {index_path}'}
            index_df = pd.read_csv(index_path)
            index_cache[(chunk_id, clip_name)] = index_df
        else:
            index_df = index_cache[(chunk_id, clip_name)]

        index_dict = index_df.set_index('frame_id')
        cot_dict = frames_df.set_index('frame_number')['cot_result'].to_dict()

        container, stream = create_video_writer(output_path, VIDEO_FPS)

        num_frames = 0
        for _, row in frames_df.iterrows():
            frame_number = int(row['frame_number'])
            cot_result = str(row['cot_result']) if pd.notna(row['cot_result']) else ''

            try:
                # 1. 预测轨迹
                pred_path = get_pred_traj_path(infer_result_dir, chunk_id, clip_name, frame_number)
                if not os.path.exists(pred_path):
                    continue
                pred_xy = np.load(pred_path)
                if pred_xy.ndim == 2:
                    pred_xy = pred_xy[np.newaxis, :, :]
                if pred_xy.shape[0] > 1:
                    pred_xy = pred_xy[0]

                # 2. 历史和真值轨迹
                history_path, future_path, _ = get_data_paths(data_base, chunk_id, clip_name, frame_number)
                if not os.path.exists(history_path) or not os.path.exists(future_path):
                    continue
                history = np.load(history_path)
                future = np.load(future_path)
                gt_xy = world_to_local(history, future)
                gt_xy = np.vstack([np.array([[0, 0]]), gt_xy])
                gt_xy = gt_xy[:64]

                # 3. ADE
                ade = compute_ade(pred_xy, gt_xy)

                # 4. 16张图片
                if frame_number not in index_dict.index:
                    continue
                index_row = index_dict.loc[frame_number]
                img_data_dir = os.path.join(data_base, f"data_sample_chunk{chunk_id}", "infer", clip_name, "data")
                images = load_images_for_frame(index_row, img_data_dir)

                # 5. 渲染
                frame_img = render_combined_frame(
                    frame_number, ade, 0, cot_result,
                    images, pred_xy, gt_xy
                )

                # 6. 写入视频
                write_video_frame(container, stream, frame_img)
                num_frames += 1

            except Exception as e:
                import traceback
                traceback.print_exc()
                continue

        finish_video(container, stream)

        if num_frames > 0:
            video_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            return {
                'clip_key': clip_key_str,
                'num_frames': num_frames,
                'video_size_mb': video_size_mb,
                'status': 'success',
                'error': None
            }
        else:
            return {'clip_key': clip_key_str, 'num_frames': 0, 'status': 'empty', 'error': 'no frames processed'}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'clip_key': clip_key_str, 'num_frames': 0, 'status': 'error', 'error': str(e)}


# ===================== 主函数 =====================

def compute_global_lateral_range(df: pd.DataFrame, infer_result_dir: str, sample_n: int = 30):
    """采样估算全局lateral范围，设置全局变量"""
    global GLOBAL_LAT_MIN, GLOBAL_LAT_MAX

    clips = df.groupby(['chunk_id', 'clip_name']).size().reset_index()
    if len(clips) > sample_n:
        clips = clips.sample(sample_n, random_state=42)

    lat_mins, lat_maxs = [], []
    for _, row in clips.iterrows():
        chunk_id, clip_name = int(row['chunk_id']), row['clip_name']
        sub = df[(df['chunk_id']==chunk_id) & (df['clip_name']==clip_name)]
        if sub.empty:
            continue
        frame_number = int(sub.iloc[0]['frame_number'])
        pred_path = get_pred_traj_path(infer_result_dir, chunk_id, clip_name, frame_number)
        hp, fp, _ = get_data_paths(DATA_BASE, chunk_id, clip_name, frame_number)
        if not os.path.exists(pred_path) or not os.path.exists(hp):
            continue
        try:
            pred = np.load(pred_path)
            if pred.ndim == 3:
                pred = pred[0]
            h, f = np.load(hp), np.load(fp)
            gt = world_to_local(h, f)
            gt = np.vstack([np.array([[0, 0]]), gt])[:64]
            all_lat = np.concatenate([gt[:, 1].ravel(), pred[:, 1].ravel()])
            lat_mins.append(all_lat.min())
            lat_maxs.append(all_lat.max())
        except Exception:
            continue

    if lat_mins:
        g_min = np.percentile(lat_mins, 5)
        g_max = np.percentile(lat_maxs, 95)
        margin = (g_max - g_min) * 0.05
        GLOBAL_LAT_MIN, GLOBAL_LAT_MAX = float(g_min - margin), float(g_max + margin)
    else:
        GLOBAL_LAT_MIN, GLOBAL_LAT_MAX = -5.0, 70.0
    print(f"  [全局横向范围: {GLOBAL_LAT_MIN:.1f} ~ {GLOBAL_LAT_MAX:.1f} m]")


def main():
    parser = argparse.ArgumentParser(description='post_video_gen - 推理结果批量视频生成')
    parser.add_argument('--infer_result', type=str, required=True,
                        help='推理结果名称或路径, 如 infer_result_20260424_010530')
    parser.add_argument('--parallel', type=int, default=30,
                        help='并行进程数 (默认 30)')
    args = parser.parse_args()

    infer_result_dir = resolve_infer_result_path(args.infer_result)
    if not os.path.exists(infer_result_dir):
        print(f"错误: 推理结果目录不存在: {infer_result_dir}")
        sys.exit(1)

    csv_path = os.path.join(infer_result_dir, "infer_results_all.csv")
    if not os.path.exists(csv_path):
        print(f"错误: infer_results_all.csv 不存在: {csv_path}")
        sys.exit(1)

    print(f"{'=' * 60}")
    print(f"post_video_gen - 推理结果批量视频生成")
    print(f"{'=' * 60}")
    print(f"推理结果目录: {infer_result_dir}")
    print(f"进程数: {args.parallel}")
    print()

    # Step 1: 读取
    print(f"[1/5] 读取 infer_results_all.csv ...")
    df = pd.read_csv(csv_path)
    total_frames = len(df)
    print(f"  总帧数: {total_frames}")

    print(f"[2/5] 估算全局横向坐标范围 ...")
    compute_global_lateral_range(df, infer_result_dir, sample_n=30)

    # Step 2: 分组
    print(f"[3/5] 按 (chunk, clip) 分组 ...")
    clip_groups = group_clips(df)
    num_clips = len(clip_groups)
    print(f"  Clip 数: {num_clips}")

    # Step 3: 分配
    print(f"[4/5] 分配 clips 到 {args.parallel} 个进程 ...")
    clip_assignments = assign_clips_to_workers(clip_groups, args.parallel)
    active_workers = sum(1 for w in clip_assignments if len(w) > 0)
    print(f"  活跃 worker 数: {active_workers}")

    # 共享索引缓存 (进程内字典)
    index_cache: Dict[Tuple[int, str], pd.DataFrame] = {}

    worker_tasks = []
    for worker_id, clip_keys in enumerate(clip_assignments):
        if not clip_keys:
            continue
        tasks_for_worker = [
            (key, clip_groups[key], infer_result_dir, DATA_BASE, index_cache)
            for key in clip_keys
        ]
        worker_tasks.append((worker_id, tasks_for_worker))

    # Step 4: 多进程
    print(f"[5/5] 生成视频 ({args.parallel} 进程) ...")

    all_results = []
    with ProcessPoolExecutor(max_workers=args.parallel) as executor:
        futures = {}
        for worker_id, tasks in worker_tasks:
            for task in tasks:
                future = executor.submit(process_single_clip, task)
                futures[future] = task[0]

        for future in tqdm(as_completed(futures), total=len(futures), desc="生成视频"):
            result = future.result()
            all_results.append(result)

    # 汇总
    print(f"\n{'=' * 60}")
    success_count = sum(1 for r in all_results if r['status'] == 'success')
    error_count = sum(1 for r in all_results if r['status'] == 'error')
    empty_count = sum(1 for r in all_results if r['status'] == 'empty')
    total_video_frames = sum(r['num_frames'] for r in all_results)
    total_video_size_gb = sum(r.get('video_size_mb', 0) for r in all_results) / 1024

    print(f"处理完成!")
    print(f"  Clip 总数: {num_clips}")
    print(f"  成功: {success_count}, 失败: {error_count}, 空: {empty_count}")
    print(f"  总帧数: {total_video_frames}")
    print(f"  视频总大小: {total_video_size_gb:.2f} GB")
    print(f"  输出目录: {infer_result_dir}/combined_videos/")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
