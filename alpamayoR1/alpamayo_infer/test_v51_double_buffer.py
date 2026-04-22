#!/usr/bin/env python3
"""
测试V5.1双缓冲 - 单Clip版本
"""

import os

os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys

sys.path.insert(0, "/home/user/mikelee/alpamayo-main/src")

import argparse
import time
import json
import traceback
import multiprocessing
import threading
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from queue import Queue
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from einops import rearrange
from scipy.spatial.transform import Rotation as R

MODEL_PATH = "/data01/vla/models--nvidia--Alpamayo-R1-10B/snapshots/22fab1399111f50b52bfbe5d8b809f39bd4c2fe1"
CAMERA_ORDER = [
    "camera_cross_left_120fov",
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_front_tele_30fov",
]


class DoubleBufferInference:
    """双缓冲推理器"""

    def __init__(self, batch_size, device="cuda"):
        self.batch_size = batch_size
        self.device = device

        # 创建两个CUDA流
        self.compute_stream = torch.cuda.Stream(device=device)
        self.transfer_stream = torch.cuda.Stream(device=device)

        # 双缓冲
        self.buffer_ping = self._allocate_buffer()
        self.buffer_pong = self._allocate_buffer()
        self.cpu_buffer = self.buffer_ping
        self.gpu_buffer = self.buffer_pong

    def _allocate_buffer(self):
        """预分配GPU输入内存"""
        return {
            "images": torch.empty(
                self.batch_size,
                4,
                4,
                3,
                224,
                224,
                dtype=torch.float32,
                device=self.device,
            ),
            "hist_xyz": torch.empty(
                self.batch_size, 1, 1, 16, 3, dtype=torch.float32, device=self.device
            ),
            "hist_rot": torch.empty(
                self.batch_size, 1, 1, 16, 3, 3, dtype=torch.float32, device=self.device
            ),
            "future_xyz": torch.empty(
                self.batch_size, 1, 1, 64, 3, dtype=torch.float32, device=self.device
            ),
            "tasks": [None] * self.batch_size,
            "valid_mask": torch.zeros(
                self.batch_size, dtype=torch.bool, device=self.device
            ),
        }

    def load_batch_to_buffer(self, buffer, batch_data):
        """异步加载数据"""
        with torch.cuda.stream(self.transfer_stream):
            valid_count = 0
            for i, data in enumerate(batch_data):
                if data is not None and i < self.batch_size:
                    buffer["images"][i].copy_(data["image_frames"], non_blocking=True)
                    buffer["hist_xyz"][i].copy_(
                        data["hist_xyz"].squeeze(), non_blocking=True
                    )
                    buffer["hist_rot"][i].copy_(
                        data["hist_rot"].squeeze(), non_blocking=True
                    )
                    buffer["future_xyz"][i].copy_(
                        data["future_xyz"].squeeze(), non_blocking=True
                    )
                    buffer["tasks"][i] = data["task"]
                    buffer["valid_mask"][i] = True
                    valid_count += 1
                else:
                    buffer["valid_mask"][i] = False
            return valid_count

    def swap_buffers(self):
        """交换buffer"""
        self.cpu_buffer, self.gpu_buffer = self.gpu_buffer, self.cpu_buffer


def load_single_frame(data_dir, result_dir, idx, index_df):
    """加载单帧"""
    try:
        row = index_df.iloc[idx]
        frame_id = int(row["frame_id"])

        # 检查是否已处理
        if os.path.exists(f"{result_dir}/pred_{frame_id:06d}.npy"):
            return None

        # 加载图像
        images = []
        for cam in CAMERA_ORDER:
            for t in range(4):
                frame_idx_img = int(row[f"{cam}_f{t}_idx"])
                img_path = f"{data_dir}/camera_images/{cam}/{frame_idx_img:06d}.jpg"
                img = Image.open(img_path).convert("RGB")
                img_array = np.array(img).astype(np.float32)
                images.append(img_array)

        images_array = np.stack(images)
        images = rearrange(images_array, "(c t) h w ch -> c t ch h w", c=4, t=4)

        # 加载egomotion
        history = np.load(f"{data_dir}/egomotion/frame_{frame_id:06d}_history.npy")
        future = np.load(f"{data_dir}/egomotion/frame_{frame_id:06d}_future_gt.npy")

        hist_xyz_world, hist_quat = history[:, 5:8], history[:, 1:5]
        future_xyz_world = future[:, 1:4]

        t0_rot_inv = R.from_quat(hist_quat[-1]).inv()
        hist_xyz = t0_rot_inv.apply(hist_xyz_world - hist_xyz_world[-1])
        hist_rot = (t0_rot_inv * R.from_quat(hist_quat)).as_matrix()

        return {
            "task": {"frame_id": frame_id, "result_dir": result_dir},
            "image_frames": torch.from_numpy(images).float(),
            "hist_xyz": torch.from_numpy(hist_xyz).float().unsqueeze(0).unsqueeze(0),
            "hist_rot": torch.from_numpy(hist_rot).float().unsqueeze(0).unsqueeze(0),
            "future_xyz": torch.from_numpy(future_xyz_world)
            .float()
            .unsqueeze(0)
            .unsqueeze(0),
        }
    except Exception as e:
        return None


def process_gpu(gpu_id, clip_id, frame_indices, data_dir, result_dir, batch_size):
    """GPU处理函数"""
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from alpamayo_r1 import helper

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    torch.cuda.init()
    device = "cuda:0"

    print(f"[GPU{gpu_id}] 加载模型...")
    model = AlpamayoR1.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        low_cpu_mem_usage=True,
    )
    model.eval()
    model = model.to(device)
    processor = helper.get_processor(model.tokenizer)
    print(f"[GPU{gpu_id}] 模型加载完成，处理 {len(frame_indices)} 帧")

    # 创建双缓冲
    double_buffer = DoubleBufferInference(batch_size, device)

    # 加载索引
    index_df = pd.read_csv(f"{data_dir}/inference_index_strict.csv")

    results = []
    times = []

    with tqdm(total=len(frame_indices), desc=f"GPU{gpu_id}") as pbar:
        # 第一批
        batch_data = []
        for idx in frame_indices[:batch_size]:
            data = load_single_frame(data_dir, result_dir, idx, index_df)
            if data:
                batch_data.append(data)

        if not batch_data:
            return 0, 0

        # 加载到cpu_buffer
        double_buffer.load_batch_to_buffer(double_buffer.cpu_buffer, batch_data)
        torch.cuda.synchronize(double_buffer.transfer_stream)

        # 处理剩余批次
        for i in range(batch_size, len(frame_indices) + batch_size, batch_size):
            # 准备下一批
            next_batch = []
            for idx in frame_indices[i : i + batch_size]:
                data = load_single_frame(data_dir, result_dir, idx, index_df)
                if data:
                    next_batch.append(data)

            # 1. 推理当前batch
            start = time.time()
            with torch.cuda.stream(double_buffer.compute_stream):
                for j in range(batch_size):
                    if not double_buffer.gpu_buffer["valid_mask"][j]:
                        continue

                    task = double_buffer.gpu_buffer["tasks"][j]
                    frame_id = task["frame_id"]

                    messages = helper.create_message(
                        double_buffer.gpu_buffer["images"][j].flatten(0, 1)
                    )
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
                        "ego_history_xyz": double_buffer.gpu_buffer["hist_xyz"][j]
                        .unsqueeze(0)
                        .unsqueeze(0),
                        "ego_history_rot": double_buffer.gpu_buffer["hist_rot"][j]
                        .unsqueeze(0)
                        .unsqueeze(0),
                    }
                    model_inputs = helper.to_device(model_inputs, device)

                    with torch.no_grad():
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            pred_xyz, _, _ = (
                                model.sample_trajectories_from_data_with_vlm_rollout(
                                    data=model_inputs,
                                    top_p=0.98,
                                    temperature=0.6,
                                    num_traj_samples=1,
                                    max_generation_length=256,
                                )
                            )

                    pred_np = pred_xyz.cpu().numpy()[0, 0, 0, :, :3]
                    np.save(f"{result_dir}/pred_{frame_id:06d}.npy", pred_np)
                    results.append(frame_id)

            # 2. 同时加载下一批
            if next_batch:
                double_buffer.load_batch_to_buffer(double_buffer.cpu_buffer, next_batch)

            # 3. 等待推理完成
            torch.cuda.synchronize(double_buffer.compute_stream)
            elapsed = time.time() - start
            times.append(elapsed)

            # 4. 等待传输完成
            if next_batch:
                torch.cuda.synchronize(double_buffer.transfer_stream)

            # 5. 交换
            double_buffer.swap_buffers()

            pbar.update(len(batch_data))
            batch_data = next_batch

    avg_time = np.mean(times) if times else 0
    print(f"[GPU{gpu_id}] 完成! 处理 {len(results)} 帧, 平均 {avg_time:.2f}s/批次")
    return len(results), avg_time


def main():
    clip_id = "04c5a3f4-7d2f-499a-ba05-b57194594735"
    gpu_list = [1, 2, 3, 5, 6, 7]
    batch_size = 8

    data_dir = f"/data01/vla/data/data_sample_chunk0/infer/{clip_id}/data"
    result_dir = f"/data01/vla/data/data_sample_chunk0/infer/{clip_id}/result_strict"
    os.makedirs(result_dir, exist_ok=True)

    # 读取索引，找出未处理的帧
    index_df = pd.read_csv(f"{data_dir}/inference_index_strict.csv")
    pending_frames = []
    for idx in range(len(index_df)):
        frame_id = int(index_df.iloc[idx]["frame_id"])
        if not os.path.exists(f"{result_dir}/pred_{frame_id:06d}.npy"):
            pending_frames.append(idx)

    total_pending = len(pending_frames)
    print(f"🚀 测试V5.1双缓冲 - Clip: {clip_id}")
    print(f"   未处理帧数: {total_pending}")
    print(f"   GPUs: {gpu_list}")
    print(f"   Batch size: {batch_size}")

    # 分配帧
    frames_per_gpu = {gpu_id: [] for gpu_id in gpu_list}
    for i, idx in enumerate(pending_frames):
        gpu_id = gpu_list[i % len(gpu_list)]
        frames_per_gpu[gpu_id].append(idx)

    for gpu_id, frames in frames_per_gpu.items():
        print(f"   GPU{gpu_id}: {len(frames)} 帧")

    # 启动多进程
    print(f"\n🎬 开始推理...")
    start = time.time()

    with ProcessPoolExecutor(max_workers=len(gpu_list)) as executor:
        futures = []
        for gpu_id in gpu_list:
            if frames_per_gpu[gpu_id]:
                future = executor.submit(
                    process_gpu,
                    gpu_id,
                    clip_id,
                    frames_per_gpu[gpu_id],
                    data_dir,
                    result_dir,
                    batch_size,
                )
                futures.append(future)

        total = 0
        for future in futures:
            count, _ = future.result()
            total += count

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"✅ 完成!")
    print(f"   处理帧数: {total}/{total_pending}")
    print(f"   耗时: {elapsed:.1f}s ({elapsed / 60:.1f}分钟)")
    print(f"   速度: {total / elapsed:.2f} fps")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
