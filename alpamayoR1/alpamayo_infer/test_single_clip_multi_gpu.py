#!/usr/bin/env python3
"""
单Clip多GPU推理测试
处理一个clip的所有帧，使用6个GPU并行
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


def process_frames_for_gpu(gpu_id, clip_id, frame_indices, data_dir, result_dir):
    """单个GPU处理指定帧"""
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from alpamayo_r1 import helper

    worker_name = f"GPU{gpu_id}"
    print(f"[{worker_name}] 启动，处理 {len(frame_indices)} 帧")

    try:
        # 设置GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        torch.cuda.init()
        device = "cuda:0"
        torch.cuda.set_device(0)

        # 加载模型
        print(f"[{worker_name}] 加载模型...")
        model = AlpamayoR1.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            low_cpu_mem_usage=True,
        )
        model.eval()
        model = model.to(device)
        processor = helper.get_processor(model.tokenizer)
        print(f"[{worker_name}] 模型加载完成")

        # 加载索引
        index_df = pd.read_csv(f"{data_dir}/inference_index_strict.csv")

        results = []
        times = []

        with tqdm(total=len(frame_indices), desc=worker_name, leave=False) as pbar:
            for idx in frame_indices:
                row = index_df.iloc[idx]
                frame_id = int(row["frame_id"])

                start_time = time.time()

                try:
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
                    images_tensor = rearrange(
                        images_array, "(c t) h w ch -> c t ch h w", c=4, t=4
                    )

                    # 加载egomotion
                    history = np.load(
                        f"{data_dir}/egomotion/frame_{frame_id:06d}_history.npy"
                    )
                    future = np.load(
                        f"{data_dir}/egomotion/frame_{frame_id:06d}_future_gt.npy"
                    )

                    hist_xyz_world, hist_quat = history[:, 5:8], history[:, 1:5]
                    future_xyz_world = future[:, 1:4]

                    t0_rot_inv = R.from_quat(hist_quat[-1]).inv()
                    hist_xyz = t0_rot_inv.apply(hist_xyz_world - hist_xyz_world[-1])
                    hist_rot = (t0_rot_inv * R.from_quat(hist_quat)).as_matrix()
                    future_xyz = t0_rot_inv.apply(future_xyz_world - hist_xyz_world[-1])

                    # 推理
                    messages = helper.create_message(images_tensor.flatten(0, 1))
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
                        "ego_history_xyz": torch.from_numpy(hist_xyz)
                        .float()
                        .unsqueeze(0)
                        .unsqueeze(0),
                        "ego_history_rot": torch.from_numpy(hist_rot)
                        .float()
                        .unsqueeze(0)
                        .unsqueeze(0),
                    }
                    model_inputs = helper.to_device(model_inputs, device)

                    with torch.no_grad():
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            pred_xyz, pred_rot, extra = (
                                model.sample_trajectories_from_data_with_vlm_rollout(
                                    data=model_inputs,
                                    top_p=0.98,
                                    temperature=0.6,
                                    num_traj_samples=1,
                                    max_generation_length=256,
                                    return_extra=True,
                                )
                            )

                    # 计算指标
                    pred_np = pred_xyz.cpu().numpy()[0, 0, 0, :, :3]
                    gt_np = future_xyz
                    ade = np.linalg.norm(pred_np - gt_np, axis=1).mean()

                    # 提取CoT
                    cot_texts = extra.get("cot", [[[]]])[0]
                    if isinstance(cot_texts, np.ndarray):
                        cot_texts = cot_texts.tolist()

                    # 保存
                    np.save(f"{result_dir}/pred_{frame_id:06d}.npy", pred_np)

                    results.append(
                        {
                            "clip_id": clip_id,
                            "frame_id": frame_id,
                            "ade": float(ade),
                            "cot_text": json.dumps(cot_texts),
                        }
                    )

                    elapsed = time.time() - start_time
                    times.append(elapsed)

                except Exception as e:
                    print(f"[{worker_name}] 处理 frame_{frame_id} 失败: {e}")

                pbar.update(1)

        # 保存CSV
        if results:
            result_csv = f"{result_dir}/inference_results_strict.csv"
            new_df = pd.DataFrame(results)
            if os.path.exists(result_csv):
                existing_df = pd.read_csv(result_csv)
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                combined_df.to_csv(result_csv, index=False)
            else:
                new_df.to_csv(result_csv, index=False)

        avg_time = np.mean(times) if times else 0
        print(f"[{worker_name}] 完成! 平均 {avg_time:.2f}s/帧")

        return len(results), avg_time

    except Exception as e:
        print(f"[{worker_name}] 错误: {e}")
        traceback.print_exc()
        return 0, 0


def main():
    clip_id = "01d3588e-bca7-4a18-8e74-c6cfe9e996db"
    gpu_list = [1, 2, 3, 5, 6, 7]

    data_dir = f"/data01/vla/data/data_sample_chunk0/infer/{clip_id}/data"
    result_dir = f"/data01/vla/data/data_sample_chunk0/infer/{clip_id}/result_strict"
    os.makedirs(result_dir, exist_ok=True)

    # 读取索引
    index_df = pd.read_csv(f"{data_dir}/inference_index_strict.csv")
    total_frames = len(index_df)

    print(f"🚀 单Clip多GPU推理测试")
    print(f"   Clip: {clip_id}")
    print(f"   总帧数: {total_frames}")
    print(f"   GPUs: {gpu_list}")

    # 分配帧到各个GPU
    frames_per_gpu = defaultdict(list)
    for i, idx in enumerate(range(total_frames)):
        gpu_idx = i % len(gpu_list)
        gpu_id = gpu_list[gpu_idx]
        frames_per_gpu[gpu_id].append(idx)

    for gpu_id, frames in frames_per_gpu.items():
        print(f"   GPU{gpu_id}: {len(frames)} 帧")

    # 启动多进程
    print(f"\n🎬 开始推理...")
    start_time = time.time()

    all_results = []
    with ProcessPoolExecutor(max_workers=len(gpu_list)) as executor:
        futures = []
        for gpu_id in gpu_list:
            if len(frames_per_gpu[gpu_id]) > 0:
                future = executor.submit(
                    process_frames_for_gpu,
                    gpu_id,
                    clip_id,
                    frames_per_gpu[gpu_id],
                    data_dir,
                    result_dir,
                )
                futures.append(future)

        for future in futures:
            try:
                count, avg_time = future.result()
                all_results.append((count, avg_time))
            except Exception as e:
                print(f"收集结果失败: {e}")

    elapsed = time.time() - start_time
    total_processed = sum(r[0] for r in all_results)
    avg_speed = total_processed / elapsed if elapsed > 0 else 0

    print(f"\n{'=' * 60}")
    print(f"✅ 测试完成!")
    print(f"   Clip: {clip_id}")
    print(f"   总帧数: {total_frames}")
    print(f"   成功处理: {total_processed}")
    print(f"   耗时: {elapsed:.1f}s ({elapsed / 60:.1f}分钟)")
    print(f"   平均速度: {avg_speed:.2f} fps")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
