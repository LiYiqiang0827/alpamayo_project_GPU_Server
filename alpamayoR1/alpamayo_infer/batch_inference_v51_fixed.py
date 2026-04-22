#!/usr/bin/env python3
"""
多GPU批量Clip推理 V5.1 - 输入双缓冲优化版 (修复版)
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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
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


def load_single_frame_safe(task, clip_cache):
    """安全加载单帧数据"""
    try:
        clip_id = task["clip_id"]
        data_dir = task["data_dir"]
        frame_idx = task["frame_idx"]
        frame_id = task["frame_id"]
        result_dir = task["result_dir"]

        # 检查是否已处理
        if os.path.exists(f"{result_dir}/pred_{frame_id:06d}.npy"):
            return None

        # 缓存index_df
        if clip_id not in clip_cache:
            clip_cache[clip_id] = pd.read_csv(f"{data_dir}/inference_index_strict.csv")
        index_df = clip_cache[clip_id]

        row = index_df.iloc[frame_idx]

        # 加载图像
        images = []
        for cam in CAMERA_ORDER:
            for t in range(4):
                frame_idx_img = int(row[f"{cam}_f{t}_idx"])
                img_path = f"{data_dir}/camera_images/{cam}/{frame_idx_img:06d}.jpg"

                if not os.path.exists(img_path):
                    return None

                img = Image.open(img_path).convert("RGB")
                img_array = np.array(img).astype(np.float32)
                images.append(img_array)

        images_array = np.stack(images)
        images = rearrange(images_array, "(c t) h w ch -> c t ch h w", c=4, t=4)

        # 加载egomotion
        history_file = f"{data_dir}/egomotion/frame_{frame_id:06d}_history.npy"
        future_file = f"{data_dir}/egomotion/frame_{frame_id:06d}_future_gt.npy"

        if not os.path.exists(history_file) or not os.path.exists(future_file):
            return None

        history = np.load(history_file, allow_pickle=False)
        future = np.load(future_file, allow_pickle=False)

        hist_xyz_world, hist_quat = history[:, 5:8], history[:, 1:5]
        future_xyz_world = future[:, 1:4]

        t0_rot_inv = R.from_quat(hist_quat[-1]).inv()
        hist_xyz = t0_rot_inv.apply(hist_xyz_world - hist_xyz_world[-1])
        hist_rot = (t0_rot_inv * R.from_quat(hist_quat)).as_matrix()

        image_frames = torch.from_numpy(images).float()
        hist_xyz_t = torch.from_numpy(hist_xyz).float().unsqueeze(0).unsqueeze(0)
        hist_rot_t = torch.from_numpy(hist_rot).float().unsqueeze(0).unsqueeze(0)
        future_xyz_t = (
            torch.from_numpy(future_xyz_world).float().unsqueeze(0).unsqueeze(0)
        )

        return {
            "task": task,
            "image_frames": image_frames,
            "hist_xyz": hist_xyz_t,
            "hist_rot": hist_rot_t,
            "future_xyz": future_xyz_t,
        }
    except Exception as e:
        return None


def inference_worker_entry_v51(worker_id, gpu_id, frame_tasks, args_dict):
    """Worker入口 - V5.1双缓冲版本"""
    # 必须在最开始设置环境变量
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from alpamayo_r1 import helper

    worker_name = f"GPU{gpu_id}-Worker{worker_id}"
    batch_size = args_dict.get("batch_size", 1)

    print(f"[{worker_name}] 启动，处理 {len(frame_tasks)} 帧，batch_size={batch_size}")

    try:
        torch.cuda.init()
        device = "cuda:0"
        torch.cuda.set_device(0)

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
    except Exception as e:
        print(f"[{worker_name}] 模型加载失败: {e}")
        traceback.print_exc()
        return []

    # 创建CUDA Streams
    compute_stream = torch.cuda.Stream(device=device)
    transfer_stream = torch.cuda.Stream(device=device)

    # 预分配双缓冲
    buffer_a = {
        "images": torch.empty(
            batch_size, 4, 4, 3, 224, 224, dtype=torch.float32, device=device
        ),
        "hist_xyz": torch.empty(
            batch_size, 1, 1, 16, 3, dtype=torch.float32, device=device
        ),
        "hist_rot": torch.empty(
            batch_size, 1, 1, 16, 3, 3, dtype=torch.float32, device=device
        ),
        "future_xyz": torch.empty(
            batch_size, 1, 1, 64, 3, dtype=torch.float32, device=device
        ),
        "tasks": [None] * batch_size,
        "valid_mask": torch.zeros(batch_size, dtype=torch.bool, device=device),
    }
    buffer_b = {
        "images": torch.empty(
            batch_size, 4, 4, 3, 224, 224, dtype=torch.float32, device=device
        ),
        "hist_xyz": torch.empty(
            batch_size, 1, 1, 16, 3, dtype=torch.float32, device=device
        ),
        "hist_rot": torch.empty(
            batch_size, 1, 1, 16, 3, 3, dtype=torch.float32, device=device
        ),
        "future_xyz": torch.empty(
            batch_size, 1, 1, 64, 3, dtype=torch.float32, device=device
        ),
        "tasks": [None] * batch_size,
        "valid_mask": torch.zeros(batch_size, dtype=torch.bool, device=device),
    }
    cpu_buffer = buffer_a
    gpu_buffer = buffer_b

    # 异步保存
    results_queue = Queue()

    def save_async():
        results_by_clip = defaultdict(list)
        while True:
            result = results_queue.get()
            if result is None:
                break
            results_by_clip[result["clip_id"]].append(result)
            if len(results_by_clip[result["clip_id"]]) >= 50:
                save_results(results_by_clip, result["result_dir"])
                results_by_clip[result["clip_id"]] = []
        for clip_id, clip_results in results_by_clip.items():
            if clip_results:
                save_results({clip_id: clip_results}, clip_results[0]["result_dir"])

    def save_results(results_by_clip, result_dir):
        try:
            for clip_id, clip_results in results_by_clip.items():
                result_csv = f"{result_dir}/inference_results_strict.csv"
                save_data = [
                    {k: v for k, v in r.items() if k != "result_dir"}
                    for r in clip_results
                ]
                new_df = pd.DataFrame(save_data)
                if os.path.exists(result_csv):
                    existing_df = pd.read_csv(result_csv)
                    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                    combined_df.to_csv(result_csv, index=False)
                else:
                    new_df.to_csv(result_csv, index=False)
        except Exception as e:
            print(f"保存失败: {e}")

    save_thread = threading.Thread(target=save_async)
    save_thread.start()

    # 数据预取
    clip_cache = {}
    results = []
    processed_count = 0
    failed_count = 0

    def load_batch_to_buffer(buffer, batch_data, stream):
        with torch.cuda.stream(stream):
            valid_count = 0
            for i, data in enumerate(batch_data):
                if data is not None and i < batch_size:
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

    # 预加载第一批
    batch_data = []
    for idx in range(min(batch_size, len(frame_tasks))):
        data = load_single_frame_safe(frame_tasks[idx], clip_cache)
        if data:
            batch_data.append(data)

    if not batch_data:
        save_thread.join()
        return []

    load_batch_to_buffer(cpu_buffer, batch_data, transfer_stream)
    torch.cuda.synchronize(transfer_stream)

    with tqdm(total=len(frame_tasks), desc=f"{worker_name}", leave=False) as pbar:
        for i in range(batch_size, len(frame_tasks) + batch_size, batch_size):
            # 准备下一批
            next_batch_data = []
            for j in range(i, min(i + batch_size, len(frame_tasks))):
                data = load_single_frame_safe(frame_tasks[j], clip_cache)
                if data:
                    next_batch_data.append(data)

            # 1. GPU推理当前batch
            with torch.cuda.stream(compute_stream):
                for j in range(batch_size):
                    if not gpu_buffer["valid_mask"][j]:
                        continue
                    try:
                        task = gpu_buffer["tasks"][j]
                        result_dir = task["result_dir"]
                        frame_id = task["frame_id"]
                        os.makedirs(result_dir, exist_ok=True)

                        messages = helper.create_message(
                            gpu_buffer["images"][j].flatten(0, 1)
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
                            "ego_history_xyz": gpu_buffer["hist_xyz"][j]
                            .unsqueeze(0)
                            .unsqueeze(0),
                            "ego_history_rot": gpu_buffer["hist_rot"][j]
                            .unsqueeze(0)
                            .unsqueeze(0),
                        }
                        model_inputs = helper.to_device(model_inputs, device)

                        with torch.no_grad():
                            with torch.autocast("cuda", dtype=torch.bfloat16):
                                pred_xyz, _, extra = (
                                    model.sample_trajectories_from_data_with_vlm_rollout(
                                        data=model_inputs,
                                        top_p=args_dict["top_p"],
                                        temperature=args_dict["temp"],
                                        num_traj_samples=args_dict["num_traj"],
                                        max_generation_length=args_dict["max_len"],
                                        return_extra=True,
                                    )
                                )

                        pred_np = pred_xyz.cpu().numpy()[0, 0, 0, :, :3]
                        gt_np = gpu_buffer["future_xyz"][j].cpu().numpy()[0, 0, :, :3]
                        ade = np.linalg.norm(pred_np - gt_np, axis=1).mean()

                        cot_texts = extra.get("cot", [[[]]])[0]
                        if isinstance(cot_texts, np.ndarray):
                            cot_texts = cot_texts.tolist()

                        np.save(f"{result_dir}/pred_{frame_id:06d}.npy", pred_np)

                        results.append(
                            {
                                "clip_id": task["clip_id"],
                                "frame_id": frame_id,
                                "ade": float(ade),
                                "cot_text": json.dumps(cot_texts),
                                "result_dir": result_dir,
                            }
                        )
                        processed_count += 1

                    except Exception as e:
                        failed_count += 1

            # 2. 同时加载下一批
            if next_batch_data:
                load_batch_to_buffer(cpu_buffer, next_batch_data, transfer_stream)

            # 3. 等待推理完成
            torch.cuda.synchronize(compute_stream)

            # 4. 等待传输完成
            if next_batch_data:
                torch.cuda.synchronize(transfer_stream)

            # 5. 交换buffer
            cpu_buffer, gpu_buffer = gpu_buffer, cpu_buffer

            # 6. 发送结果到保存线程
            for r in results:
                results_queue.put(r)
            results = []

            pbar.update(len(batch_data))
            batch_data = next_batch_data

            if not batch_data:
                break

    results_queue.put(None)
    save_thread.join()

    print(f"[{worker_name}] 完成: {processed_count}成功, {failed_count}失败")
    return processed_count


def main():
    parser = argparse.ArgumentParser(description="V5.1双缓冲测试")
    parser.add_argument("--clip", type=str, required=True, help="Clip ID")
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--gpus", type=str, default="1,2,3,5,6,7")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    clip_id = args.clip
    gpu_list = [int(x) for x in args.gpus.split(",")]

    data_dir = f"/data01/vla/data/data_sample_chunk{args.chunk}/infer/{clip_id}/data"
    result_dir = (
        f"/data01/vla/data/data_sample_chunk{args.chunk}/infer/{clip_id}/result_strict"
    )

    # 收集未处理帧
    index_df = pd.read_csv(f"{data_dir}/inference_index_strict.csv")
    pending_frames = []
    for idx in range(len(index_df)):
        frame_id = int(index_df.iloc[idx]["frame_id"])
        if not os.path.exists(f"{result_dir}/pred_{frame_id:06d}.npy"):
            pending_frames.append(
                {
                    "clip_id": clip_id,
                    "data_dir": data_dir,
                    "result_dir": result_dir,
                    "frame_idx": idx,
                    "frame_id": frame_id,
                }
            )

    total = len(pending_frames)
    print(f"🚀 V5.1双缓冲测试 - Clip: {clip_id}")
    print(f"   未处理帧: {total}")
    print(f"   GPUs: {gpu_list}")
    print(f"   Batch: {args.batch_size}")

    # 分配帧
    tasks_per_gpu = {gpu_id: [] for gpu_id in gpu_list}
    for i, task in enumerate(pending_frames):
        gpu_id = gpu_list[i % len(gpu_list)]
        tasks_per_gpu[gpu_id].append(task)

    for gpu_id, tasks in tasks_per_gpu.items():
        print(f"   GPU{gpu_id}: {len(tasks)} 帧")

    # 启动
    print(f"\n🎬 开始推理...")
    start = time.time()

    args_dict = {
        "top_p": 0.98,
        "temp": 0.6,
        "num_traj": 1,
        "max_len": 256,
        "batch_size": args.batch_size,
    }

    total_processed = 0
    with ProcessPoolExecutor(max_workers=len(gpu_list)) as executor:
        futures = []
        for gpu_id in gpu_list:
            if tasks_per_gpu[gpu_id]:
                future = executor.submit(
                    inference_worker_entry_v51,
                    0,
                    gpu_id,
                    tasks_per_gpu[gpu_id],
                    args_dict,
                )
                futures.append(future)

        for future in futures:
            try:
                count = future.result()
                total_processed += count
            except Exception as e:
                print(f"错误: {e}")

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"✅ 完成!")
    print(f"   处理: {total_processed}/{total}")
    print(f"   耗时: {elapsed:.1f}s")
    print(f"   速度: {total_processed / elapsed:.2f} fps")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
