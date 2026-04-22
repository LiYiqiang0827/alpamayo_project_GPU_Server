#!/usr/bin/env python3
"""
Alpamayo 1.5 多GPU批量Clip推理 V1.0
基于 batch_inference_multi_gpu_v5_v1.py 适配 Alpamayo 1.5

关键修改：
1. 模型路径改为 Alpamayo-1.5-10B
2. 导入从 alpamayo_r1 改为 alpamayo1_5
3. 模型类从 AlpamayoR1 改为 Alpamayo1_5

用法:
    python3 batch_inference_alpamayo1_5.py --chunk 0 --num_frames 0 --gpus 1,2,3,5,6,7 --batch_size 8
"""

import os
import sys

os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import subprocess
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

sys.path.insert(0, "/home/user/mikelee/alpamayo1.5-main/src")

# Alpamayo 1.5 模型路径
MODEL_PATH = "/data01/vla/models--nvidia--Alpamayo-1.5-10B"
CAMERA_ORDER = [
    "camera_cross_left_120fov",
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_front_tele_30fov",
]


class DoubleBufferInference:
    """
    双缓冲推理器 - 输入双缓冲实现

    Ping-Pong机制：
    - Ping: CPU填充数据（从CPU内存→GPU显存）
    - Pong: GPU读取数据（用于推理计算）
    """

    def __init__(self, batch_size, device="cuda"):
        self.batch_size = batch_size
        self.device = device

        # 创建两个CUDA流
        self.compute_stream = torch.cuda.Stream(device=device)
        self.transfer_stream = torch.cuda.Stream(device=device)

        # 双缓冲：预分配两个输入buffer
        self.buffer_ping = self._allocate_buffer()
        self.buffer_pong = self._allocate_buffer()
        self.cpu_buffer = self.buffer_ping  # CPU当前填充的buffer
        self.gpu_buffer = self.buffer_pong  # GPU当前使用的buffer

    def _allocate_buffer(self):
        """预分配GPU输入内存缓冲区"""
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
        """异步加载数据到指定的buffer"""
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
        """交换CPU和GPU的buffer"""
        self.cpu_buffer, self.gpu_buffer = self.gpu_buffer, self.cpu_buffer


def discover_clips(chunk_id, clip_filter=None, base_dir="/data01/vla/data"):
    """发现所有已预处理的clips"""
    chunk_dir = f"{base_dir}/data_sample_chunk{chunk_id}/infer"
    if not os.path.exists(chunk_dir):
        return []

    clips = []
    for item in sorted(os.listdir(chunk_dir)):
        clip_path = os.path.join(chunk_dir, item)
        if os.path.isdir(clip_path):
            data_dir = os.path.join(clip_path, "data")
            index_file = os.path.join(data_dir, "inference_index_strict.csv")
            if os.path.exists(index_file):
                # 如果指定了 clip_filter，只包含匹配的 clip
                if clip_filter is not None:
                    if item not in clip_filter:
                        continue
                clips.append(
                    {
                        "clip_id": item,
                        "data_dir": data_dir,
                        "result_dir": os.path.join(clip_path, "result_alpamayo1_5"),
                    }
                )
    return clips


def collect_all_pending_frames(clips, num_frames, step):
    """收集所有未处理的帧（检查CSV而非pred文件）"""
    all_pending_frames = []

    for clip in clips:
        clip_id = clip["clip_id"]
        data_dir = clip["data_dir"]
        result_dir = clip["result_dir"]

        try:
            index_df = pd.read_csv(f"{data_dir}/inference_index_strict.csv")
            total_frames = len(index_df)

            if num_frames == 0:
                target_indices = list(range(0, total_frames, step))
            else:
                target_indices = list(range(0, total_frames, step))[:num_frames]

            # 读取现有CSV结果
            result_csv = f"{result_dir}/inference_results_alpamayo1_5.csv"
            completed_ids = set()
            if os.path.exists(result_csv):
                result_df = pd.read_csv(result_csv)
                completed_ids = set(result_df.frame_id.values)

            # 检查哪些帧在CSV中缺失
            for idx in target_indices:
                frame_id = int(index_df.iloc[idx]["frame_id"])
                if frame_id not in completed_ids:
                    all_pending_frames.append(
                        {
                            "clip_id": clip_id,
                            "data_dir": data_dir,
                            "result_dir": result_dir,
                            "frame_idx": idx,
                            "frame_id": frame_id,
                        }
                    )

        except Exception as e:
            print(f"  检查 {clip_id} 失败: {e}")

    return all_pending_frames


def load_single_frame_safe(task, clip_cache):
    """
    安全加载单帧数据 - 修复版
    添加错误处理和类型检查
    """
    from scipy.spatial.transform import Rotation as R

    try:
        clip_id = task["clip_id"]
        data_dir = task["data_dir"]
        frame_idx = task["frame_idx"]
        frame_id = task["frame_id"]

        # 缓存index_df
        if clip_id not in clip_cache:
            clip_cache[clip_id] = pd.read_csv(f"{data_dir}/inference_index_strict.csv")
        index_df = clip_cache[clip_id]

        row = index_df.iloc[frame_idx]

        # 加载图像 - 添加错误检查
        images = []
        for cam in CAMERA_ORDER:
            for t in range(4):
                frame_idx_img = int(row[f"{cam}_f{t}_idx"])
                img_path = f"{data_dir}/camera_images/{cam}/{frame_idx_img:06d}.jpg"

                if not os.path.exists(img_path):
                    print(f"警告: 图像不存在 {img_path}")
                    return None

                img = Image.open(img_path).convert("RGB")
                img_array = np.array(img)

                # 检查shape
                if len(img_array.shape) != 3 or img_array.shape[2] != 3:
                    print(f"警告: 图像shape异常 {img_path}: {img_array.shape}")
                    return None

                images.append(img_array)

        # 关键修复：先转换为float32再rearrange
        images_array = np.stack(images).astype(np.float32)
        # 确保是 numpy array 而不是其他类型
        if not isinstance(images_array, np.ndarray):
            images_array = np.array(images_array)
        images = rearrange(images_array, "(c t) h w ch -> c t ch h w", c=4, t=4)

        # 加载egomotion
        history_file = f"{data_dir}/egomotion/frame_{frame_id:06d}_history.npy"
        future_file = f"{data_dir}/egomotion/frame_{frame_id:06d}_future_gt.npy"

        if not os.path.exists(history_file) or not os.path.exists(future_file):
            print(f"警告: egomotion文件不存在 frame_{frame_id:06d}")
            return None

        history = np.load(history_file, allow_pickle=False)
        future = np.load(future_file, allow_pickle=False)

        hist_xyz_world, hist_quat = history[:, 5:8], history[:, 1:5]
        future_xyz_world = future[:, 1:4]

        t0_rot_inv = R.from_quat(hist_quat[-1]).inv()
        hist_xyz = t0_rot_inv.apply(hist_xyz_world - hist_xyz_world[-1])
        hist_rot = (t0_rot_inv * R.from_quat(hist_quat)).as_matrix()
        future_xyz = t0_rot_inv.apply(future_xyz_world - hist_xyz_world[-1])

        # 转换为CPU tensor
        image_frames = torch.from_numpy(images).float()
        hist_xyz_t = torch.from_numpy(hist_xyz).float().unsqueeze(0).unsqueeze(0)
        hist_rot_t = torch.from_numpy(hist_rot).float().unsqueeze(0).unsqueeze(0)
        future_xyz_t = torch.from_numpy(future_xyz).float().unsqueeze(0).unsqueeze(0)

        return {
            "task": task,
            "image_frames": image_frames,
            "hist_xyz": hist_xyz_t,
            "hist_rot": hist_rot_t,
            "future_xyz": future_xyz_t,
        }

    except Exception as e:
        print(f"加载帧失败 frame_{task.get('frame_id', 'unknown')}: {e}")
        traceback.print_exc()
        return None


class DataPrefetcher:
    """数据预取器"""

    def __init__(self, frame_tasks, batch_size, num_prefetch=2, num_workers=4):
        self.frame_tasks = frame_tasks
        self.batch_size = batch_size
        self.num_prefetch = num_prefetch
        self.num_workers = num_workers
        self.queue = Queue(maxsize=num_prefetch)
        self.clip_cache = {}
        self.stop_flag = threading.Event()
        self.prefetch_thread = threading.Thread(target=self._prefetch_loop)
        self.prefetch_thread.start()

    def _prefetch_loop(self):
        """预取线程"""
        for i in range(0, len(self.frame_tasks), self.batch_size):
            if self.stop_flag.is_set():
                break

            batch_tasks = self.frame_tasks[i : i + self.batch_size]

            # 多线程并行加载
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                batch_data = list(
                    executor.map(
                        lambda t: load_single_frame_safe(t, self.clip_cache),
                        batch_tasks,
                    )
                )

            batch_data = [d for d in batch_data if d is not None]

            if batch_data:
                self.queue.put(batch_data)

        self.queue.put(None)

    def get_batch(self):
        return self.queue.get()

    def stop(self):
        self.stop_flag.set()
        self.prefetch_thread.join(timeout=5)


def process_batch_with_double_buffer(gpu_buffer, model, processor, args_dict):
    """使用双缓冲的批处理"""
    from alpamayo1_5 import helper

    results = []
    batch_size = len(gpu_buffer["tasks"])

    # 逐帧处理（模型不支持真批处理）
    for i in range(batch_size):
        if not gpu_buffer["valid_mask"][i]:
            continue

        try:
            task = gpu_buffer["tasks"][i]
            result_dir = task["result_dir"]
            frame_id = task["frame_id"]

            os.makedirs(result_dir, exist_ok=True)

            # 单帧推理
            messages = helper.create_message(gpu_buffer["images"][i].flatten(0, 1))
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
                "ego_history_xyz": gpu_buffer["hist_xyz"][i].unsqueeze(0).unsqueeze(0),
                "ego_history_rot": gpu_buffer["hist_rot"][i].unsqueeze(0).unsqueeze(0),
            }
            model_inputs = helper.to_device(model_inputs, "cuda")
            torch.cuda.manual_seed_all(42)

            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred_xyz, pred_rot, extra = (
                        model.sample_trajectories_from_data_with_vlm_rollout(
                            data=model_inputs,
                            top_p=args_dict["top_p"],
                            temperature=args_dict["temp"],
                            num_traj_samples=args_dict["num_traj"],
                            max_generation_length=args_dict["max_len"],
                            return_extra=True,
                        )
                    )

            # 计算指标
            pred_np = pred_xyz.cpu().numpy()[0, 0, 0, :, :3]
            gt_np = gpu_buffer["future_xyz"][i].cpu().numpy()[0, 0, :, :3]
            ade = np.linalg.norm(pred_np - gt_np, axis=1).mean()

            # 提取CoT
            cot_texts = extra.get("cot", [[[]]])[0]
            if isinstance(cot_texts, np.ndarray):
                cot_texts = cot_texts.tolist()

            # 保存预测
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

        except Exception as e:
            print(f"处理单帧失败: {e}")

    return results


def save_results_async(results_queue):
    """异步保存结果"""
    results_by_clip = defaultdict(list)

    while True:
        result = results_queue.get()

        if result is None:
            break

        results_by_clip[result["clip_id"]].append(result)

        # 每积累50条保存一次
        if len(results_by_clip[result["clip_id"]]) >= 50:
            save_clip_results(
                result["clip_id"],
                results_by_clip[result["clip_id"]],
                result["result_dir"],
            )
            results_by_clip[result["clip_id"]] = []

    # 保存剩余
    for clip_id, clip_results in results_by_clip.items():
        if clip_results:
            result_dir = clip_results[0]["result_dir"]
            save_clip_results(clip_id, clip_results, result_dir)


def save_clip_results(clip_id, clip_results, result_dir):
    """保存结果到CSV"""
    try:
        result_csv = f"{result_dir}/inference_results_alpamayo1_5.csv"
        save_results = [
            {k: v for k, v in r.items() if k != "result_dir"} for r in clip_results
        ]
        new_df = pd.DataFrame(save_results)

        if os.path.exists(result_csv):
            existing_df = pd.read_csv(result_csv)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df.to_csv(result_csv, index=False)
        else:
            new_df.to_csv(result_csv, index=False)
    except Exception as e:
        print(f"保存结果失败 {clip_id}: {e}")


def inference_worker_entry_with_double_buffer(
    worker_id, gpu_id, frame_tasks, args_dict
):
    """Worker入口 - 使用输入双缓冲"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
    from alpamayo1_5 import helper

    worker_name = f"GPU{gpu_id}-Worker{worker_id}"
    batch_size = args_dict.get("batch_size", 1)

    print(f"[{worker_name}] 启动，处理 {len(frame_tasks)} 帧，batch_size={batch_size}")

    try:
        print(f"[{worker_name}] 加载 Alpamayo 1.5 模型...")
        model = Alpamayo1_5.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            low_cpu_mem_usage=True,
        )
        model.eval()
        model = model.to("cuda")
        processor = helper.get_processor(model.tokenizer)
        print(f"[{worker_name}] 模型加载完成")
    except Exception as e:
        print(f"[{worker_name}] 模型加载失败: {e}")
        traceback.print_exc()
        return []

    # 创建双缓冲推理器
    double_buffer = DoubleBufferInference(batch_size, device="cuda")

    # 异步保存线程
    results_queue = Queue()
    save_thread = threading.Thread(target=save_results_async, args=(results_queue,))
    save_thread.start()

    results = []
    processed_count = 0
    failed_count = 0

    # 数据预取器
    prefetcher = DataPrefetcher(frame_tasks, batch_size, num_prefetch=3, num_workers=4)

    with tqdm(total=len(frame_tasks), desc=f"{worker_name}", leave=False) as pbar:
        # 获取第一批数据（预热）
        first_batch = prefetcher.get_batch()
        if first_batch is None or not first_batch:
            print(f"[{worker_name}] 没有数据可处理")
            prefetcher.stop()
            results_queue.put(None)
            save_thread.join()
            return results

        # 加载第一批到cpu_buffer
        double_buffer.load_batch_to_buffer(double_buffer.cpu_buffer, first_batch)

        # 等待传输完成
        torch.cuda.synchronize(double_buffer.transfer_stream)

        # 主循环：双缓冲流水线
        next_batch = prefetcher.get_batch()

        while next_batch is not None:
            # 1. 启动当前buffer的推理（在compute_stream上）
            compute_event = torch.cuda.Event()
            with torch.cuda.stream(double_buffer.compute_stream):
                batch_results = process_batch_with_double_buffer(
                    double_buffer.gpu_buffer, model, processor, args_dict
                )
                results.extend(batch_results)
                processed_count += len(batch_results)
                failed_count += batch_size - len(batch_results)

                for r in batch_results:
                    results_queue.put(r)

                compute_event.record(double_buffer.compute_stream)

            # 2. 同时加载下一批到cpu_buffer（在transfer_stream上）
            if next_batch:
                transfer_event = torch.cuda.Event()
                double_buffer.load_batch_to_buffer(double_buffer.cpu_buffer, next_batch)
                transfer_event.record(double_buffer.transfer_stream)

            # 3. 等待推理完成
            compute_event.synchronize()

            # 4. 等待数据传输完成（如果还有下一批）
            if next_batch:
                transfer_event.synchronize()

            # 5. 交换buffer
            double_buffer.swap_buffers()

            # 6. 获取再下一批
            next_batch = prefetcher.get_batch()
            pbar.update(len(first_batch) if "first_batch" in dir() else batch_size)

            if first_batch:
                first_batch = None

    prefetcher.stop()
    results_queue.put(None)
    save_thread.join()

    print(f"[{worker_name}] 完成: {processed_count}成功, {failed_count}失败")
    return results


def distribute_frames_to_gpus(all_frames, gpu_list):
    """均匀分配帧到GPU"""
    tasks_per_gpu = {gpu_id: [] for gpu_id in gpu_list}

    for i, frame_info in enumerate(all_frames):
        gpu_idx = i % len(gpu_list)
        gpu_id = gpu_list[gpu_idx]
        tasks_per_gpu[gpu_id].append(frame_info)

    return tasks_per_gpu


def main():
    parser = argparse.ArgumentParser(
        description="Alpamayo 1.5 多GPU批量Clip推理 V1.0"
    )
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument(
        "--clip", type=str, default="0",
        help="指定clip ID(s)，多个用逗号分隔 (如: 'clip1,clip2')，默认'0'表示全部clips"
    )
    parser.add_argument(
        "--num_frames", type=int, default=1000, help="每clip帧数 (0=全部)"
    )
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--traj", type=int, default=1)
    parser.add_argument("--top_p", type=float, default=0.98)
    parser.add_argument("--temp", type=float, default=0.6)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument(
        "--gpus", type=str, default="1,2,3,5,6,7", help="GPU列表 (如: 1,2,3,5,6,7)"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="批处理大小")
    parser.add_argument("--workers_per_gpu", type=int, default=1, help="每GPU进程数")

    args = parser.parse_args()

    # 解析 clip 参数
    if args.clip == "0" or args.clip == "" or args.clip is None:
        clip_filter = None  # 处理所有 clips
        clip_desc = "全部clips"
    else:
        clip_filter = set(args.clip.split(","))
        clip_desc = f"指定clips: {', '.join(sorted(clip_filter))}"

    print(f"======================================================================")
    print(f"🚀 Alpamayo 1.5 多GPU批量Clip推理 V1.0")
    print(f"======================================================================")
    print(f"配置: chunk={args.chunk}, {clip_desc}")
    print(
        f"每clip帧数: {args.num_frames if args.num_frames > 0 else '全部'}, step={args.step}"
    )
    print(f"使用GPU: {[int(x) for x in args.gpus.split(',')]}")
    print(f"批处理大小: {args.batch_size}")
    print(f"优化: 输入双缓冲 (Ping-Pong)")

    gpu_list = [int(x) for x in args.gpus.split(",")]

    # 发现clips
    print(f"\n📂 扫描clips...")
    clips = discover_clips(args.chunk, clip_filter=clip_filter)
    print(f"发现 {len(clips)} 个clips")

    # 收集待处理帧
    print(f"\n📋 收集所有未处理帧...")
    all_pending_frames = collect_all_pending_frames(clips, args.num_frames, args.step)
    print(f"总计 {len(all_pending_frames)} 帧待处理")

    if len(all_pending_frames) == 0:
        print("✅ 所有帧已处理完成！")
        return

    # 分配到GPU
    print(f"\n📊 分配任务到 {len(gpu_list)} 个GPU...")
    tasks_per_gpu = distribute_frames_to_gpus(all_pending_frames, gpu_list)
    for gpu_id, tasks in tasks_per_gpu.items():
        print(
            f"  GPU{gpu_id}: {len(tasks)} 帧 ({len(tasks) // args.batch_size + (1 if len(tasks) % args.batch_size else 0)} 批次)"
        )

    # 准备参数字典
    args_dict = {
        "top_p": args.top_p,
        "temp": args.temp,
        "num_traj": args.traj,
        "max_len": args.max_len,
        "batch_size": args.batch_size,
    }

    # 多进程启动GPU workers
    print(f"\n🔄 启动推理进程...")
    start_time = time.time()

    all_results = []
    with ProcessPoolExecutor(max_workers=len(gpu_list)) as executor:
        futures = []
        for gpu_id in gpu_list:
            if len(tasks_per_gpu[gpu_id]) > 0:
                future = executor.submit(
                    inference_worker_entry_with_double_buffer,
                    0,  # worker_id
                    gpu_id,
                    tasks_per_gpu[gpu_id],
                    args_dict,
                )
                futures.append(future)

        # 收集结果
        for future in futures:
            try:
                results = future.result()
                all_results.extend(results)
            except Exception as e:
                print(f"收集结果失败: {e}")

    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"📊 处理完成!")
    print(f"{'=' * 70}")
    print(f"总帧数: {len(all_pending_frames)}")
    print(
        f"成功处理: {len(all_results)}/{len(all_pending_frames)} ({len(all_results) / len(all_pending_frames) * 100:.1f}%)"
    )
    print(f"总耗时: {elapsed:.1f}s ({elapsed / 60:.1f}分钟)")
    print(f"平均速度: {len(all_results) / elapsed:.2f} 帧/秒")
    if all_results:
        avg_ade = np.mean([r["ade"] for r in all_results])
        print(f"平均ADE: {avg_ade:.4f}m")
    print(f"{'=' * 70}")

    # 保存汇总
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "chunk": args.chunk,
        "num_frames_requested": args.num_frames,
        "total_frames": len(all_pending_frames),
        "processed": len(all_results),
        "elapsed_seconds": elapsed,
        "fps": len(all_results) / elapsed if elapsed > 0 else 0,
        "avg_ade": float(np.mean([r["ade"] for r in all_results]))
        if all_results
        else 0,
    }

    summary_path = (
        f"/data01/vla/cot_collection/batch_alpamayo1_5_chunk{args.chunk}.json"
    )
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✅ 汇总保存: {summary_path}")


if __name__ == "__main__":
    main()
