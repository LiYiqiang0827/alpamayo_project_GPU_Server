#!/usr/bin/env python3
"""
多GPU批量Clip推理 V4 - 异步流水线优化版
优化点：
1. 批量处理 (Batching) - 每次处理多帧
2. 多线程数据预取 - 预加载下一批数据
3. CUDA Stream异步 - 重叠数据传输和计算
4. pin_memory - 加速CPU→GPU传输
5. 异步保存 - 结果写入不阻塞推理

用法:
    python3 batch_inference_multi_gpu_v4.py --chunk 0 --num_frames 0 --gpus 1,2,3,5,6,7 --batch_size 4
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

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from einops import rearrange

sys.path.insert(0, "/home/user/mikelee/alpamayo-main/src")

MODEL_PATH = "/data01/vla/models--nvidia--Alpamayo-R1-10B/snapshots/22fab1399111f50b52bfbe5d8b809f39bd4c2fe1"
CAMERA_ORDER = [
    "camera_cross_left_120fov",
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_front_tele_30fov",
]


def discover_clips(chunk_id, base_dir="/data01/vla/data"):
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
                clips.append(
                    {
                        "clip_id": item,
                        "data_dir": data_dir,
                        "result_dir": os.path.join(clip_path, "result_strict"),
                    }
                )
    return clips


def collect_all_pending_frames(clips, num_frames, step):
    """
    收集所有clips中所有未处理的帧
    返回: [(clip_id, data_dir, result_dir, frame_idx, frame_id), ...]
    """
    all_pending_frames = []

    for clip in clips:
        clip_id = clip["clip_id"]
        data_dir = clip["data_dir"]
        result_dir = clip["result_dir"]

        try:
            index_df = pd.read_csv(f"{data_dir}/inference_index_strict.csv")
            total_frames = len(index_df)

            # 确定要处理的帧索引
            if num_frames == 0:
                target_indices = list(range(0, total_frames, step))
            else:
                target_indices = list(range(0, total_frames, step))[:num_frames]

            # 检查哪些帧已经处理过
            for idx in target_indices:
                frame_id = int(index_df.iloc[idx]["frame_id"])
                pred_file = f"{result_dir}/pred_{frame_id:06d}.npy"
                if not os.path.exists(pred_file):
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


def distribute_frames_to_gpus(all_frames, gpu_list):
    """
    将所有帧均匀分配给所有GPU
    返回: {gpu_id: [frame_info, ...], ...}
    """
    tasks_per_gpu = {gpu_id: [] for gpu_id in gpu_list}

    for i, frame_info in enumerate(all_frames):
        gpu_idx = i % len(gpu_list)
        gpu_id = gpu_list[gpu_idx]
        tasks_per_gpu[gpu_id].append(frame_info)

    return tasks_per_gpu


def load_single_frame(task, clip_cache):
    """
    加载单帧数据（用于多线程预取）
    返回处理好的CPU tensor
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

        # 加载图像
        images = []
        for cam in CAMERA_ORDER:
            for t in range(4):
                frame_idx_img = int(row[f"{cam}_f{t}_idx"])
                img_path = f"{data_dir}/camera_images/{cam}/{frame_idx_img:06d}.jpg"
                img = Image.open(img_path).convert("RGB")
                images.append(np.array(img))

        images = rearrange(np.stack(images), "(c t) h w ch -> c t ch h w", c=4, t=4)

        # 加载egomotion
        history = np.load(
            f"{data_dir}/egomotion/frame_{frame_id:06d}_history.npy", allow_pickle=False
        )
        future = np.load(
            f"{data_dir}/egomotion/frame_{frame_id:06d}_future_gt.npy",
            allow_pickle=False,
        )

        hist_xyz_world, hist_quat = history[:, 5:8], history[:, 1:5]
        future_xyz_world = future[:, 1:4]

        t0_rot_inv = R.from_quat(hist_quat[-1]).inv()
        hist_xyz = t0_rot_inv.apply(hist_xyz_world - hist_xyz_world[-1])
        hist_rot = (t0_rot_inv * R.from_quat(hist_quat)).as_matrix()
        future_xyz = t0_rot_inv.apply(future_xyz_world - hist_xyz_world[-1])

        # 转换为CPU tensor (使用pin_memory加速后续传输)
        image_frames = torch.from_numpy(images).float()  # .pin_memory()
        hist_xyz_t = (
            torch.from_numpy(hist_xyz).float().unsqueeze(0).unsqueeze(0)
        )  # .pin_memory()
        hist_rot_t = (
            torch.from_numpy(hist_rot).float().unsqueeze(0).unsqueeze(0)
        )  # .pin_memory()
        future_xyz_t = (
            torch.from_numpy(future_xyz).float().unsqueeze(0).unsqueeze(0)
        )  # .pin_memory()

        return {
            "task": task,
            "image_frames": image_frames,
            "hist_xyz": hist_xyz_t,
            "hist_rot": hist_rot_t,
            "future_xyz": future_xyz_t,
        }
    except Exception as e:
        print(f"加载帧失败: {e}")
        return None


class DataPrefetcher:
    """
    数据预取器 - 多线程预加载下一批数据
    """

    def __init__(self, frame_tasks, batch_size, num_prefetch=2):
        self.frame_tasks = frame_tasks
        self.batch_size = batch_size
        self.num_prefetch = num_prefetch
        self.queue = Queue(maxsize=num_prefetch)
        self.clip_cache = {}
        self.stop_flag = threading.Event()
        self.prefetch_thread = threading.Thread(target=self._prefetch_loop)
        self.prefetch_thread.start()

    def _prefetch_loop(self):
        """预取线程主循环"""
        for i in range(0, len(self.frame_tasks), self.batch_size):
            if self.stop_flag.is_set():
                break

            batch_tasks = self.frame_tasks[i : i + self.batch_size]

            # 多线程并行加载一批数据
            with ThreadPoolExecutor(max_workers=4) as executor:
                batch_data = list(
                    executor.map(
                        lambda t: load_single_frame(t, self.clip_cache), batch_tasks
                    )
                )

            # 过滤失败的
            batch_data = [d for d in batch_data if d is not None]

            if batch_data:
                self.queue.put(batch_data)

        # 标记结束
        self.queue.put(None)

    def get_batch(self):
        """获取预取的数据批次"""
        return self.queue.get()

    def stop(self):
        """停止预取"""
        self.stop_flag.set()
        self.prefetch_thread.join(timeout=5)


def inference_worker_entry(worker_id, gpu_id, frame_tasks, args_dict):
    """
    Worker入口 - 使用CUDA Stream异步处理
    """
    # 设置GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from alpamayo_r1 import helper
    from scipy.spatial.transform import Rotation as R

    worker_name = f"GPU{gpu_id}-Worker{worker_id}"
    batch_size = args_dict.get("batch_size", 1)

    print(f"[{worker_name}] 启动，处理 {len(frame_tasks)} 帧，batch_size={batch_size}")

    try:
        # 加载模型
        print(f"[{worker_name}] 加载模型...")
        model = AlpamayoR1.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            low_cpu_mem_usage=True,
        )
        processor = helper.get_processor(model.tokenizer)
        print(f"[{worker_name}] 模型加载完成")
    except Exception as e:
        print(f"[{worker_name}] 模型加载失败: {e}")
        traceback.print_exc()
        return []

    # 创建CUDA Stream用于异步处理
    stream = torch.cuda.Stream(device=f"cuda:0")

    # 结果保存队列（异步保存）
    results_queue = Queue()
    save_thread = threading.Thread(target=save_results_async, args=(results_queue,))
    save_thread.start()

    results = []
    processed_count = 0

    # 使用数据预取器
    prefetcher = DataPrefetcher(frame_tasks, batch_size, num_prefetch=3)

    with tqdm(total=len(frame_tasks), desc=f"{worker_name}", leave=False) as pbar:
        while True:
            # 获取预取的数据批次
            batch_data = prefetcher.get_batch()

            if batch_data is None:  # 结束标记
                break

            if not batch_data:
                continue

            # 使用CUDA Stream异步处理
            with torch.cuda.stream(stream):
                batch_results = process_batch(batch_data, model, processor, args_dict)
                results.extend(batch_results)
                processed_count += len(batch_results)

                # 异步保存结果
                for r in batch_results:
                    results_queue.put(r)

            # 同步Stream（确保GPU完成）
            stream.synchronize()

            pbar.update(len(batch_data))

    # 停止预取器
    prefetcher.stop()

    # 等待所有结果保存完成
    results_queue.put(None)  # 结束标记
    save_thread.join()

    print(f"[{worker_name}] 完成，处理了 {processed_count} 帧")
    return results


def process_batch(batch_data, model, processor, args_dict):
    """
    批量处理一批数据
    """
    from alpamayo_r1 import helper

    results = []

    for data in batch_data:
        try:
            task = data["task"]
            result_dir = task["result_dir"]
            frame_id = task["frame_id"]
            future_xyz_t = data["future_xyz"]

            os.makedirs(result_dir, exist_ok=True)

            # 数据传输到GPU
            image_frames = data["image_frames"].cuda(non_blocking=True)
            hist_xyz_t = data["hist_xyz"].cuda(non_blocking=True)
            hist_rot_t = data["hist_rot"].cuda(non_blocking=True)
            future_xyz_gpu = future_xyz_t.cuda(non_blocking=True)

            # 推理
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
                "ego_history_xyz": hist_xyz_t,
                "ego_history_rot": hist_rot_t,
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
            gt_np = future_xyz_gpu.cpu().numpy()[0, 0, :, :3]
            ade = np.linalg.norm(pred_np - gt_np, axis=1).mean()

            # 提取CoT
            cot_texts = extra.get("cot", [[[]]])[0]
            if isinstance(cot_texts, np.ndarray):
                cot_texts = cot_texts.tolist()

            # 保存预测（在异步线程中实际写入）
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
            print(f"处理帧失败: {e}")

    return results


def save_results_async(results_queue):
    """
    异步保存结果到CSV
    """
    results_by_clip = defaultdict(list)

    while True:
        result = results_queue.get()

        if result is None:  # 结束标记
            break

        results_by_clip[result["clip_id"]].append(result)

        # 每积累100条保存一次
        if len(results_by_clip[result["clip_id"]]) >= 100:
            save_clip_results(
                result["clip_id"],
                results_by_clip[result["clip_id"]],
                result["result_dir"],
            )
            results_by_clip[result["clip_id"]] = []

    # 保存剩余结果
    for clip_id, clip_results in results_by_clip.items():
        if clip_results:
            # 找到result_dir
            result_dir = clip_results[0]["result_dir"]
            save_clip_results(clip_id, clip_results, result_dir)


def save_clip_results(clip_id, clip_results, result_dir):
    """
    保存单个clip的结果到CSV
    """
    try:
        result_csv = f"{result_dir}/inference_results_strict.csv"

        # 移除result_dir字段（不需要保存）
        save_results = [
            {k: v for k, v in r.items() if k != "result_dir"} for r in clip_results
        ]
        new_df = pd.DataFrame(save_results)

        if os.path.exists(result_csv):
            existing_df = pd.read_csv(result_csv)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df.drop_duplicates(subset=["frame_id"], keep="last", inplace=True)
            combined_df.sort_values("frame_id", inplace=True)
            combined_df.to_csv(result_csv, index=False)
        else:
            new_df.to_csv(result_csv, index=False)

    except Exception as e:
        print(f"保存结果失败 {clip_id}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="多GPU批量Clip推理 V4 - 异步流水线优化"
    )
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument(
        "--num_frames", type=int, default=1000, help="每clip帧数 (0=全部)"
    )
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--traj", type=int, default=1)
    parser.add_argument("--top_p", type=float, default=0.98)
    parser.add_argument("--temp", type=float, default=0.6)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument(
        "--gpus", type=str, default="0", help="GPU列表 (如: 1,2,3,5,6,7)"
    )
    parser.add_argument("--batch_size", type=int, default=4, help="批处理大小")
    args = parser.parse_args()

    # 解析GPU列表
    gpu_list = [int(g.strip()) for g in args.gpus.split(",")]

    print("=" * 70)
    print("🚀 多GPU批量Clip推理 V4 - 异步流水线优化")
    print("=" * 70)
    frame_info = "全部" if args.num_frames == 0 else str(args.num_frames)
    print(f"配置: chunk={args.chunk}, {frame_info}帧/clip, step={args.step}")
    print(f"使用GPU: {gpu_list}")
    print(f"批处理大小: {args.batch_size}")
    print()

    # 发现clips
    print("📂 扫描clips...")
    clips = discover_clips(args.chunk)
    print(f"发现 {len(clips)} 个clips")

    if not clips:
        print("❌ 没有clips可处理")
        return

    # 收集所有未处理的帧
    print("\n📋 收集所有未处理帧...")
    all_pending_frames = collect_all_pending_frames(clips, args.num_frames, args.step)

    if not all_pending_frames:
        print("✅ 所有帧已处理完成！")
        return

    print(f"总计 {len(all_pending_frames)} 帧待处理")

    # 分配给GPU
    print(f"\n📊 分配任务到 {len(gpu_list)} 个GPU...")
    tasks_per_gpu = distribute_frames_to_gpus(all_pending_frames, gpu_list)

    for gpu_id, tasks in tasks_per_gpu.items():
        print(
            f"  GPU{gpu_id}: {len(tasks)} 帧 ({len(tasks) // args.batch_size + (1 if len(tasks) % args.batch_size else 0)} 批次)"
        )

    # 准备参数
    args_dict = {
        "num_traj": args.traj,
        "top_p": args.top_p,
        "temp": args.temp,
        "max_len": args.max_len,
        "batch_size": args.batch_size,
    }

    # 启动多进程
    print(f"\n🔄 启动推理进程...")
    start_time = time.time()
    all_results = []

    # 使用spawn模式
    multiprocessing.set_start_method("spawn", force=True)

    with ProcessPoolExecutor(max_workers=len(gpu_list)) as executor:
        futures = []
        for worker_id, (gpu_id, tasks) in enumerate(tasks_per_gpu.items()):
            if tasks:  # 只启动有任务的worker
                future = executor.submit(
                    inference_worker_entry, worker_id, gpu_id, tasks, args_dict
                )
                futures.append(future)

        # 收集结果
        for future in futures:
            try:
                results = future.result()
                all_results.extend(results)
            except Exception as e:
                print(f"Worker错误: {e}")
                traceback.print_exc()

    elapsed = time.time() - start_time

    # 总结
    print("\n" + "=" * 70)
    print("📊 处理完成!")
    print("=" * 70)
    print(f"总帧数: {len(all_results)}/{len(all_pending_frames)}")
    print(f"总耗时: {elapsed / 60:.1f} 分钟")
    print(f"平均速度: {len(all_results) / elapsed:.2f} 帧/秒")

    if all_results:
        results_df = pd.DataFrame(all_results)
        print(f"\n平均ADE: {results_df['ade'].mean():.4f}m")

        # 保存汇总
        summary_dir = "/data01/vla/cot_collection"
        os.makedirs(summary_dir, exist_ok=True)
        summary = {
            "chunk": args.chunk,
            "total_frames_requested": len(all_pending_frames),
            "total_frames_processed": len(all_results),
            "elapsed_seconds": elapsed,
            "avg_speed": len(all_results) / elapsed,
            "avg_ade": float(results_df["ade"].mean()),
            "gpus_used": gpu_list,
            "batch_size": args.batch_size,
        }
        with open(f"{summary_dir}/batch_multi_gpu_v4_chunk{args.chunk}.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n✅ 汇总保存: {summary_dir}/batch_multi_gpu_v4_chunk{args.chunk}.json")


if __name__ == "__main__":
    main()
