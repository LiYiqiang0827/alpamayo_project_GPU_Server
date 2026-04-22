#!/usr/bin/env python3
"""
多GPU批量Clip推理脚本 - Alpamayo CoT收集专用
自动遍历chunk内所有clips，支持多GPU并行处理多个clips

用法:
    # 处理chunk0的所有clips
    python3 run_inference_batch_clips.py --chunk 0

    # 处理chunk0的前10个clips（用于测试）
    python3 run_inference_batch_clips.py --chunk 0 --max_clips 10

    # 处理多个chunks
    python3 run_inference_batch_clips.py --chunks 0,1

    # 指定每clip处理的帧数
    python3 run_inference_batch_clips.py --chunk 0 --num_frames 500
"""

import os
import sys

os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import subprocess
import time
import json
import copy
import traceback
import glob
from pathlib import Path
from multiprocessing import Process, Manager
from concurrent.futures import ThreadPoolExecutor

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
VRAM_SINGLE, VRAM_DOUBLE, MAX_RESTARTS = 35, 70, 3


def get_gpu_info():
    """检测GPU资源"""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
        )
        gpu_info = []
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                total_gb = float(parts[2]) / 1024
                free_gb = float(parts[3]) / 1024
                instances = (
                    2
                    if free_gb >= VRAM_DOUBLE
                    else (1 if free_gb >= VRAM_SINGLE else 0)
                )
                gpu_info.append(
                    {
                        "id": int(parts[0]),
                        "name": parts[1],
                        "total_gb": total_gb,
                        "free_gb": free_gb,
                        "instances": instances,
                    }
                )
        return gpu_info
    except Exception as e:
        print(f"GPU检测错误: {e}")
        return []


def discover_clips(chunk_id, base_dir="/data01/vla/data"):
    """自动发现指定chunk的所有clips"""
    chunk_dir = f"{base_dir}/data_sample_chunk{chunk_id}/infer"
    if not os.path.exists(chunk_dir):
        print(f"⚠️  Chunk {chunk_id} 目录不存在: {chunk_dir}")
        return []

    clips = []
    for item in os.listdir(chunk_dir):
        clip_path = os.path.join(chunk_dir, item)
        if os.path.isdir(clip_path):
            # 检查是否有预处理数据
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

    clips.sort(key=lambda x: x["clip_id"])
    return clips


def inference_worker(args_dict):
    """推理Worker - 处理分配给它的所有clips和帧"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args_dict["gpu_id"])
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from alpamayo_r1 import helper
    from scipy.spatial.transform import Rotation as R

    worker_name = f"GPU{args_dict['gpu_id']}-Inst{args_dict['instance_id']}"
    worker_tasks = args_dict["tasks"]

    print(f"[{worker_name}] 启动，处理 {len(worker_tasks)} 个任务")

    # 加载模型
    model = AlpamayoR1.from_pretrained(MODEL_PATH, dtype=torch.bfloat16).to("cuda")
    processor = helper.get_processor(model.tokenizer)

    results = []

    for task in tqdm(worker_tasks, desc=worker_name):
        clip_id = task["clip_id"]
        data_dir = task["data_dir"]
        result_dir = task["result_dir"]
        frame_tasks = task["frames"]

        # 确保结果目录存在
        os.makedirs(result_dir, exist_ok=True)

        # 加载这个clip的索引
        try:
            index_df = pd.read_csv(f"{data_dir}/inference_index_strict.csv")
        except Exception as e:
            print(f"[{worker_name}] {clip_id} 加载索引失败: {e}")
            continue

        clip_results = []

        for frame_idx in frame_tasks:
            try:
                row = index_df.iloc[frame_idx]
                frame_id = int(row["frame_id"])

                # 加载图像
                images = []
                for cam in CAMERA_ORDER:
                    for t in range(4):
                        img_idx = int(row[f"{cam}_f{t}_idx"])
                        img_path = f"{data_dir}/camera_images/{cam}/{img_idx:06d}.jpg"
                        img = Image.open(img_path).convert("RGB")
                        images.append(np.array(img))

                images = rearrange(
                    np.stack(images), "(c t) h w ch -> c t ch h w", c=4, t=4
                )

                # 加载egomotion
                history = np.load(
                    f"{data_dir}/egomotion/frame_{frame_id:06d}_history.npy",
                    allow_pickle=False,
                )
                future = np.load(
                    f"{data_dir}/egomotion/frame_{frame_id:06d}_future_gt.npy",
                    allow_pickle=False,
                )

                hist_xyz_world = history[:, 5:8]
                hist_quat = history[:, 1:5]
                future_xyz_world = future[:, 1:4]

                t0_rot_inv = R.from_quat(hist_quat[-1]).inv()
                hist_xyz = t0_rot_inv.apply(hist_xyz_world - hist_xyz_world[-1])
                hist_rot = (t0_rot_inv * R.from_quat(hist_quat)).as_matrix()
                future_xyz_local = t0_rot_inv.apply(
                    future_xyz_world - hist_xyz_world[-1]
                )

                image_frames = torch.from_numpy(images).float()
                hist_xyz_t = (
                    torch.from_numpy(hist_xyz).float().unsqueeze(0).unsqueeze(0)
                )
                hist_rot_t = (
                    torch.from_numpy(hist_rot).float().unsqueeze(0).unsqueeze(0)
                )
                future_xyz_t = (
                    torch.from_numpy(future_xyz_local).float().unsqueeze(0).unsqueeze(0)
                )

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

                torch.cuda.synchronize()
                start = time.time()

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred_xyz, pred_rot, extra = (
                        model.sample_trajectories_from_data_with_vlm_rollout(
                            data=copy.deepcopy(model_inputs),
                            top_p=args_dict["top_p"],
                            temperature=args_dict["temp"],
                            num_traj_samples=args_dict["num_traj"],
                            max_generation_length=args_dict["max_len"],
                            return_extra=True,
                        )
                    )

                torch.cuda.synchronize()
                inference_time = (time.time() - start) * 1000

                # 计算指标
                pred_np = pred_xyz.cpu().numpy()[0, 0, 0, :, :3]
                gt_np = future_xyz_t.cpu().numpy()[0, 0, :, :3]
                ade = np.linalg.norm(pred_np - gt_np, axis=1).mean()

                # 提取CoT
                cot_texts = extra.get("cot", [[[]]])[0]
                if isinstance(cot_texts, np.ndarray):
                    cot_texts = cot_texts.tolist()

                # 保存预测结果
                np.save(f"{result_dir}/pred_{frame_id:06d}.npy", pred_np)

                clip_results.append(
                    {
                        "clip_id": clip_id,
                        "frame_id": frame_id,
                        "ego_idx": int(row["ego_idx"]),
                        "ade": float(ade),
                        "inference_time_ms": round(inference_time, 1),
                        "cot_text": json.dumps(cot_texts),
                    }
                )

                # 报告进度
                args_dict["progress_queue"].put(
                    {"clip_id": clip_id, "frame_id": frame_id, "success": True}
                )

            except Exception as e:
                print(f"[{worker_name}] {clip_id} frame {frame_idx} 错误: {e}")
                args_dict["progress_queue"].put(
                    {"clip_id": clip_id, "frame_id": frame_idx, "success": False}
                )

        # 保存这个clip的结果
        if clip_results:
            # 读取已有的结果（如果有）
            result_csv = f"{result_dir}/inference_results_strict.csv"
            if os.path.exists(result_csv):
                existing_df = pd.read_csv(result_csv)
                new_df = pd.DataFrame(clip_results)
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                combined_df.drop_duplicates(
                    subset=["frame_id"], keep="last", inplace=True
                )
                combined_df.sort_values("frame_id", inplace=True)
                combined_df.to_csv(result_csv, index=False)
            else:
                pd.DataFrame(clip_results).to_csv(result_csv, index=False)

        results.extend(clip_results)

    return results


def start_worker_with_restart(args_dict, max_restarts=MAX_RESTARTS):
    """带自动重启的worker启动器"""
    worker_name = f"GPU{args_dict['gpu_id']}-Inst{args_dict['instance_id']}"

    for attempt in range(max_restarts + 1):
        if attempt > 0:
            print(f"[{worker_name}] 第{attempt}次重启...")
            time.sleep(5)

        try:
            result = inference_worker(args_dict)
            if result is not None:
                return result
        except Exception as e:
            print(f"[{worker_name}] 错误: {e}")
            traceback.print_exc()

    print(f"[{worker_name}] 达到最大重启次数，放弃")
    return []


def distribute_tasks(clips, num_frames_per_clip, step, num_workers):
    """将任务均匀分配给workers"""
    all_tasks = []

    for clip_info in clips:
        clip_id = clip_info["clip_id"]
        data_dir = clip_info["data_dir"]
        result_dir = clip_info["result_dir"]

        # 加载索引确定总帧数
        try:
            index_df = pd.read_csv(f"{data_dir}/inference_index_strict.csv")
            total_frames = len(index_df)
        except:
            print(f"⚠️  无法加载 {clip_id} 的索引，跳过")
            continue

        # 采样帧
        sampled_indices = list(range(0, total_frames, step))[:num_frames_per_clip]

        if not sampled_indices:
            continue

        all_tasks.append(
            {
                "clip_id": clip_id,
                "data_dir": data_dir,
                "result_dir": result_dir,
                "total_frames": total_frames,
                "sampled_frames": sampled_indices,
            }
        )

    if not all_tasks:
        return []

    # 将任务均匀分配给workers
    # 策略：尽量让每个worker处理完整的clips，避免频繁切换
    tasks_per_worker = [[] for _ in range(num_workers)]

    for i, task in enumerate(all_tasks):
        worker_id = i % num_workers

        # 将这个clip的所有帧分配给这个worker
        frame_tasks = task["sampled_frames"]

        tasks_per_worker[worker_id].append(
            {
                "clip_id": task["clip_id"],
                "data_dir": task["data_dir"],
                "result_dir": task["result_dir"],
                "frames": frame_tasks,
            }
        )

    return tasks_per_worker


def main():
    parser = argparse.ArgumentParser(description="多GPU批量Clip推理")
    parser.add_argument("--chunk", type=int, default=None, help="指定单个chunk (如: 0)")
    parser.add_argument(
        "--chunks", type=str, default=None, help='指定多个chunks (如: "0,1")'
    )
    parser.add_argument(
        "--max_clips", type=int, default=None, help="最多处理多少个clips (用于测试)"
    )
    parser.add_argument("--num_frames", type=int, default=1000, help="每clip处理的帧数")
    parser.add_argument("--step", type=int, default=1, help="采样步长")
    parser.add_argument("--traj", type=int, default=1, help="轨迹数量")
    parser.add_argument("--top_p", type=float, default=0.98)
    parser.add_argument("--temp", type=float, default=0.6)
    parser.add_argument("--max_len", type=int, default=256)
    args = parser.parse_args()

    # 确定要处理的chunks
    if args.chunks:
        chunks_to_process = [int(c.strip()) for c in args.chunks.split(",")]
    elif args.chunk is not None:
        chunks_to_process = [args.chunk]
    else:
        print("❌ 请指定 --chunk 或 --chunks")
        return

    print("=" * 70)
    print("🚀 多GPU批量Clip推理")
    print("=" * 70)
    print(f"配置: {args.num_frames}帧/clip, 步长={args.step}, 轨迹数={args.traj}")
    print()

    # 发现所有clips
    all_clips = []
    for chunk_id in chunks_to_process:
        clips = discover_clips(chunk_id)
        print(f"📂 Chunk {chunk_id}: 发现 {len(clips)} 个clips")
        for clip in clips:
            clip["chunk_id"] = chunk_id
        all_clips.extend(clips)

    if not all_clips:
        print("❌ 没有发现任何clips")
        return

    print(f"\n📊 总共发现 {len(all_clips)} 个clips")

    # 限制clips数量（用于测试）
    if args.max_clips and args.max_clips < len(all_clips):
        print(f"⚠️  限制处理前 {args.max_clips} 个clips")
        all_clips = all_clips[: args.max_clips]

    # 检测GPU
    print("\n📊 检测GPU资源...")
    gpu_info = get_gpu_info()
    if not gpu_info:
        print("❌ 未检测到GPU")
        return

    gpu_assignments = []
    for gpu in gpu_info:
        for i in range(gpu["instances"]):
            gpu_assignments.append((gpu["id"], i))
        print(f"  GPU {gpu['id']}: {gpu['name']} - {gpu['instances']} 实例")

    total_workers = len(gpu_assignments)
    print(f"\n✅ 总worker数: {total_workers}")

    # 分配任务
    print("\n📋 分配任务...")
    worker_tasks = distribute_tasks(
        all_clips, args.num_frames, args.step, total_workers
    )

    for i, tasks in enumerate(worker_tasks):
        gpu_id, inst_id = gpu_assignments[i]
        total_frames = sum(len(t["frames"]) for t in tasks)
        print(
            f"  Worker {i} (GPU{gpu_id}-Inst{inst_id}): {len(tasks)} clips, {total_frames} 帧"
        )

    # 准备worker配置
    print("\n🔄 启动推理进程...")
    manager = Manager()
    progress_queue = manager.Queue()

    worker_configs = []
    for i, (gpu_id, inst_id) in enumerate(gpu_assignments):
        worker_configs.append(
            {
                "gpu_id": gpu_id,
                "instance_id": inst_id,
                "tasks": worker_tasks[i],
                "num_traj": args.traj,
                "top_p": args.top_p,
                "temp": args.temp,
                "max_len": args.max_len,
                "progress_queue": progress_queue,
                "worker_id": f"GPU{gpu_id}-Inst{inst_id}",
            }
        )

    # 启动workers
    start_time = time.time()
    total_tasks = sum(len(t["frames"]) for tasks in worker_tasks for t in tasks)

    all_results = []
    completed = failed = 0

    with ThreadPoolExecutor(max_workers=total_workers) as executor:
        futures = {
            executor.submit(start_worker_with_restart, cfg): cfg["worker_id"]
            for cfg in worker_configs
        }

        with tqdm(total=total_tasks, desc="总体进度") as pbar:
            # 收集结果
            for future in futures:
                try:
                    worker_results = future.result()
                    all_results.extend(worker_results)
                except Exception as e:
                    print(f"Worker错误: {e}")

            # 更新进度
            while completed + failed < total_tasks:
                try:
                    msg = progress_queue.get(timeout=1.0)
                    if msg["success"]:
                        completed += 1
                    else:
                        failed += 1
                    pbar.update(1)
                except:
                    if all(f.done() for f in futures):
                        break

    elapsed = time.time() - start_time

    # 统计结果
    print("\n" + "=" * 70)
    print("📊 处理完成!")
    print("=" * 70)
    print(f"总任务: {total_tasks}")
    print(f"成功: {completed}")
    print(f"失败: {failed}")
    print(f"总耗时: {elapsed / 60:.1f} 分钟")
    print(f"平均速度: {total_tasks / elapsed:.1f} 帧/秒")

    if all_results:
        results_df = pd.DataFrame(all_results)
        avg_ade = results_df["ade"].mean()
        avg_time = results_df["inference_time_ms"].mean()

        print(f"\n平均ADE: {avg_ade:.4f}m")
        print(f"平均推理时间: {avg_time:.1f}ms")

        # 保存汇总
        summary = {
            "chunks": chunks_to_process,
            "clips_processed": len(all_clips),
            "total_frames": total_tasks,
            "successful": completed,
            "failed": failed,
            "elapsed_seconds": elapsed,
            "avg_ade": float(avg_ade),
            "avg_inference_time_ms": float(avg_time),
        }

        summary_dir = "/data01/vla/cot_collection"
        os.makedirs(summary_dir, exist_ok=True)
        summary_file = f"{summary_dir}/batch_inference_summary.json"

        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n✅ 汇总保存: {summary_file}")

        # 显示各clip的统计
        print("\n各Clip统计:")
        for clip_id in results_df["clip_id"].unique():
            clip_df = results_df[results_df["clip_id"] == clip_id]
            print(
                f"  {clip_id[:8]}...: {len(clip_df)} 帧, "
                f"ADE={clip_df['ade'].mean():.2f}m, "
                f"Time={clip_df['inference_time_ms'].mean():.0f}ms"
            )


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except:
        pass
    main()
