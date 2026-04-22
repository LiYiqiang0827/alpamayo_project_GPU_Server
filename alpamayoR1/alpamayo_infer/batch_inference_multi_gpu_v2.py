#!/usr/bin/env python3
"""
多GPU批量Clip推理 V2 - 支持指定GPU列表
修复了模型加载错误，支持断点续跑

用法:
    python3 batch_inference_multi_gpu_v2.py --chunk 0 --num_frames 0 --gpus 1,2,3,5,6,7
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
import multiprocessing
from multiprocessing import Process, Manager
from concurrent.futures import ProcessPoolExecutor

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


def get_gpu_info():
    """获取GPU信息"""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
        )
        gpu_info = []
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                gpu_info.append(
                    {
                        "id": int(parts[0]),
                        "name": parts[1],
                        "total_gb": float(parts[1]) / 1024,
                        "free_gb": float(parts[2]) / 1024,
                    }
                )
        return gpu_info
    except:
        return []


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


def check_missing_frames(clip_info, num_frames, step):
    """检查clip中未处理的帧"""
    data_dir = clip_info["data_dir"]
    result_dir = clip_info["result_dir"]

    try:
        index_df = pd.read_csv(f"{data_dir}/inference_index_strict.csv")
        total_frames = len(index_df)

        # 确定要处理的帧索引
        if num_frames == 0:
            target_indices = list(range(0, total_frames, step))
        else:
            target_indices = list(range(0, total_frames, step))[:num_frames]

        # 检查哪些帧已经处理过
        missing_indices = []
        for idx in target_indices:
            frame_id = int(index_df.iloc[idx]["frame_id"])
            pred_file = f"{result_dir}/pred_{frame_id:06d}.npy"
            if not os.path.exists(pred_file):
                missing_indices.append(idx)

        return missing_indices
    except Exception as e:
        print(f"检查 {clip_info['clip_id']} 失败: {e}")
        return []


def inference_worker_entry(worker_id, gpu_id, tasks, args_dict):
    """Worker入口 - 处理分配给它的所有clips"""
    # 设置GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # 导入必须在设置GPU后
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from alpamayo_r1 import helper
    from scipy.spatial.transform import Rotation as R

    worker_name = f"GPU{gpu_id}-Worker{worker_id}"

    print(f"[{worker_name}] 启动，处理 {len(tasks)} 个clips")

    try:
        # 加载模型 - 使用不触发meta tensor错误的方式
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

    all_results = []

    for task in tasks:
        clip_id = task["clip_id"]
        data_dir = task["data_dir"]
        result_dir = task["result_dir"]
        frame_indices = task["frame_indices"]

        if not frame_indices:
            print(f"[{worker_name}] {clip_id}: 没有需要处理的帧")
            continue

        os.makedirs(result_dir, exist_ok=True)

        try:
            index_df = pd.read_csv(f"{data_dir}/inference_index_strict.csv")
        except Exception as e:
            print(f"[{worker_name}] {clip_id} 加载索引失败: {e}")
            continue

        print(f"[{worker_name}] {clip_id}: 处理 {len(frame_indices)} 帧")

        clip_results = []

        for idx in tqdm(
            frame_indices, desc=f"{worker_name}-{clip_id[:8]}", leave=False
        ):
            try:
                row = index_df.iloc[idx]
                frame_id = int(row["frame_id"])

                # 加载图像
                images = []
                for cam in CAMERA_ORDER:
                    for t in range(4):
                        frame_idx = int(row[f"{cam}_f{t}_idx"])
                        img_path = f"{data_dir}/camera_images/{cam}/{frame_idx:06d}.jpg"
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

                hist_xyz_world, hist_quat = history[:, 5:8], history[:, 1:5]
                future_xyz_world = future[:, 1:4]

                t0_rot_inv = R.from_quat(hist_quat[-1]).inv()
                hist_xyz = t0_rot_inv.apply(hist_xyz_world - hist_xyz_world[-1])
                hist_rot = (t0_rot_inv * R.from_quat(hist_quat)).as_matrix()
                future_xyz = t0_rot_inv.apply(future_xyz_world - hist_xyz_world[-1])

                image_frames = torch.from_numpy(images).float()
                hist_xyz_t = (
                    torch.from_numpy(hist_xyz).float().unsqueeze(0).unsqueeze(0)
                )
                hist_rot_t = (
                    torch.from_numpy(hist_rot).float().unsqueeze(0).unsqueeze(0)
                )
                future_xyz_t = (
                    torch.from_numpy(future_xyz).float().unsqueeze(0).unsqueeze(0)
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
                gt_np = future_xyz_t.cpu().numpy()[0, 0, :, :3]
                ade = np.linalg.norm(pred_np - gt_np, axis=1).mean()

                # 提取CoT
                coc_texts = extra.get("cot", [[[]]])[0]
                if isinstance(coc_texts, np.ndarray):
                    coc_texts = coc_texts.tolist()

                # 保存预测
                np.save(f"{result_dir}/pred_{frame_id:06d}.npy", pred_np)

                clip_results.append(
                    {
                        "clip_id": clip_id,
                        "frame_id": frame_id,
                        "ego_idx": int(row["ego_idx"]),
                        "ade": float(ade),
                        "inference_time_ms": 0,  # 简化，不统计时间了
                        "cot_text": json.dumps(coc_texts),
                    }
                )

            except Exception as e:
                print(f"[{worker_name}] {clip_id} frame {frame_id} 错误: {e}")

        # 保存这个clip的结果
        if clip_results:
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

            print(f"[{worker_name}] {clip_id}: 完成 {len(clip_results)} 帧")
            all_results.extend(clip_results)

    print(f"[{worker_name}] 完成")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="多GPU批量Clip推理 V2")
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
    args = parser.parse_args()

    # 解析GPU列表
    gpu_list = [int(g.strip()) for g in args.gpus.split(",")]

    print("=" * 70)
    print("🚀 多GPU批量Clip推理 V2")
    print("=" * 70)
    frame_info = "全部" if args.num_frames == 0 else str(args.num_frames)
    print(f"配置: chunk={args.chunk}, {frame_info}帧/clip, step={args.step}")
    print(f"使用GPU: {gpu_list}")
    print()

    # 发现clips
    print("📂 扫描clips...")
    clips = discover_clips(args.chunk)
    print(f"发现 {len(clips)} 个clips")

    if not clips:
        print("❌ 没有clips可处理")
        return

    # 检查每个clip的未处理帧
    print("\n📋 检查未处理帧...")
    clip_tasks = []
    for clip in clips:
        missing = check_missing_frames(clip, args.num_frames, args.step)
        total = len(pd.read_csv(f"{clip['data_dir']}/inference_index_strict.csv"))
        processed = total - len(missing) if args.num_frames == 0 else len(missing)
        print(
            f"  {clip['clip_id'][:8]}...: 已处理{total - len(missing)}/{total}, 剩余{len(missing)}帧"
        )
        if missing:
            clip_tasks.append(
                {
                    "clip_id": clip["clip_id"],
                    "data_dir": clip["data_dir"],
                    "result_dir": clip["result_dir"],
                    "frame_indices": missing,
                }
            )

    if not clip_tasks:
        print("\n✅ 所有clips已完成！")
        return

    total_remaining = sum(len(t["frame_indices"]) for t in clip_tasks)
    print(f"\n📊 总计: {len(clip_tasks)} 个clips, {total_remaining} 帧待处理")

    # 分配任务给workers
    print(f"\n📋 分配任务到 {len(gpu_list)} 个GPU...")
    tasks_per_gpu = [[] for _ in gpu_list]

    for i, task in enumerate(clip_tasks):
        gpu_idx = i % len(gpu_list)
        tasks_per_gpu[gpu_idx].append(task)

    for i, (gpu_id, tasks) in enumerate(zip(gpu_list, tasks_per_gpu)):
        total_frames = sum(len(t["frame_indices"]) for t in tasks)
        print(f"  GPU{gpu_id}: {len(tasks)} clips, {total_frames} 帧")

    # 准备参数
    args_dict = {
        "num_traj": args.traj,
        "top_p": args.top_p,
        "temp": args.temp,
        "max_len": args.max_len,
    }

    # 启动多进程
    print(f"\n🔄 启动推理进程...")
    start_time = time.time()
    all_results = []

    # 使用spawn模式
    multiprocessing.set_start_method("spawn", force=True)

    with ProcessPoolExecutor(max_workers=len(gpu_list)) as executor:
        futures = []
        for worker_id, (gpu_id, tasks) in enumerate(zip(gpu_list, tasks_per_gpu)):
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
    print(f"总clips: {len(clip_tasks)}")
    print(f"总帧数: {len(all_results)}")
    print(f"总耗时: {elapsed / 60:.1f} 分钟")

    if all_results:
        results_df = pd.DataFrame(all_results)
        print(f"\n平均ADE: {results_df['ade'].mean():.4f}m")

        # 保存汇总
        summary_dir = "/data01/vla/cot_collection"
        os.makedirs(summary_dir, exist_ok=True)
        summary = {
            "chunk": args.chunk,
            "clips_processed": len(clip_tasks),
            "total_frames": len(all_results),
            "elapsed_seconds": elapsed,
            "avg_ade": float(results_df["ade"].mean()),
            "gpus_used": gpu_list,
        }
        with open(f"{summary_dir}/batch_multi_gpu_v2_chunk{args.chunk}.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n✅ 汇总保存: {summary_dir}/batch_multi_gpu_v2_chunk{args.chunk}.json")


if __name__ == "__main__":
    main()
