#!/usr/bin/env python3
"""
多GPU批量Clip推理 V6.4 - 完整双缓冲优化版
关键优化：
1. 输入输出双缓冲：完全重叠计算和传输
2. 独立锁保护：每个槽位有自己的锁，GPU和CPU可并行操作不同槽位
3. 状态机管理：Empty->Filling->Ready->Processing->Done->Empty
4. 预分配内存：避免动态分配开销
5. 异步结果保存

用法:
    python3 batch_inference_multi_gpu_v6.py --chunk 0 --num_frames 0 --gpus 1,2,3,5,6,7 --batch_size 16
"""

import os
import sys

os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import time
import json
import traceback
import multiprocessing
import threading
from threading import Lock, Condition
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

sys.path.insert(0, "/home/user/mikelee/alpamayo-main/src")

MODEL_PATH = "/data01/vla/models--nvidia--Alpamayo-R1-10B/snapshots/22fab1399111f50b52bfbe5d8b809f39bd4c2fe1"
CAMERA_ORDER = [
    "camera_cross_left_120fov",
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_front_tele_30fov",
]


class SlotState:
    """槽位状态机"""

    EMPTY = 0  # 空闲
    CPU_FILLING = 1  # CPU正在填充输入
    INPUT_READY = 2  # 输入就绪，GPU可以读取
    GPU_PROCESSING = 3  # GPU正在处理
    OUTPUT_READY = 4  # 输出就绪，CPU可以读取
    CPU_READING = 5  # CPU正在读取输出


class BufferSlot:
    """
    缓冲槽位 - 包含输入+输出+锁+状态
    """

    def __init__(self, batch_size, device="cuda"):
        self.device = device
        self.batch_size = batch_size

        # 输入数据
        self.input_data = {
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

        # 输出数据
        self.output_data = {
            "pred_xyz": [None] * batch_size,
            "cot_texts": [None] * batch_size,
            "ades": [0.0] * batch_size,
        }

        # 同步原语
        self.lock = Lock()
        self.condition = Condition(self.lock)
        self.state = SlotState.EMPTY

    def cpu_begin_fill(self):
        """CPU开始填充输入"""
        with self.condition:
            while self.state not in [SlotState.EMPTY, SlotState.CPU_READING]:
                self.condition.wait()
            self.state = SlotState.CPU_FILLING
            return self.input_data

    def cpu_end_fill(self):
        """CPU完成填充"""
        with self.condition:
            self.state = SlotState.INPUT_READY
            self.condition.notify_all()

    def gpu_begin_process(self):
        """GPU开始处理"""
        with self.condition:
            while self.state != SlotState.INPUT_READY:
                self.condition.wait()
            self.state = SlotState.GPU_PROCESSING
            return self.input_data

    def gpu_end_process(self):
        """GPU完成处理"""
        with self.condition:
            self.state = SlotState.OUTPUT_READY
            self.condition.notify_all()

    def cpu_begin_read(self):
        """CPU开始读取输出"""
        with self.condition:
            while self.state != SlotState.OUTPUT_READY:
                self.condition.wait()
            self.state = SlotState.CPU_READING
            return self.output_data, self.input_data

    def cpu_end_read(self):
        """CPU完成读取"""
        with self.condition:
            self.state = SlotState.EMPTY
            self.condition.notify_all()


class DoubleBufferManager:
    """双缓冲管理器"""

    def __init__(self, batch_size, device="cuda"):
        self.slot_a = BufferSlot(batch_size, device)
        self.slot_b = BufferSlot(batch_size, device)

        self.cpu_slot = self.slot_a
        self.gpu_slot = self.slot_b

    def get_cpu_slot(self):
        return self.cpu_slot

    def get_gpu_slot(self):
        return self.gpu_slot

    def try_swap(self):
        """尝试交换槽位"""
        # 快速检查
        cpu_ready = self.cpu_slot.state == SlotState.INPUT_READY
        gpu_done = self.gpu_slot.state == SlotState.OUTPUT_READY

        if not (cpu_ready and gpu_done):
            return False

        # 获取两把锁
        a_acquired = self.slot_a.lock.acquire(blocking=False)
        if not a_acquired:
            return False

        b_acquired = self.slot_b.lock.acquire(blocking=False)
        if not b_acquired:
            self.slot_a.lock.release()
            return False

        try:
            # 再次检查并交换
            if (
                self.cpu_slot.state == SlotState.INPUT_READY
                and self.gpu_slot.state == SlotState.OUTPUT_READY
            ):
                self.cpu_slot, self.gpu_slot = self.gpu_slot, self.cpu_slot
                return True
            return False
        finally:
            self.slot_a.lock.release()
            self.slot_b.lock.release()


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
    """收集所有未处理的帧"""
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

            result_csv = f"{result_dir}/inference_results_strict.csv"
            completed_ids = set()
            if os.path.exists(result_csv):
                result_df = pd.read_csv(result_csv)
                completed_ids = set(result_df.frame_id.values)

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
            print(f"检查 {clip_id} 失败: {e}")

    return all_pending_frames


def load_single_frame_safe(task, clip_cache):
    """安全加载单帧数据"""
    from scipy.spatial.transform import Rotation as R

    try:
        clip_id = task["clip_id"]
        data_dir = task["data_dir"]
        frame_idx = task["frame_idx"]
        frame_id = task["frame_id"]

        if clip_id not in clip_cache:
            clip_cache[clip_id] = pd.read_csv(f"{data_dir}/inference_index_strict.csv")
        index_df = clip_cache[clip_id]

        row = index_df.iloc[frame_idx]

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

                if len(img_array.shape) != 3 or img_array.shape[2] != 3:
                    print(f"警告: 图像shape异常 {img_path}")
                    return None

                images.append(img_array)

        images_array = np.stack(images).astype(np.float32)
        images = rearrange(images_array, "(c t) h w ch -> c t ch h w", c=4, t=4)

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
        future_xyz = t0_rot_inv.apply(future_xyz_world - hist_xyz_world[-1])

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
        print(f"加载帧失败: {e}")
        return None


class DataPrefetcher:
    """CPU端数据预取器"""

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
        for i in range(0, len(self.frame_tasks), self.batch_size):
            if self.stop_flag.is_set():
                break

            batch_tasks = self.frame_tasks[i : i + self.batch_size]

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


def gpu_worker_with_double_buffer(
    worker_id, gpu_id, frame_tasks, batch_size, args_dict
):
    """GPU工作进程 - 使用完整双缓冲"""
    from alpamayo_r1 import helper

    worker_name = f"GPU-{gpu_id}-Worker-{worker_id}"
    print(f"[{worker_name}] 启动，处理 {len(frame_tasks)} 帧")

    try:
        # 设置GPU和离线模式
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        torch.cuda.init()
        device = "cuda:0"
        torch.cuda.set_device(0)

        print(f"[{worker_name}] 加载模型...")
        model = helper.get_model(MODEL_PATH)
        model.eval()
        model = model.to(device)
        processor = helper.get_processor(model.tokenizer)
        print(f"[{worker_name}] 模型加载完成")

        # 创建双缓冲管理器
        dbm = DoubleBufferManager(batch_size, device)

        # 数据预取器
        prefetcher = DataPrefetcher(
            frame_tasks, batch_size, num_prefetch=3, num_workers=4
        )

        # 结果队列和保存线程
        results_queue = Queue()

        def save_async():
            results_by_clip = defaultdict(list)
            while True:
                try:
                    result = results_queue.get(timeout=1)
                    if result is None:
                        break
                    results_by_clip[result["clip_id"]].append(result)
                    if len(results_by_clip[result["clip_id"]]) >= 50:
                        save_clip_results(
                            result["clip_id"],
                            results_by_clip[result["clip_id"]],
                            result["result_dir"],
                        )
                        results_by_clip[result["clip_id"]] = []
                except:
                    continue
            for clip_id, clip_results in results_by_clip.items():
                if clip_results:
                    save_clip_results(
                        clip_id, clip_results, clip_results[0]["result_dir"]
                    )

        save_thread = threading.Thread(target=save_async)
        save_thread.start()

        # 主循环
        results = []
        processed_count = 0
        failed_count = 0

        with tqdm(total=len(frame_tasks), desc=worker_name, leave=False) as pbar:
            batch_data = prefetcher.get_batch()

            while batch_data is not None:
                if not batch_data:
                    failed_count += batch_size
                    batch_data = prefetcher.get_batch()
                    continue

                # === 使用双缓冲处理 ===
                # 1. CPU填充输入槽位
                cpu_slot = dbm.get_cpu_slot()
                input_buf = cpu_slot.cpu_begin_fill()

                for i, data in enumerate(batch_data):
                    if data is not None and i < batch_size:
                        input_buf["images"][i].copy_(data["image_frames"])
                        input_buf["hist_xyz"][i].copy_(data["hist_xyz"].squeeze())
                        input_buf["hist_rot"][i].copy_(data["hist_rot"].squeeze())
                        input_buf["future_xyz"][i].copy_(data["future_xyz"].squeeze())
                        input_buf["tasks"][i] = data["task"]
                        input_buf["valid_mask"][i] = True

                cpu_slot.cpu_end_fill()

                # 2. GPU处理槽位（并行）
                gpu_slot = dbm.get_gpu_slot()
                try:
                    gpu_input = gpu_slot.gpu_begin_process()

                    for i in range(batch_size):
                        if not gpu_input["valid_mask"][i]:
                            continue

                        task = gpu_input["tasks"][i]
                        result_dir = task["result_dir"]
                        frame_id = task["frame_id"]
                        clip_id = task["clip_id"]

                        os.makedirs(result_dir, exist_ok=True)

                        messages = helper.create_message(
                            gpu_input["images"][i].flatten(0, 1)
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
                            "ego_history_xyz": gpu_input["hist_xyz"][i]
                            .unsqueeze(0)
                            .unsqueeze(0),
                            "ego_history_rot": gpu_input["hist_rot"][i]
                            .unsqueeze(0)
                            .unsqueeze(0),
                        }
                        model_inputs = helper.to_device(model_inputs, device)

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

                        pred_np = pred_xyz.cpu().numpy()[0, 0, 0, :, :3]
                        gt_np = gpu_input["future_xyz"][i].cpu().numpy()[0, 0, :, :3]
                        ade = np.linalg.norm(pred_np - gt_np, axis=1).mean()

                        cot_texts = extra.get("cot", [[[]]])[0]
                        if isinstance(cot_texts, np.ndarray):
                            cot_texts = cot_texts.tolist()

                        gpu_slot.output_data["pred_xyz"][i] = pred_np
                        gpu_slot.output_data["cot_texts"][i] = cot_texts
                        gpu_slot.output_data["ades"][i] = float(ade)

                    gpu_slot.gpu_end_process()
                    processed_count += sum(gpu_input["valid_mask"].cpu().numpy())

                except Exception as e:
                    print(f"[{worker_name}] GPU处理失败: {e}")
                    failed_count += len(batch_data)

                # 3. CPU读取输出（上一个batch的结果）
                try:
                    prev_slot = dbm.get_gpu_slot()
                    if prev_slot.state == SlotState.OUTPUT_READY:
                        output_buf, input_buf = prev_slot.cpu_begin_read()

                        for i in range(batch_size):
                            if input_buf["valid_mask"][i]:
                                result = {
                                    "clip_id": input_buf["tasks"][i]["clip_id"],
                                    "frame_id": input_buf["tasks"][i]["frame_id"],
                                    "ade": output_buf["ades"][i],
                                    "cot_text": json.dumps(output_buf["cot_texts"][i]),
                                    "result_dir": input_buf["tasks"][i]["result_dir"],
                                }
                                results.append(result)
                                results_queue.put(result)

                                # 保存pred文件
                                np.save(
                                    f"{result['result_dir']}/pred_{result['frame_id']:06d}.npy",
                                    output_buf["pred_xyz"][i],
                                )

                        prev_slot.cpu_end_read()
                except:
                    pass

                # 4. 尝试交换
                swapped = dbm.try_swap()

                # 下一批
                batch_data = prefetcher.get_batch()
                pbar.update(len(batch_data) if batch_data else 0)

        # 清理
        prefetcher.stop()
        results_queue.put(None)
        save_thread.join(timeout=10)

        print(f"[{worker_name}] 完成: {processed_count}成功, {failed_count}失败")
        return results

    except Exception as e:
        print(f"[{worker_name}] 错误: {e}")
        traceback.print_exc()
        return []


def save_clip_results(clip_id, clip_results, result_dir):
    """保存结果到CSV"""
    try:
        result_csv = f"{result_dir}/inference_results_strict.csv"
        save_results = [
            {k: v for k, v in r.items() if k not in ["result_dir"]}
            for r in clip_results
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


def distribute_frames_to_gpus(all_frames, gpu_list):
    """分配帧到GPU"""
    tasks_per_gpu = {gpu_id: [] for gpu_id in gpu_list}
    for i, frame_info in enumerate(all_frames):
        gpu_idx = i % len(gpu_list)
        gpu_id = gpu_list[gpu_idx]
        tasks_per_gpu[gpu_id].append(frame_info)
    return tasks_per_gpu


def main():
    parser = argparse.ArgumentParser(
        description="多GPU批量Clip推理 V6.4 - 双缓冲优化版"
    )
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=0, help="每clip帧数 (0=全部)")
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--traj", type=int, default=1)
    parser.add_argument("--top_p", type=float, default=0.98)
    parser.add_argument("--temp", type=float, default=0.6)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--gpus", type=str, default="1,2,3,5,6,7", help="GPU ID列表")
    parser.add_argument("--batch_size", type=int, default=16, help="批大小")
    parser.add_argument("--workers_per_gpu", type=int, default=1, help="每GPU进程数")

    args = parser.parse_args()

    print(f"🚀 多GPU批量推理 V6.4 (完整双缓冲优化)")
    print(f"   Chunk: {args.chunk}")
    print(f"   GPUs: {args.gpus}")
    print(f"   Batch size: {args.batch_size}")

    gpu_list = [int(x) for x in args.gpus.split(",")]

    # 发现clips
    print(f"\n📁 发现clips...")
    clips = discover_clips(args.chunk)
    print(f"   找到 {len(clips)} 个已预处理clips")

    # 收集待处理帧
    print(f"\n🔍 检查未处理帧...")
    all_pending_frames = collect_all_pending_frames(clips, args.num_frames, args.step)
    print(f"   待处理帧总数: {len(all_pending_frames)}")

    if len(all_pending_frames) == 0:
        print("✅ 所有帧已处理完成！")
        return

    # 分配到GPU
    print(f"\n📊 分配任务到GPU...")
    tasks_per_gpu = distribute_frames_to_gpus(all_pending_frames, gpu_list)
    for gpu_id, tasks in tasks_per_gpu.items():
        print(f"   GPU {gpu_id}: {len(tasks)} 帧")

    args_dict = {
        "top_p": args.top_p,
        "temp": args.temp,
        "num_traj": args.traj,
        "max_len": args.max_len,
    }

    # 启动
    print(f"\n🎬 启动GPU工作进程...")
    start_time = time.time()

    all_results = []
    with ProcessPoolExecutor(max_workers=len(gpu_list)) as executor:
        futures = []
        for gpu_id in gpu_list:
            if len(tasks_per_gpu[gpu_id]) > 0:
                future = executor.submit(
                    gpu_worker_with_double_buffer,
                    0,
                    gpu_id,
                    tasks_per_gpu[gpu_id],
                    args.batch_size,
                    args_dict,
                )
                futures.append(future)

        for future in futures:
            try:
                results = future.result()
                all_results.extend(results)
            except Exception as e:
                print(f"收集结果失败: {e}")

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"✅ 全部完成!")
    print(f"   总帧数: {len(all_pending_frames)}")
    print(f"   成功: {len(all_results)}")
    print(f"   失败: {len(all_pending_frames) - len(all_results)}")
    print(f"   耗时: {elapsed:.1f}s")
    print(f"   速度: {len(all_results) / elapsed:.2f} fps")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
