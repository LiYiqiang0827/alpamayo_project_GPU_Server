#!/usr/bin/env python3
"""
多GPU批量推理脚本 - Alpamayo 1.5
支持多GPU自动检测、灵活的任务分配、崩溃重启

输入参数:
    --gpu       GPU ID列表,逗号分隔,默认 "1" (如 "1,2,3")
    --chunk     data_sample chunk编号,默认 0
    --clip      clip ID,逗号分隔或 "all"(默认 all)
    --num_frames 帧采样比例 0.0~1.0,0表示全部帧,默认 0
    --step     帧采样步长,必须>1,默认 1
    --traj     轨迹数量,默认 1

用法:
    python3 run_inference_mulGPU_1_5.py --clip <clip_id> --num_frames 0.5 --step 5
"""
import os
import sys
import argparse
import subprocess
import datetime
from tqdm import tqdm
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd

# ========================
# 常量定义
# ========================
BASE_DATA_DIR = "/data01/mikelee/data/data_sample_chunk"
CAMERA_ORDER = [
    "camera_cross_left_120fov",
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_front_tele_30fov",
]
NUM_CAMERAS = 4
NUM_FRAMES_PER_CAMERA = 4  # 每相机4帧历史


# ========================
# 第一步:参数解析和验证
# ========================
def parse_and_validate_args() -> dict:
    """解析并验证输入参数,返回参数字典"""

    parser = argparse.ArgumentParser(
        description="多GPU批量推理 - Alpamayo 1.5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
参数说明:
  --gpu       GPU ID列表,逗号分隔,默认 "1"
              示例: --gpu 1 或 --gpu 1,2,3
  --chunk     data_sample chunk编号,默认 0
  --clip      clip ID,逗号分隔或 "all"(默认 all)
              示例: --clip 01d3588e... 或 --clip 01d3588e...,038678de...
  --num_frames 帧采样比例 0.0~1.0,0表示全部帧(100%%)
              示例: 0.5 表示前50%%帧, 0.1 表示前10%%帧
  --step     帧采样步长,必须>=1,默认 1
              step=1 表示每帧都处理,step=5 表示每隔4帧取1帧
  --traj     轨迹数量,默认 1
        """
    )

    parser.add_argument(
        "--gpu",
        type=str,
        default="1",
        help="GPU ID列表,逗号分隔 (默认: 1)"
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=0,
        help="data_sample chunk编号 (默认: 0)"
    )
    parser.add_argument(
        "--clip",
        type=str,
        default="all",
        help="clip ID,逗号分隔或 'all' (默认: all)"
    )
    parser.add_argument(
        "--num_frames",
        type=float,
        default=0,
        help="帧采样比例 0.0~1.0,0表示全部帧 (默认: 0)"
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="帧采样步长,必须>=1 (默认: 1)"
    )
    parser.add_argument(
        "--traj",
        type=int,
        default=1,
        help="轨迹数量 (默认: 1)"
    )

    args = parser.parse_args()

    # ---------- GPU ID 解析和验证 ----------
    gpu_ids = []
    for part in args.gpu.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            gpu_id = int(part)
            if gpu_id < 0:
                raise ValueError(f"GPU ID 必须 >= 0,实际: {gpu_id}")
            gpu_ids.append(gpu_id)
        except ValueError as e:
            raise ValueError(f"GPU ID 无效: '{part}' - {e}")

    if not gpu_ids:
        raise ValueError("必须至少指定一个 GPU ID")

    # ---------- chunk 验证 ----------
    chunk = args.chunk
    if chunk < 0:
        raise ValueError(f"chunk 必须 >= 0,实际: {chunk}")

    # ---------- num_frames 验证 ----------
    num_frames = args.num_frames
    if not (0 <= num_frames <= 1):
        raise ValueError(f"num_frames 必须在 0~1 范围内,实际: {num_frames}")

    # ---------- step 验证 ----------
    step = args.step
    if step < 1:
        raise ValueError(f"step 必须 >= 1,实际: {step}")

    # ---------- traj 验证 ----------
    traj = args.traj
    if traj < 1:
        raise ValueError(f"traj 必须 >= 1,实际: {traj}")

    # ---------- clip 解析 ----------
    clip_input = args.clip.strip()
    if not clip_input:
        raise ValueError("clip 不能为空")

    use_all_clips = (clip_input.lower() == "all")
    clip_list = None
    if not use_all_clips:
        clip_list = [c.strip() for c in clip_input.split(",") if c.strip()]
        if not clip_list:
            raise ValueError("clip 列表为空")

    # ---------- 构建参数字典 ----------
    params = {
        "gpu_ids": gpu_ids,
        "chunk": chunk,
        "use_all_clips": use_all_clips,
        "clip_list": clip_list,
        "num_frames_ratio": num_frames,  # 0.0~1.0, 0表示全部
        "step": step,
        "traj": traj,
    }

    return params


# ========================
# 第二步:构建任务 DataFrame
# ========================
def build_task_dataframe(params: dict) -> pd.DataFrame:
    """根据参数构建所有待处理任务的 DataFrame"""

    chunk = params["chunk"]
    base_dir = f"{BASE_DATA_DIR}{chunk}/infer"

    # ---------- 第一步:确定 clip 列表 ----------
    if params["use_all_clips"]:
        # 遍历 chunk 文件夹下所有 clip
        clip_dirs = sorted([
            d for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d))
        ])
        print(f"  找到 {len(clip_dirs)} 个 clips (all)")
    else:
        clip_dirs = params["clip_list"]
        # 验证 clip 是否存在
        missing = []
        for clip_id in clip_dirs:
            clip_path = os.path.join(base_dir, clip_id)
            if not os.path.isdir(clip_path):
                missing.append(clip_id)
        if missing:
            raise FileNotFoundError(f"Clip 不存在: {missing}")
        print(f"  指定了 {len(clip_dirs)} 个 clips")

    # ---------- 第二步:收集每个 clip 的帧 ----------
    all_tasks = []

    for clip_id in clip_dirs:
        clip_data_dir = os.path.join(base_dir, clip_id, "data")
        index_file = os.path.join(clip_data_dir, "inference_index_strict.csv")

        if not os.path.exists(index_file):
            print(f"  警告: {clip_id} 没有 inference_index_strict.csv,跳过")
            continue

        # 读取索引文件
        index_df = pd.read_csv(index_file)
        total_frames = len(index_df)

        # ---------- 确定要处理的帧 ----------
        # step: 每隔 step-1 帧取一帧 (step=1 表示全部)
        # step=5: 取 0, 5, 10, 15, ...
        sampled_indices = list(range(0, total_frames, params["step"]))

        # num_frames_ratio: 取前百分之多少
        # 0 表示全部,其他如 0.5 表示前 50%
        if params["num_frames_ratio"] > 0:
            max_indices = max(1, int(total_frames * params["num_frames_ratio"]))
            # 四舍五入到 step 的整数倍
            max_indices = ((max_indices + params["step"] - 1) // params["step"]) * params["step"]
            sampled_indices = sampled_indices[:max_indices]

        print(f"  {clip_id}: 总帧数={total_frames}, 采样后={len(sampled_indices)} (step={params['step']}, ratio={params['num_frames_ratio']})")

        # ---------- 为每个采样帧构建记录 ----------
        for idx in sampled_indices:
            row = index_df.iloc[idx]
            frame_id = int(row["frame_id"])

            # 构建 16 张图片路径
            image_paths = {}
            for cam in CAMERA_ORDER:
                for t in range(NUM_FRAMES_PER_CAMERA):
                    col_idx = f"{cam}_f{t}_idx"
                    col_ts = f"{cam}_f{t}_ts"
                    frame_idx = int(row[col_idx])
                    img_path = os.path.join(
                        clip_data_dir,
                        "camera_images",
                        cam,
                        f"{frame_idx:06d}.jpg"
                    )
                    # 用 (camera, t) 作为 key
                    image_paths[(cam, t)] = img_path

            # 历史和未来轨迹路径
            history_path = os.path.join(
                clip_data_dir,
                "egomotion",
                f"frame_{frame_id:06d}_history.npy"
            )
            future_path = os.path.join(
                clip_data_dir,
                "egomotion",
                f"frame_{frame_id:06d}_future_gt.npy"
            )

            all_tasks.append({
                "chunk_id": chunk,
                "clip_id": clip_id,
                "frame_idx": idx,
                "frame_id": frame_id,
                "image_paths": image_paths,
                "history_path": history_path,
                "future_path": future_path,
            })

    # ---------- 构造成 DataFrame ----------
    if not all_tasks:
        raise ValueError("没有找到任何有效帧")

    task_df = pd.DataFrame(all_tasks)

    print(f"\n  总计: {len(task_df)} 帧待处理")
    print(f"  Clip 数: {task_df['clip_id'].nunique()}")

    return task_df


# ========================
# 第二步:GPU 可用性检测和 Worker 分配
# ========================
def check_gpu_available(gpu_id: int) -> bool:
    """检查 GPU 是否空闲(没有被其他进程占用)"""
    try:
        # 查询正在使用指定 GPU 的进程
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
             '--format=csv,noheader', f'--id={gpu_id}'],
            capture_output=True,
            text=True,
            timeout=5
        )
        # 如果有输出,说明有进程在使用该 GPU
        if result.stdout.strip():
            return False
        return True
    except Exception:
        # 查询失败时,保守假设 GPU 可用
        print(f"    警告: GPU {gpu_id} 查询失败,假设可用")
        return True


def allocate_workers(gpu_ids: list) -> pd.DataFrame:
    """
    根据 GPU 可用性分配 workers
    每个可用的 GPU 分配 2 个 worker

    Returns:
        DataFrame with columns: worker_id, gpu_id
    """
    print("\n" + "=" * 70)
    print("第二步:GPU 可用性检测和 Worker 分配")
    print("=" * 70)

    print("\n检测 GPU 可用性...")
    available_gpus = []
    for gpu_id in gpu_ids:
        is_avail = check_gpu_available(gpu_id)
        status = "✓ 可用" if is_avail else "✗ 占用中"
        print(f"  GPU {gpu_id}: {status}")
        if is_avail:
            available_gpus.append(gpu_id)

    if not available_gpus:
        raise RuntimeError("没有可用的 GPU,所有指定的 GPU 都被占用")

    print(f"\n可用 GPU 数量: {len(available_gpus)} 个 ({available_gpus})")
    print("每个 GPU 分配 2 个 worker")

    # 分配 workers: 每个可用 GPU 2 个 worker
    workers = []
    worker_id = 0
    for gpu_id in available_gpus:
        for instance in range(2):
            workers.append({
                "worker_id": worker_id,
                "gpu_id": gpu_id,
            })
            worker_id += 1

    worker_df = pd.DataFrame(workers)

    print(f"\nWorker 分配结果:")
    print(worker_df.to_string(index=False))
    print(f"\n总 worker 数量: {len(worker_df)}")

    return worker_df


# ========================
# 第三步:任务分配到 Worker
# ========================
def assign_tasks_to_workers(task_df: pd.DataFrame, worker_df: pd.DataFrame) -> list:
    """
    将 frame dataframe 均匀分配给各个 worker

    Args:
        task_df: 帧任务 DataFrame (frame dataframe)
        worker_df: Worker DataFrame,包含 worker_id 和 gpu_id

    Returns:
        list of dicts, 每个 dict 包含:
            - worker_id: int
            - gpu_id: int
            - frame_df: DataFrame (该 worker 负责的帧)
    """
    print("\n" + "=" * 70)
    print("第三步:任务分配到 Worker")
    print("=" * 70)

    num_frames = len(task_df)
    num_workers = len(worker_df)

    print(f"  总帧数: {num_frames}")
    print(f"  总 worker 数: {num_workers}")

    # 计算每个 worker 应分配的帧数(尽量均匀)
    frames_per_worker = num_frames // num_workers
    remainder = num_frames % num_workers

    print(f"  平均每 worker 帧数: {frames_per_worker}")
    print(f"  剩余未分配帧数: {remainder}")

    # 为每个 worker 创建 task dict
    tasks = []
    start = 0
    for worker_id in range(num_workers):
        # 获取该 worker 对应的 gpu_id
        gpu_id = worker_df.loc[worker_df['worker_id'] == worker_id, 'gpu_id'].values[0]

        # 每个 worker 分配 frames_per_worker 帧,前 remainder 个 worker 多分 1 帧
        if worker_id < remainder:
            end = start + frames_per_worker + 1
        else:
            end = start + frames_per_worker

        # 切分出该 worker 负责的 frame_df
        frame_df = task_df.iloc[start:end].copy()

        task = {
            "worker_id": worker_id,
            "gpu_id": gpu_id,
            "frame_df": frame_df
        }
        tasks.append(task)

        print(f"    Worker {worker_id} (GPU {gpu_id}): {len(frame_df)} 帧")
        start = end

    print(f"\n  验证: 总帧数 = {sum(t['frame_df'].shape[0] for t in tasks)}")

    return tasks


# ========================
# 第四步:启动多GPU推理进程
# ========================
def create_output_dir(base_dir: str = "/data01/mikelee/infer_result") -> str:
    """创建带时间戳的输出目录"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"infer_result_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def run_worker_inference(task: Dict[str, Any], output_root: str, params: dict):
    """
    单个 worker 的推理函数（在子进程中运行）
    
    Args:
        task: 包含 worker_id, gpu_id, frame_df 的 dict
        output_root: 推理输出根目录
        params: 参数字典（包含 traj 等配置）
    """
    worker_id = task["worker_id"]
    gpu_id = task["gpu_id"]
    frame_df = task["frame_df"]
    traj = params["traj"]
    
    # 设置 GPU 环境（必须在 import torch 之前）
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    
    # Monkey-patch 必须在 import Alpamayo1_5 之前
    from transformers import PreTrainedModel
    def patched_get_correct_attn(self, requested_attention, is_init_check=False):
        return requested_attention
    PreTrainedModel.get_correct_attn_implementation = patched_get_correct_attn
    
    # 导入推理依赖
    import torch
    from PIL import Image
    from einops import rearrange
    from scipy.spatial.transform import Rotation as R
    
    # 添加 alpamayo1.5 路径并导入
    sys.path.insert(0, '/home/user/mikelee/alpamayo_project/alpamayo_1_5/alpamayo1.5-main/src')
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
    from alpamayo1_5 import helper
    
    # 模型路径
    MODEL_PATH = "/data01/mikelee/weight/models--nvidia--Alpamayo-1.5-10B"
    
    # 加载模型（每个进程只加载一次）
    print(f"[Worker {worker_id}] GPU {gpu_id} - 加载模型...")
    
    model = Alpamayo1_5.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    ).to("cuda")
    processor = helper.get_processor(model.tokenizer)
    model.eval()
    print(f"[Worker {worker_id}] GPU {gpu_id} - 模型加载完成")

    # 按 clip 分组处理
    clip_ids = frame_df['clip_id'].unique()

    # 创建 pred_traj 和 cot 文件夹
    pred_traj_dir = os.path.join(output_root, "pred_traj")
    cot_dir = os.path.join(output_root, "cot")
    os.makedirs(pred_traj_dir, exist_ok=True)
    os.makedirs(cot_dir, exist_ok=True)

    # 存储所有 CoT 结果用于汇总 CSV
    all_cot_results = []

    for clip_id in clip_ids:
        # 获取该 clip 的所有帧
        clip_frames = frame_df[frame_df['clip_id'] == clip_id]

        # 构建 clip 数据路径
        chunk_id = clip_frames.iloc[0]['chunk_id']
        data_dir = f"/data01/mikelee/data/data_sample_chunk{chunk_id}/infer/{clip_id}/data"

        for _, row in tqdm(clip_frames.iterrows(), desc=f"[W{worker_id} G{gpu_id}] {clip_id[:8]}", unit="fr"):
            frame_idx = int(row['frame_idx'])
            frame_id = int(row['frame_id'])

            # 加载图片
            images = []
            image_paths = row['image_paths']
            for cam in CAMERA_ORDER:
                for t in range(NUM_FRAMES_PER_CAMERA):
                    img_path = image_paths[(cam, t)]
                    img = Image.open(img_path).convert('RGB')
                    images.append(np.array(img))

            images = np.stack(images, axis=0)  # (16, H, W, 3)
            images = rearrange(images, '(c t) h w ch -> c t ch h w', c=4, t=4)
            images_tensor = torch.from_numpy(images).float()

            # 加载 egomotion
            history = np.load(row['history_path'], allow_pickle=False)
            future = np.load(row['future_path'], allow_pickle=False)

            hist_xyz_world = history[:, 5:8]
            hist_quat = history[:, 1:5]
            future_xyz_world = future[:, 1:4]

            t0_xyz = hist_xyz_world[-1].copy()
            t0_quat = hist_quat[-1].copy()

            hist_rot = R.from_quat(hist_quat).as_matrix()
            t0_rot = R.from_quat(t0_quat)
            t0_rot_inv = t0_rot.inv()

            hist_xyz_local = t0_rot_inv.apply(hist_xyz_world - t0_xyz)
            hist_rot_local = (t0_rot_inv * R.from_quat(hist_quat)).as_matrix()
            future_xyz_local = t0_rot_inv.apply(future_xyz_world - t0_xyz)

            hist_xyz_t = torch.from_numpy(hist_xyz_local).float().unsqueeze(0).unsqueeze(0)
            hist_rot_t = torch.from_numpy(hist_rot_local).float().unsqueeze(0).unsqueeze(0)
            future_xyz_t = torch.from_numpy(future_xyz_local).float().unsqueeze(0).unsqueeze(0)

            # 推理
            messages = helper.create_message(images_tensor.flatten(0, 1))
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False,
                continue_final_message=True, return_dict=True, return_tensors="pt",
            )

            model_inputs = {
                "tokenized_data": inputs,
                "ego_history_xyz": hist_xyz_t,
                "ego_history_rot": hist_rot_t,
            }
            model_inputs = helper.to_device(model_inputs, "cuda")
            torch.cuda.manual_seed_all(42)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                    data=model_inputs,
                    top_p=0.9, temperature=0.7,
                    num_traj_samples=traj,
                    max_generation_length=256,
                    return_extra=True,
                )

            # 保存预测轨迹到 pred_traj 文件夹
            pred_xyz_np = pred_xyz.cpu().numpy()[0, 0, 0, :, :2]
            pred_path = os.path.join(pred_traj_dir, f"{clip_id}_{frame_id:06d}_predtraj.npy")
            np.save(pred_path, pred_xyz_np)

            # 保存 CoT 结果到 cot 文件夹
            coc_texts = extra.get("cot", [[[]]])
            if isinstance(coc_texts, np.ndarray):
                coc_texts = coc_texts.tolist()
            elif isinstance(coc_texts, list) and len(coc_texts) > 0:
                if isinstance(coc_texts[0], np.ndarray):
                    coc_texts = [c.tolist() if isinstance(c, np.ndarray) else c for c in coc_texts]

            # 展平 CoT 文本
            cot_text = ""
            if coc_texts and len(coc_texts) > 0:
                def flatten(lst):
                    result = []
                    for item in lst:
                        if isinstance(item, list):
                            result.extend(flatten(item))
                        else:
                            result.append(str(item))
                    return result
                cot_text = " ".join(flatten(coc_texts))

            cot_path = os.path.join(cot_dir, f"{clip_id}_{frame_id:06d}_cot.txt")
            with open(cot_path, 'w') as f:
                f.write(cot_text)

            # 记录 CoT 结果
            all_cot_results.append({
                'chunk_id': chunk_id,
                'worker_id': worker_id,
                'clip_name': clip_id,
                'frame_number': frame_id,
                'cot_result': cot_text
            })

        print(f"[Worker {worker_id}] GPU {gpu_id} - Clip {clip_id}: {len(clip_frames)} 帧处理完成")

    # 保存 worker 结果汇总 CSV (文件名格式: infer_results_worker00.csv)
    result_csv_path = os.path.join(output_root, f"infer_results_worker{worker_id:02d}.csv")
    result_df = pd.DataFrame(all_cot_results)
    result_df.to_csv(result_csv_path, index=False)
    print(f"[Worker {worker_id}] GPU {gpu_id} - 结果已保存到 {result_csv_path}")

    return worker_id, gpu_id, len(frame_df)


def launch_inference_workers(tasks: List[Dict], output_root: str, params: dict, start_time: str, infer_info: dict):
    """
    启动多个 worker 进程进行推理

    Args:
        tasks: task list from Step 3
        output_root: 推理输出根目录
        params: 参数字典
        start_time: 推理开始时间
        infer_info: 推理相关信息字典
    """
    print("\n" + "=" * 70)
    print("第四步:启动多GPU推理进程")
    print("=" * 70)

    print(f"\n  输出目录: {output_root}")
    print(f"  总 worker 数: {len(tasks)}")

    # 使用 multiprocessing 启动每个 worker
    # 由于每个 worker 需要加载大模型,使用 Process 而非 Thread
    processes = []

    for task in tasks:
        p = mp.Process(
            target=run_worker_inference,
            args=(task, output_root, params)
        )
        p.start()
        processes.append(p)
        print(f"    启动 Worker {task['worker_id']} (GPU {task['gpu_id']}), 帧数: {len(task['frame_df'])}")

    # 等待所有进程完成
    print(f"\n  等待所有 worker 完成...")
    for p in processes:
        p.join()

    print(f"\n  所有 worker 推理完成!")

    # 合并所有 worker 的结果 CSV
    all_results = []
    for task in tasks:
        worker_id = task["worker_id"]
        result_csv = os.path.join(output_root, f"infer_results_worker{worker_id:02d}.csv")
        if os.path.exists(result_csv):
            df = pd.read_csv(result_csv)
            all_results.append(df)

    # 计算总完成帧数
    total_completed = 0
    if all_results:
        merged_df = pd.concat(all_results, ignore_index=True)
        merged_csv = os.path.join(output_root, "infer_results_all.csv")
        merged_df.to_csv(merged_csv, index=False)
        total_completed = len(merged_df)
        print(f"  合并结果已保存到: {merged_csv}")

    # 计算总帧数
    total_frames = sum(len(task["frame_df"]) for task in tasks)

    # 写入 infer_para.log
    end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path = os.path.join(output_root, "infer_para.log")
    with open(log_path, 'w') as f:
        f.write(f"=== Alpamayo 1.5 推理参数日志 ===\n\n")
        f.write(f"推理开始时间: {start_time}\n")
        f.write(f"推理结束时间: {end_time}\n\n")
        f.write(f"=== 推理参数 ===\n")
        f.write(f"GPU IDs: {params['gpu_ids']}\n")
        f.write(f"Chunk: {params['chunk']}\n")
        f.write(f"Clip: {'all' if params['use_all_clips'] else params['clip_list']}\n")
        f.write(f"num_frames ratio: {params['num_frames_ratio']}\n")
        f.write(f"step: {params['step']}\n")
        f.write(f"traj: {params['traj']}\n\n")
        f.write(f"=== 路径信息 ===\n")
        f.write(f"模型源码目录: {infer_info['model_src_dir']}\n")
        f.write(f"模型权重目录: {infer_info['model_weight_dir']}\n")
        f.write(f"推理脚本路径: {infer_info['script_path']}\n")
        f.write(f"推理结果目录: {output_root}\n\n")
        f.write(f"=== 推理统计 ===\n")
        f.write(f"总待推理帧数: {total_frames}\n")
        f.write(f"已完成帧数: {total_completed}\n")
        f.write(f"Worker 数量: {len(tasks)}\n")
    print(f"  参数日志已保存到: {log_path}")

    return len(processes)


# ========================
# 主函数
# ========================
def main():
    print("=" * 70)
    print("多GPU批量推理 - Alpamayo 1.5")
    print("=" * 70)

    # 第一步:解析参数
    print("\n[1/4] 解析参数...")
    try:
        params = parse_and_validate_args()
        print("  参数解析成功!")
        print(f"    GPU IDs: {params['gpu_ids']}")
        print(f"    Chunk: {params['chunk']}")
        print(f"    Clip: {'all' if params['use_all_clips'] else params['clip_list']}")
        print(f"    num_frames ratio: {params['num_frames_ratio']}")
        print(f"    step: {params['step']}")
        print(f"    traj: {params['traj']}")
    except Exception as e:
        print(f"  参数解析失败: {e}")
        sys.exit(1)

    # 第二步:构建任务 DataFrame
    print("\n[2/4] 构建任务 DataFrame...")
    try:
        task_df = build_task_dataframe(params)
        print(f"  任务 DataFrame: {len(task_df)} 帧, {task_df['clip_id'].nunique()} clips")
    except Exception as e:
        print(f"  任务构建失败: {e}")
        sys.exit(1)

    # 第三步:GPU 可用性检测和 Worker 分配
    print("\n[3/4] GPU 可用性检测和 Worker 分配...")
    try:
        worker_df = allocate_workers(params['gpu_ids'])
    except Exception as e:
        print(f"  Worker 分配失败: {e}")
        sys.exit(1)

    # 第四步:任务分配到 Worker
    print("\n[4/4] 任务分配到 Worker...")
    try:
        tasks = assign_tasks_to_workers(task_df, worker_df)
    except Exception as e:
        print(f"  任务分配失败: {e}")
        sys.exit(1)

    # 创建输出目录
    output_root = create_output_dir()
    print(f"\n  输出根目录: {output_root}")

    # 记录推理开始时间
    start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 推理相关信息
    infer_info = {
        "model_src_dir": "/home/user/mikelee/alpamayo_project/alpamayo_1_5/alpamayo1.5-main/src",
        "model_weight_dir": "/data01/mikelee/weight/models--nvidia--Alpamayo-1.5-10B",
        "script_path": "/home/user/mikelee/alpamayo_project/alpamayo_1_5/alpamayo_infer_1_5/run_inference_mulGPU_1_5.py",
    }

    # 第五步:启动推理进程
    num_workers = launch_inference_workers(tasks, output_root, params, start_time, infer_info)

    print("\n" + "=" * 70)
    print(f"全部完成! 共启动 {num_workers} 个 worker")
    print(f"结果目录: {output_root}")
    print("=" * 70)


if __name__ == "__main__":
    main()
