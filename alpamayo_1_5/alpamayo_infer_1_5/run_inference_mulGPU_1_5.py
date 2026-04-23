#!/usr/bin/env python3
"""
多GPU批量推理脚本 - Alpamayo 1.5
支持多GPU自动检测、灵活的任务分配、崩溃重启

输入参数:
    --gpu       GPU ID列表,逗号分隔,默认 "1" (如 "1,2,3")
    --chunk     data_sample chunk编号,支持单个或逗号分隔,如 "0" 或 "3,4,5"
    --clip      clip ID,逗号分隔或 "all"(默认 all)
    --num_frames 帧采样比例 0.0~1.0,0表示全部帧,默认 0
    --step     帧采样步长,必须>=1,默认 1
    --traj     轨迹数量,默认 1

环境要求:
    必须使用 conda 环境 alpamayo_env:
        conda activate alpamayo_env
        python3 run_inference_mulGPU_1_5.py ...

    或使用 conda run 启动:
        conda run -n alpamayo_env python3 run_inference_mulGPU_1_5.py ...

用法示例:
    # 激活环境后运行
    conda activate alpamayo_env
    python3 run_inference_mulGPU_1_5.py --chunk 3,4,5 --clip all --step 1 --gpu 1,2,3

    # 单次运行用 conda run
    conda run -n alpamayo_env python3 run_inference_mulGPU_1_5.py --chunk 0 --clip all --gpu 1
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
  --chunk     data_sample chunk编号,支持单个或逗号分隔
              示例: --chunk 0 或 --chunk 3,4,5
  --clip      clip ID,逗号分隔或 "all"(默认 all)
              示例: --clip 01d3588e... 或 --clip 01d3588e...,038678de...
              注意: clip可以在不同chunk中,脚本会自动查找
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
        type=str,
        default="0",
        help="data_sample chunk编号,单个或逗号分隔 (默认: 0)"
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
    parser.add_argument(
        "--save_diffusion_steps",
        action="store_true",
        help="保存 Flow Matching 每一步的中间结果 (action 和 trajectory)"
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

    # ---------- chunk 解析 (支持逗号分隔) ----------
    chunk_input = args.chunk.strip()
    if not chunk_input:
        raise ValueError("chunk 不能为空")
    
    chunk_ids = []
    for part in chunk_input.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            chunk_id = int(part)
            if chunk_id < 0:
                raise ValueError(f"chunk 必须 >= 0,实际: {chunk_id}")
            chunk_ids.append(chunk_id)
        except ValueError:
            raise ValueError(f"chunk ID 无效: '{part}'")
    
    chunk_ids = sorted(set(chunk_ids))  # 去重并排序

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
        "chunks": chunk_ids,
        "use_all_clips": use_all_clips,
        "clip_list": clip_list,
        "num_frames_ratio": num_frames,
        "step": step,
        "traj": traj,
        "save_diffusion_steps": args.save_diffusion_steps,
    }

    return params


# ========================
# 第二步:构建任务 DataFrame
# ========================
def build_task_dataframe(params: dict) -> pd.DataFrame:
    """根据参数构建所有待处理任务的 DataFrame"""

    all_tasks = []

    # ---------- 建立 clip_id -> chunk_id 的映射 ----------
    clip_to_chunk = {}
    for chunk in params["chunks"]:
        base_dir = f"{BASE_DATA_DIR}{chunk}/infer"
        if not os.path.exists(base_dir):
            print(f"  警告: chunk {chunk} 目录不存在: {base_dir}")
            continue
        for d in os.listdir(base_dir):
            clip_path = os.path.join(base_dir, d)
            if os.path.isdir(clip_path):
                clip_to_chunk[d] = chunk

    print(f"  扫描到 {len(clip_to_chunk)} 个 clips")

    # ---------- 确定要处理的 clip 列表 ----------
    if params["use_all_clips"]:
        # 处理所有 clips
        selected_clips = [(clip_id, chunk) for clip_id, chunk in clip_to_chunk.items()]
        print(f"  模式: all, 共 {len(selected_clips)} 个 clips")
    else:
        # 处理指定的 clips
        selected_clips = []
        missing = []
        for clip_id in params["clip_list"]:
            if clip_id in clip_to_chunk:
                selected_clips.append((clip_id, clip_to_chunk[clip_id]))
            else:
                missing.append(clip_id)
        
        if missing:
            raise FileNotFoundError(f"Clip 不存在: {missing}")
        
        print(f"  模式: 指定 clips, 共 {len(selected_clips)} 个 clips")

    # ---------- 收集每个 clip 的帧 ----------
    for clip_id, chunk in selected_clips:
        base_dir = f"{BASE_DATA_DIR}{chunk}/infer"
        clip_data_dir = os.path.join(base_dir, clip_id, "data")
        index_file = os.path.join(clip_data_dir, "inference_index_strict.csv")

        if not os.path.exists(index_file):
            print(f"  警告: {clip_id} 没有 inference_index_strict.csv,跳过")
            continue

        index_df = pd.read_csv(index_file)
        total_frames = len(index_df)

        # ---------- 确定要处理的帧 ----------
        sampled_indices = list(range(0, total_frames, params["step"]))

        if params["num_frames_ratio"] > 0:
            max_indices = max(1, int(total_frames * params["num_frames_ratio"]))
            max_indices = ((max_indices + params["step"] - 1) // params["step"]) * params["step"]
            sampled_indices = sampled_indices[:max_indices]

        print(f"    chunk{chunk:04d}/{clip_id}: 总帧数={total_frames}, 采样后={len(sampled_indices)} (step={params['step']}, ratio={params['num_frames_ratio']})")

        # ---------- 为每个采样帧构建记录 ----------
        for idx in sampled_indices:
            row = index_df.iloc[idx]
            frame_id = int(row["frame_id"])

            image_paths = {}
            for cam in CAMERA_ORDER:
                for t in range(NUM_FRAMES_PER_CAMERA):
                    col_idx = f"{cam}_f{t}_idx"
                    frame_idx = int(row[col_idx])
                    img_path = os.path.join(
                        clip_data_dir,
                        "camera_images",
                        cam,
                        f"{frame_idx:06d}.jpg"
                    )
                    image_paths[(cam, t)] = img_path

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
    print(f"  Chunk 数: {task_df['chunk_id'].nunique()}")
    print(f"  Clip 数: {task_df['clip_id'].nunique()}")

    return task_df


# ========================
# 第二步:GPU 可用性检测和 Worker 分配
# ========================
def check_gpu_available(gpu_id: int) -> bool:
    """检查 GPU 是否空闲(没有被其他进程占用)"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
             '--format=csv,noheader', f'--id={gpu_id}'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.stdout.strip():
            return False
        return True
    except Exception:
        print(f"    警告: GPU {gpu_id} 查询失败,假设可用")
        return True


def allocate_workers(gpu_ids: list) -> pd.DataFrame:
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

    workers = []
    worker_id = 0
    for gpu_id in available_gpus:
        for instance in range(2):
            workers.append({"worker_id": worker_id, "gpu_id": gpu_id})
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
    print("\n" + "=" * 70)
    print("第三步:任务分配到 Worker")
    print("=" * 70)

    num_frames = len(task_df)
    num_workers = len(worker_df)

    print(f"  总帧数: {num_frames}")
    print(f"  总 worker 数: {num_workers}")

    frames_per_worker = num_frames // num_workers
    remainder = num_frames % num_workers

    print(f"  平均每 worker 帧数: {frames_per_worker}")
    print(f"  剩余未分配帧数: {remainder}")

    tasks = []
    start = 0
    for worker_id in range(num_workers):
        gpu_id = worker_df.loc[worker_df['worker_id'] == worker_id, 'gpu_id'].values[0]

        if worker_id < remainder:
            end = start + frames_per_worker + 1
        else:
            end = start + frames_per_worker

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
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"infer_result_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def run_worker_inference(task: Dict[str, Any], output_root: str, params: dict):
    worker_id = task["worker_id"]
    gpu_id = task["gpu_id"]
    frame_df = task["frame_df"]
    traj = params["traj"]
    save_diffusion_steps = params.get("save_diffusion_steps", False)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    
    from transformers import PreTrainedModel
    def patched_get_correct_attn(self, requested_attention, is_init_check=False):
        return requested_attention
    PreTrainedModel.get_correct_attn_implementation = patched_get_correct_attn
    
    import torch
    from PIL import Image
    from einops import rearrange
    from scipy.spatial.transform import Rotation as R
    
    sys.path.insert(0, '/home/user/mikelee/alpamayo_project/alpamayo_1_5/alpamayo1.5-main/src')
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
    from alpamayo1_5 import helper
    
    MODEL_PATH = "/data01/mikelee/weight/models--nvidia--Alpamayo-1.5-10B"
    
    print(f"[Worker {worker_id}] GPU {gpu_id} - 加载模型...")
    
    model = Alpamayo1_5.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    ).to("cuda")
    processor = helper.get_processor(model.tokenizer)
    model.eval()
    print(f"[Worker {worker_id}] GPU {gpu_id} - 模型加载完成")

    clip_ids = frame_df['clip_id'].unique()

    pred_traj_dir = os.path.join(output_root, "pred_traj")
    cot_dir = os.path.join(output_root, "cot")
    os.makedirs(pred_traj_dir, exist_ok=True)
    os.makedirs(cot_dir, exist_ok=True)

    all_cot_results = []

    for clip_id in clip_ids:
        clip_frames = frame_df[frame_df['clip_id'] == clip_id]
        chunk_id = clip_frames.iloc[0]['chunk_id']
        data_dir = f"/data01/mikelee/data/data_sample_chunk{chunk_id}/infer/{clip_id}/data"

        for _, row in tqdm(clip_frames.iterrows(), desc=f"[W{worker_id} G{gpu_id}] {clip_id[:8]}", unit="fr"):
            frame_idx = int(row['frame_idx'])
            frame_id = int(row['frame_id'])

            images = []
            image_paths = row['image_paths']
            for cam in CAMERA_ORDER:
                for t in range(NUM_FRAMES_PER_CAMERA):
                    img_path = image_paths[(cam, t)]
                    img = Image.open(img_path).convert('RGB')
                    images.append(np.array(img))

            images = np.stack(images, axis=0)
            images = rearrange(images, '(c t) h w ch -> c t ch h w', c=4, t=4)
            images_tensor = torch.from_numpy(images).float()

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
                    return_diffusion_steps=save_diffusion_steps,
                )

            pred_xyz_np = pred_xyz.cpu().numpy()[0, 0, 0, :, :2]
            pred_path = os.path.join(pred_traj_dir, f"chunk{chunk_id:04d}_{clip_id}_{frame_id:06d}_predtraj.npy")
            np.save(pred_path, pred_xyz_np)

            coc_texts = extra.get("cot", [[[]]])
            if isinstance(coc_texts, np.ndarray):
                coc_texts = coc_texts.tolist()
            elif isinstance(coc_texts, list) and len(coc_texts) > 0:
                if isinstance(coc_texts[0], np.ndarray):
                    coc_texts = [c.tolist() if isinstance(c, np.ndarray) else c for c in coc_texts]

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

            cot_path = os.path.join(cot_dir, f"chunk{chunk_id:04d}_{clip_id}_{frame_id:06d}_cot.txt")
            with open(cot_path, 'w') as f:
                f.write(cot_text)

            # Save diffusion steps if requested
            if save_diffusion_steps and extra is not None:
                diff_steps_xyz = extra.get("diffusion_steps_xyz")
                diff_steps_action = extra.get("diffusion_steps_action")
                diff_steps_time = extra.get("diffusion_steps_time")
                
                if diff_steps_xyz is not None:
                    diff_steps_dir = os.path.join(output_root, "diffusion_steps")
                    os.makedirs(diff_steps_dir, exist_ok=True)
                    
                    # Save as npz format
                    diff_steps_path = os.path.join(
                        diff_steps_dir, 
                        f"chunk{chunk_id:04d}_{clip_id}_{frame_id:06d}_diffusion_steps.npz"
                    )
                    np.savez(
                        diff_steps_path,
                        xyz=diff_steps_xyz[0, 0],  # (num_steps+1, 64, 2)
                        action=diff_steps_action[0, 0],  # (num_steps+1, 64, 2)
                        time=diff_steps_time,  # (num_steps+1,)
                    )

            all_cot_results.append({
                'chunk_id': chunk_id,
                'worker_id': worker_id,
                'clip_name': clip_id,
                'frame_number': frame_id,
                'cot_result': cot_text
            })

        print(f"[Worker {worker_id}] GPU {gpu_id} - Clip {clip_id}: {len(clip_frames)} 帧处理完成")

    result_csv_path = os.path.join(output_root, f"infer_results_worker{worker_id:02d}.csv")
    result_df = pd.DataFrame(all_cot_results)
    result_df.to_csv(result_csv_path, index=False)
    print(f"[Worker {worker_id}] GPU {gpu_id} - 结果已保存到 {result_csv_path}")

    return worker_id, gpu_id, len(frame_df)


def launch_inference_workers(tasks: List[Dict], output_root: str, params: dict, start_time: str, infer_info: dict):
    print("\n" + "=" * 70)
    print("第四步:启动多GPU推理进程")
    print("=" * 70)

    print(f"\n  输出目录: {output_root}")
    print(f"  总 worker 数: {len(tasks)}")

    processes = []

    for task in tasks:
        p = mp.Process(
            target=run_worker_inference,
            args=(task, output_root, params)
        )
        p.start()
        processes.append(p)
        print(f"    启动 Worker {task['worker_id']} (GPU {task['gpu_id']}), 帧数: {len(task['frame_df'])}")

    print(f"\n  等待所有 worker 完成...")
    for p in processes:
        p.join()

    print(f"\n  所有 worker 推理完成!")

    all_results = []
    for task in tasks:
        worker_id = task["worker_id"]
        result_csv = os.path.join(output_root, f"infer_results_worker{worker_id:02d}.csv")
        if os.path.exists(result_csv):
            df = pd.read_csv(result_csv)
            all_results.append(df)

    total_completed = 0
    if all_results:
        merged_df = pd.concat(all_results, ignore_index=True)
        merged_csv = os.path.join(output_root, "infer_results_all.csv")
        merged_df.to_csv(merged_csv, index=False)
        total_completed = len(merged_df)
        print(f"  合并结果已保存到: {merged_csv}")

    total_frames = sum(len(task["frame_df"]) for task in tasks)

    end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path = os.path.join(output_root, "infer_para.log")
    with open(log_path, 'w') as f:
        f.write(f"=== Alpamayo 1.5 推理参数日志 ===\n\n")
        f.write(f"推理开始时间: {start_time}\n")
        f.write(f"推理结束时间: {end_time}\n\n")
        f.write(f"=== 推理参数 ===\n")
        f.write(f"GPU IDs: {params['gpu_ids']}\n")
        f.write(f"Chunk: {params['chunks']}\n")
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

    print("\n[1/4] 解析参数...")
    try:
        params = parse_and_validate_args()
        print("  参数解析成功!")
        print(f"    GPU IDs: {params['gpu_ids']}")
        print(f"    Chunk: {params['chunks']}")
        print(f"    Clip: {'all' if params['use_all_clips'] else params['clip_list']}")
        print(f"    num_frames ratio: {params['num_frames_ratio']}")
        print(f"    step: {params['step']}")
        print(f"    traj: {params['traj']}")
    except Exception as e:
        print(f"  参数解析失败: {e}")
        sys.exit(1)

    print("\n[2/4] 构建任务 DataFrame...")
    try:
        task_df = build_task_dataframe(params)
        print(f"  任务 DataFrame: {len(task_df)} 帧, {task_df['clip_id'].nunique()} clips")
    except Exception as e:
        print(f"  任务构建失败: {e}")
        sys.exit(1)

    print("\n[3/4] GPU 可用性检测和 Worker 分配...")
    try:
        worker_df = allocate_workers(params['gpu_ids'])
    except Exception as e:
        print(f"  Worker 分配失败: {e}")
        sys.exit(1)

    print("\n[4/4] 任务分配到 Worker...")
    try:
        tasks = assign_tasks_to_workers(task_df, worker_df)
    except Exception as e:
        print(f"  任务分配失败: {e}")
        sys.exit(1)

    output_root = create_output_dir()
    print(f"\n  输出根目录: {output_root}")

    start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    infer_info = {
        "model_src_dir": "/home/user/mikelee/alpamayo_project/alpamayo_1_5/alpamayo1.5-main/src",
        "model_weight_dir": "/data01/mikelee/weight/models--nvidia--Alpamayo-1.5-10B",
        "script_path": "/home/user/mikelee/alpamayo_project/alpamayo_1_5/alpamayo_infer_1_5/run_inference_mulGPU_1_5.py",
    }

    num_workers = launch_inference_workers(tasks, output_root, params, start_time, infer_info)

    print("\n" + "=" * 70)
    print(f"全部完成! 共启动 {num_workers} 个 worker")
    print(f"结果目录: {output_root}")
    print("=" * 70)


if __name__ == "__main__":
    main()
