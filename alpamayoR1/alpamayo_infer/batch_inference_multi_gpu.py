#!/usr/bin/env python3
"""
多GPU批量Clip推理 - 自动遍历所有clips
支持：--num_frames 0 (处理全部帧), 断点续跑

用法:
    python3 batch_inference_multi_gpu.py --chunk 0 --num_frames 0
    python3 batch_inference_multi_gpu.py --chunk 0 --num_frames 1000
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
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from einops import rearrange

sys.path.insert(0, '/home/user/mikelee/alpamayo-main/src')

MODEL_PATH = "/data01/vla/models--nvidia--Alpamayo-R1-10B/snapshots/22fab1399111f50b52bfbe5d8b809f39bd4c2fe1"
CAMERA_ORDER = ['camera_cross_left_120fov','camera_front_wide_120fov','camera_cross_right_120fov','camera_front_tele_30fov']
VRAM_SINGLE, VRAM_DOUBLE, MAX_RESTARTS = 35, 70, 3

def get_gpu_info():
    try:
        result = subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.total,memory.free,memory.used','--format=csv,noheader,nounits'], capture_output=True, text=True)
        gpu_info = []
        for line in result.stdout.strip().split('\n'):
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 5:
                total_gb, free_gb = float(parts[2])/1024, float(parts[3])/1024
                instances = 2 if free_gb >= VRAM_DOUBLE else (1 if free_gb >= VRAM_SINGLE else 0)
                gpu_info.append({'id':int(parts[0]),'name':parts[1],'total_gb':total_gb,'free_gb':free_gb,'instances':instances})
        return gpu_info
    except: return []

def discover_clips(chunk_id, base_dir='/data01/vla/data'):
    """发现所有已预处理的clips"""
    chunk_dir = f"{base_dir}/data_sample_chunk{chunk_id}/infer"
    if not os.path.exists(chunk_dir):
        return []
    
    clips = []
    for item in sorted(os.listdir(chunk_dir)):
        clip_path = os.path.join(chunk_dir, item)
        if os.path.isdir(clip_path):
            data_dir = os.path.join(clip_path, 'data')
            index_file = os.path.join(data_dir, 'inference_index_strict.csv')
            if os.path.exists(index_file):
                clips.append({
                    'clip_id': item,
                    'data_dir': data_dir,
                    'result_dir': os.path.join(clip_path, 'result_strict')
                })
    return clips

def inference_worker_entry(args_dict):
    """Worker入口 - 处理分配给它的所有clips"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args_dict['gpu_id'])
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from alpamayo_r1 import helper
    from scipy.spatial.transform import Rotation as R
    
    worker_name = f"GPU{args_dict['gpu_id']}-Inst{args_dict['instance_id']}"
    worker_tasks = args_dict['tasks']
    
    print(f"[{worker_name}] 启动，处理 {len(worker_tasks)} 个clips")
    
    # 加载模型（每个worker只加载一次）
    model = AlpamayoR1.from_pretrained(MODEL_PATH, dtype=torch.bfloat16).to("cuda")
    processor = helper.get_processor(model.tokenizer)
    
    all_results = []
    total_frames = 0
    
    for task in worker_tasks:
        clip_id = task['clip_id']
        data_dir = task['data_dir']
        result_dir = task['result_dir']
        num_frames = task['num_frames']
        step = task['step']
        
        os.makedirs(result_dir, exist_ok=True)
        
        # 加载索引
        try:
            index_df = pd.read_csv(f"{data_dir}/inference_index_strict.csv")
        except Exception as e:
            print(f"[{worker_name}] {clip_id} 加载索引失败: {e}")
            continue
        
        total_clip_frames = len(index_df)
        # num_frames=0 表示处理全部
        if num_frames == 0:
            sampled_indices = list(range(0, total_clip_frames, step))
        else:
            sampled_indices = list(range(0, total_clip_frames, step))[:num_frames]
        
        if not sampled_indices:
            print(f"[{worker_name}] {clip_id}: 没有可处理的帧")
            continue
        
        print(f"[{worker_name}] {clip_id}: 处理 {len(sampled_indices)}/{total_clip_frames} 帧")
        
        clip_results = []
        processed_count = 0
        
        for idx in sampled_indices:
            try:
                row = index_df.iloc[idx]
                frame_id = int(row['frame_id'])
                
                # 检查是否已处理
                pred_file = f"{result_dir}/pred_{frame_id:06d}.npy"
                if os.path.exists(pred_file):
                    continue
                
                # 加载图像
                images = []
                for cam in CAMERA_ORDER:
                    for t in range(4):
                        frame_idx = int(row[f'{cam}_f{t}_idx'])
                        img_path = f'{data_dir}/camera_images/{cam}/{frame_idx:06d}.jpg'
                        img = Image.open(img_path).convert('RGB')
                        images.append(np.array(img))
                
                images = rearrange(np.stack(images), '(c t) h w ch -> c t ch h w', c=4, t=4)
                
                # 加载egomotion
                history = np.load(f"{data_dir}/egomotion/frame_{frame_id:06d}_history.npy", allow_pickle=False)
                future = np.load(f"{data_dir}/egomotion/frame_{frame_id:06d}_future_gt.npy", allow_pickle=False)
                
                hist_xyz_world, hist_quat = history[:,5:8], history[:,1:5]
                future_xyz_world = future[:,1:4]
                
                t0_rot_inv = R.from_quat(hist_quat[-1]).inv()
                hist_xyz = t0_rot_inv.apply(hist_xyz_world - hist_xyz_world[-1])
                hist_rot = (t0_rot_inv * R.from_quat(hist_quat)).as_matrix()
                future_xyz = t0_rot_inv.apply(future_xyz_world - hist_xyz_world[-1])
                
                image_frames = torch.from_numpy(images).float()
                hist_xyz_t = torch.from_numpy(hist_xyz).float().unsqueeze(0).unsqueeze(0)
                hist_rot_t = torch.from_numpy(hist_rot).float().unsqueeze(0).unsqueeze(0)
                future_xyz_t = torch.from_numpy(future_xyz).float().unsqueeze(0).unsqueeze(0)
                
                # 推理
                messages = helper.create_message(image_frames.flatten(0, 1))
                inputs = processor.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=False,
                    continue_final_message=True, return_dict=True, return_tensors="pt"
                )
                
                model_inputs = {
                    "tokenized_data": inputs,
                    "ego_history_xyz": hist_xyz_t,
                    "ego_history_rot": hist_rot_t
                }
                model_inputs = helper.to_device(model_inputs, "cuda")
                torch.cuda.manual_seed_all(42)
                
                torch.cuda.synchronize()
                start = time.time()
                
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                        data=copy.deepcopy(model_inputs),
                        top_p=args_dict['top_p'], temperature=args_dict['temp'],
                        num_traj_samples=args_dict['num_traj'],
                        max_generation_length=args_dict['max_len'],
                        return_extra=True
                    )
                
                torch.cuda.synchronize()
                inference_time = (time.time() - start) * 1000
                
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
                
                clip_results.append({
                    'clip_id': clip_id,
                    'frame_id': frame_id,
                    'ego_idx': int(row['ego_idx']),
                    'ade': float(ade),
                    'inference_time_ms': round(inference_time, 1),
                    'cot_text': json.dumps(coc_texts)
                })
                
                processed_count += 1
                args_dict['progress_queue'].put({'clip_id': clip_id, 'frame_id': frame_id, 'success': True})
                
            except Exception as e:
                print(f"[{worker_name}] {clip_id} frame {frame_id} 错误: {e}")
                args_dict['progress_queue'].put({'clip_id': clip_id, 'frame_id': frame_id, 'success': False})
        
        # 保存这个clip的结果
        if clip_results:
            result_csv = f"{result_dir}/inference_results_strict.csv"
            if os.path.exists(result_csv):
                existing_df = pd.read_csv(result_csv)
                new_df = pd.DataFrame(clip_results)
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                combined_df.drop_duplicates(subset=['frame_id'], keep='last', inplace=True)
                combined_df.sort_values('frame_id', inplace=True)
                combined_df.to_csv(result_csv, index=False)
            else:
                pd.DataFrame(clip_results).to_csv(result_csv, index=False)
            
            print(f"[{worker_name}] {clip_id}: 完成 {len(clip_results)} 新帧")
            all_results.extend(clip_results)
            total_frames += processed_count
    
    print(f"[{worker_name}] 所有clips完成，共 {total_frames} 新帧")
    return all_results

def start_worker_with_restart(args_dict, max_restarts=MAX_RESTARTS):
    """带自动重启的worker"""
    worker_name = f"GPU{args_dict['gpu_id']}-Inst{args_dict['instance_id']}"
    
    for attempt in range(max_restarts + 1):
        if attempt > 0:
            print(f"[{worker_name}] 第{attempt}次重启...")
            time.sleep(5)
        
        try:
            return inference_worker_entry(args_dict)
        except Exception as e:
            print(f"[{worker_name}] 错误: {e}")
            traceback.print_exc()
    
    print(f"[{worker_name}] 达到最大重启次数")
    return []

def main():
    parser = argparse.ArgumentParser(description='多GPU批量Clip推理')
    parser.add_argument('--chunk', type=int, default=0)
    parser.add_argument('--num_frames', type=int, default=1000, help='每clip帧数 (0=全部)')
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--traj', type=int, default=1)
    parser.add_argument('--top_p', type=float, default=0.98)
    parser.add_argument('--temp', type=float, default=0.6)
    parser.add_argument('--max_len', type=int, default=256)
    args = parser.parse_args()
    
    print("="*70)
    print("🚀 多GPU批量Clip推理")
    print("="*70)
    frame_info = "全部" if args.num_frames == 0 else str(args.num_frames)
    print(f"配置: chunk={args.chunk}, {frame_info}帧/clip, step={args.step}")
    print()
    
    # 发现clips
    print("📂 扫描clips...")
    clips = discover_clips(args.chunk)
    print(f"发现 {len(clips)} 个clips")
    
    if not clips:
        print("❌ 没有clips可处理")
        return
    
    # 显示clips信息
    for i, clip in enumerate(clips, 1):
        index_df = pd.read_csv(f"{clip['data_dir']}/inference_index_strict.csv")
        total = len(index_df)
        target = total if args.num_frames == 0 else min(args.num_frames, total)
        print(f"  [{i}] {clip['clip_id'][:8]}...: {total}帧 -> 处理{target}帧")
    
    # 检测GPU
    print("\n📊 检测GPU资源...")
    gpu_info = get_gpu_info()
    if not gpu_info:
        print("❌ 未检测到GPU")
        return
    
    gpu_assignments = []
    for gpu in gpu_info:
        for i in range(gpu['instances']):
            gpu_assignments.append((gpu['id'], i))
    
    total_workers = len(gpu_assignments)
    print(f"总workers: {total_workers}")
    for gpu_id, inst_id in gpu_assignments:
        print(f"  GPU{gpu_id}-Inst{inst_id}")
    
    # 分配clips给workers
    print("\n📋 分配任务...")
    tasks_per_worker = [[] for _ in range(total_workers)]
    
    for i, clip in enumerate(clips):
        worker_id = i % total_workers
        tasks_per_worker[worker_id].append({
            'clip_id': clip['clip_id'],
            'data_dir': clip['data_dir'],
            'result_dir': clip['result_dir'],
            'num_frames': args.num_frames,
            'step': args.step
        })
    
    for i, tasks in enumerate(tasks_per_worker):
        gpu_id, inst_id = gpu_assignments[i]
        print(f"  Worker {i} (GPU{gpu_id}-Inst{inst_id}): {len(tasks)} clips")
    
    # 计算总任务数
    total_tasks = sum(
        len(pd.read_csv(f"{clip['data_dir']}/inference_index_strict.csv")) 
        if args.num_frames == 0 else min(args.num_frames, len(pd.read_csv(f"{clip['data_dir']}/inference_index_strict.csv")))
        for clip in clips
    )
    
    # 启动workers
    print(f"\n🔄 启动推理进程...")
    manager = Manager()
    progress_queue = manager.Queue()
    
    worker_configs = []
    for i, (gpu_id, inst_id) in enumerate(gpu_assignments):
        worker_configs.append({
            'gpu_id': gpu_id,
            'instance_id': inst_id,
            'tasks': tasks_per_worker[i],
            'num_traj': args.traj,
            'top_p': args.top_p,
            'temp': args.temp,
            'max_len': args.max_len,
            'progress_queue': progress_queue,
            'worker_id': f"GPU{gpu_id}-Inst{inst_id}"
        })
    
    start_time = time.time()
    all_results = []
    completed = failed = 0
    
    with ThreadPoolExecutor(max_workers=total_workers) as executor:
        futures = {executor.submit(start_worker_with_restart, cfg): cfg['worker_id'] 
                   for cfg in worker_configs}
        
        with tqdm(total=total_tasks, desc='总体进度') as pbar:
            for future in futures:
                try:
                    worker_results = future.result()
                    all_results.extend(worker_results)
                except Exception as e:
                    print(f"Worker错误: {e}")
            
            while completed + failed < total_tasks:
                try:
                    msg = progress_queue.get(timeout=1.0)
                    if msg['success']:
                        completed += 1
                    else:
                        failed += 1
                    pbar.update(1)
                except:
                    if all(f.done() for f in futures):
                        break
    
    elapsed = time.time() - start_time
    
    # 总结
    print("\n" + "="*70)
    print("📊 处理完成!")
    print("="*70)
    print(f"总clips: {len(clips)}")
    print(f"成功: {completed}/{total_tasks}")
    print(f"失败: {failed}")
    print(f"总耗时: {elapsed/60:.1f} 分钟")
    
    if all_results:
        results_df = pd.DataFrame(all_results)
        print(f"\n平均ADE: {results_df['ade'].mean():.4f}m")
        print(f"平均推理时间: {results_df['inference_time_ms'].mean():.1f}ms")
        
        # 保存汇总
        summary_dir = "/data01/vla/cot_collection"
        os.makedirs(summary_dir, exist_ok=True)
        summary = {
            'chunk': args.chunk,
            'clips_processed': len(clips),
            'total_frames': len(all_results),
            'successful': completed,
            'failed': failed,
            'elapsed_seconds': elapsed,
            'avg_ade': float(results_df['ade'].mean()),
            'avg_inference_time_ms': float(results_df['inference_time_ms'].mean())
        }
        with open(f"{summary_dir}/batch_multi_gpu_chunk{args.chunk}.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✅ 汇总保存: {summary_dir}/batch_multi_gpu_chunk{args.chunk}.json")

if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except:
        pass
    main()
