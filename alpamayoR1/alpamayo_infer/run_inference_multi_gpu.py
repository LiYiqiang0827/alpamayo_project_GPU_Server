#!/usr/bin/env python3
"""
多GPU并行推理脚本 - Alpamayo V5 严格版本
支持自动GPU检测、显存分配、进程崩溃重启(3次)
输出与非并行化版本一致：result_strict/inference_results_strict.csv

用法:
    python3 run_inference_multi_gpu.py --clip <clip_id> --num_frames 100
"""
import os
import sys

# 必须在任何transformers导入之前设置
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import subprocess
import time
import json
import copy
import traceback
import threading
import multiprocessing
from multiprocessing import Process, Manager
from threading import Thread
import queue as thread_queue

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

def inference_worker_entry(args_dict):
    """Worker入口 - 必须在开头设置GPU"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args_dict['gpu_id'])
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from alpamayo_r1 import helper
    from scipy.spatial.transform import Rotation as R
    
    worker_name = f"GPU{args_dict['gpu_id']}-Inst{args_dict['instance_id']}"
    
    try:
        print(f"[{worker_name}] 启动，处理 {len(args_dict['tasks'])} 帧")
        
        model = AlpamayoR1.from_pretrained(MODEL_PATH, dtype=torch.bfloat16).to("cuda")
        processor = helper.get_processor(model.tokenizer)
        
        # 确保结果目录存在
        os.makedirs(args_dict['result_dir'], exist_ok=True)
        
        results = []
        for idx, row in tqdm(args_dict['tasks'], desc=worker_name, position=args_dict['gpu_id']*2+args_dict['instance_id']):
            try:
                # 加载图像
                images = []
                for cam in CAMERA_ORDER:
                    for t in range(4):
                        frame_idx = int(row[f'{cam}_f{t}_idx'])
                        img = Image.open(f"{args_dict['data_dir']}/camera_images/{cam}/{frame_idx:06d}.jpg").convert('RGB')
                        images.append(np.array(img))
                images = rearrange(np.stack(images), '(c t) h w ch -> c t ch h w', c=4, t=4)
                
                # 加载egomotion
                frame_id = int(row['frame_id'])
                history = np.load(f"{args_dict['data_dir']}/egomotion/frame_{frame_id:06d}_history.npy", allow_pickle=False)
                future = np.load(f"{args_dict['data_dir']}/egomotion/frame_{frame_id:06d}_future_gt.npy", allow_pickle=False)
                
                R_inst = __import__('scipy.spatial.transform').spatial.transform.Rotation
                hist_xyz_world, hist_quat = history[:,5:8], history[:,1:5]
                t0_rot_inv = R_inst.from_quat(hist_quat[-1]).inv()
                hist_xyz = t0_rot_inv.apply(hist_xyz_world - hist_xyz_world[-1])
                hist_rot = (t0_rot_inv * R_inst.from_quat(hist_quat)).as_matrix()
                future_xyz = t0_rot_inv.apply(future[:,1:4] - hist_xyz_world[-1])
                
                image_frames = torch.from_numpy(images).float()
                hist_xyz_t = torch.from_numpy(hist_xyz).float().unsqueeze(0).unsqueeze(0)
                hist_rot_t = torch.from_numpy(hist_rot).float().unsqueeze(0).unsqueeze(0)
                future_xyz_t = torch.from_numpy(future_xyz).float().unsqueeze(0).unsqueeze(0)
                
                # 推理
                messages = helper.create_message(image_frames.flatten(0, 1))
                inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, continue_final_message=True, return_dict=True, return_tensors="pt")
                
                model_inputs = {"tokenized_data": inputs, "ego_history_xyz": hist_xyz_t, "ego_history_rot": hist_rot_t}
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
                
                # 计算ADE
                pred_np = pred_xyz.cpu().numpy()[0, 0, 0, :, :3]
                gt_np = future_xyz_t.cpu().numpy()[0, 0, :, :3]
                ade = np.linalg.norm(pred_np - gt_np, axis=1).mean()
                
                coc_texts = extra.get("cot", [[[]]])[0]
                if isinstance(coc_texts, np.ndarray):
                    coc_texts = coc_texts.tolist()
                
                # 保存结果 - 直接写入标准目录
                np.save(f"{args_dict['result_dir']}/pred_{frame_id:06d}.npy", pred_np)
                results.append({
                    'frame_id': frame_id, 'ego_idx': int(row['ego_idx']),
                    'ade': float(ade), 'inference_time_ms': round(inference_time, 1),
                    'coc_text': json.dumps(coc_texts)
                })
                
                args_dict['progress_queue'].put({'frame_id': frame_id, 'success': True})
                
            except Exception as e:
                print(f"[{worker_name}] Frame {int(row['frame_id'])} 错误: {e}")
                args_dict['progress_queue'].put({'frame_id': int(row['frame_id']), 'success': False})
        
        # 返回结果列表给主进程合并
        return results
        
    except Exception as e:
        print(f"[{worker_name}] 致命错误: {e}")
        traceback.print_exc()
        return []

def start_worker_with_restart(args_dict, max_restarts=MAX_RESTARTS):
    """带自动重启的worker启动器"""
    worker_name = f"GPU{args_dict['gpu_id']}-Inst{args_dict['instance_id']}"
    
    for attempt in range(max_restarts + 1):
        if attempt > 0:
            print(f"[{worker_name}] 第{attempt}次重启...")
            time.sleep(5)
        
        # 使用进程池执行并获取返回值
        import multiprocessing.pool
        with multiprocessing.pool.Pool(1) as pool:
            result = pool.apply(inference_worker_entry, (args_dict,))
            if result:  # 如果有结果返回，说明成功
                return result
    
    print(f"[{worker_name}] 达到最大重启次数，放弃")
    return []

def main():
    parser = argparse.ArgumentParser(description='多GPU并行推理 - V5严格版本')
    parser.add_argument('--clip', type=str, required=True)
    parser.add_argument('--chunk', type=int, default=0)
    parser.add_argument('--num_frames', type=int, default=100)
    parser.add_argument('--traj', type=int, default=1)
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--top_p', type=float, default=0.98)
    parser.add_argument('--temp', type=float, default=0.6)
    parser.add_argument('--max_len', type=int, default=64, help='最大生成长度')
    args = parser.parse_args()
    
    BASE_DIR = f"/data01/vla/data/data_sample_chunk{args.chunk}"
    DATA_DIR = f"{BASE_DIR}/infer/{args.clip}/data"
    # 输出到标准目录 result_strict，与非并行化版本一致
    RESULT_DIR = f"{BASE_DIR}/infer/{args.clip}/result_strict"
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    print("="*70)
    print(f"🚀 多GPU并行推理 - Clip: {args.clip}")
    print("="*70)
    
    # 检测GPU
    print("\n📊 检测GPU资源...")
    gpu_info = get_gpu_info()
    if not gpu_info:
        print("❌ 未检测到GPU"); return
    
    gpu_assignments = []
    for gpu in gpu_info:
        for i in range(gpu['instances']):
            gpu_assignments.append((gpu['id'], i))
        print(f"  GPU {gpu['id']}: {gpu['instances']}实例")
    
    total_instances = len(gpu_assignments)
    print(f"\n✅ 总实例数: {total_instances}")
    
    # 加载数据
    print("\n📂 加载数据...")
    index_df = pd.read_csv(f"{DATA_DIR}/inference_index_strict.csv")
    sampled = list(range(0, len(index_df), args.step))[:args.num_frames]
    print(f"   总帧数: {len(index_df)}, 采样后: {len(sampled)}")
    tasks = [(i, index_df.iloc[i]) for i in sampled]
    
    # 分配任务
    tasks_per = len(tasks) // total_instances
    extra = len(tasks) % total_instances
    
    worker_configs = []
    start = 0
    for i, (gpu_id, inst_id) in enumerate(gpu_assignments):
        count = tasks_per + (1 if i < extra else 0)
        end = start + count
        worker_configs.append({
            'gpu_id': gpu_id, 'instance_id': inst_id,
            'tasks': tasks[start:end],
            'data_dir': DATA_DIR,
            'result_dir': RESULT_DIR,  # 所有worker输出到同一目录
            'num_traj': args.traj, 'top_p': args.top_p, 'temp': args.temp, 'max_len': args.max_len,
            'progress_queue': None,
            'worker_id': f"GPU{gpu_id}-Inst{inst_id}"
        })
        print(f"  GPU{gpu_id}-Inst{inst_id}: {count}帧")
        start = end
    
    # 启动workers
    print("\n🔄 启动推理进程...")
    manager = Manager()
    progress_queue = manager.Queue()
    
    for cfg in worker_configs:
        cfg['progress_queue'] = progress_queue
    
    from concurrent.futures import ThreadPoolExecutor
    
    def run_worker(cfg):
        return start_worker_with_restart(cfg)
    
    all_results = []
    completed = failed = 0
    
    with ThreadPoolExecutor(max_workers=total_instances) as executor:
        futures = {executor.submit(run_worker, cfg): cfg['worker_id'] for cfg in worker_configs}
        
        with tqdm(total=len(tasks), desc='总体进度') as pbar:
            # 收集结果
            for future in futures:
                worker_results = future.result()
                all_results.extend(worker_results)
            
            # 更新进度
            while completed + failed < len(tasks):
                try:
                    msg = progress_queue.get(timeout=0.5)
                    if msg['success']: completed += 1
                    else: failed += 1
                    pbar.update(1)
                except:
                    if all(f.done() for f in futures):
                        break
    
    # 合并保存结果 - 与非并行化版本格式一致
    print("\n📝 保存结果...")
    if all_results:
        # 按frame_id排序
        all_results.sort(key=lambda x: x['frame_id'])
        results_df = pd.DataFrame(all_results)
        # 保存为标准的 inference_results_strict.csv
        results_df.to_csv(f"{RESULT_DIR}/inference_results_strict.csv", index=False)
        
        success_count = results_df['ade'].notna().sum()
        avg_ade = results_df['ade'].mean()
        avg_time = results_df['inference_time_ms'].mean()
        
        total_time_single = avg_time * len(tasks) / 1000
        total_time_parallel = total_time_single / total_instances
        
        print(f"\n{'='*70}")
        print("✅ 推理完成!")
        print(f"{'='*70}")
        print(f"成功: {success_count}/{len(tasks)}")
        print(f"平均ADE: {avg_ade:.2f}m")
        print(f"平均推理时间: {avg_time:.1f}ms")
        print(f"并行加速: {total_instances}x ({total_time_single:.1f}s → {total_time_parallel:.1f}s)")
        print(f"结果: {RESULT_DIR}/inference_results_strict.csv")

if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except: pass
    main()
