#!/usr/bin/env python3
"""
简化版多GPU批量Clip推理
每个worker加载一次模型，顺序处理多个clips
用法:
    # 处理所有帧
    python3 batch_inference_simple.py --chunk 0 --num_frames 0
    
    # 处理前1000帧
    python3 batch_inference_simple.py --chunk 0 --num_frames 1000
"""
import os
import sys
import argparse
import subprocess
import time
import json
import glob
from pathlib import Path

os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from einops import rearrange

sys.path.insert(0, '/home/user/mikelee/alpamayo-main/src')
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper
from scipy.spatial.transform import Rotation as R

MODEL_PATH = "/data01/vla/models--nvidia--Alpamayo-R1-10B/snapshots/22fab1399111f50b52bfbe5d8b809f39bd4c2fe1"
CAMERA_ORDER = ['camera_cross_left_120fov','camera_front_wide_120fov','camera_cross_right_120fov','camera_front_tele_30fov']

def process_single_clip(clip_info, model, processor, num_frames, step, args):
    """处理单个clip"""
    clip_id = clip_info['clip_id']
    data_dir = clip_info['data_dir']
    result_dir = clip_info['result_dir']
    
    os.makedirs(result_dir, exist_ok=True)
    
    # 加载索引
    try:
        index_df = pd.read_csv(f"{data_dir}/inference_index_strict.csv")
    except Exception as e:
        print(f"  {clip_id}: 加载索引失败 - {e}")
        return None
    
    total_frames = len(index_df)
    
    # 如果num_frames=0，处理所有帧；否则处理指定数量
    if num_frames == 0:
        sampled_indices = list(range(0, total_frames, step))
    else:
        sampled_indices = list(range(0, total_frames, step))[:num_frames]
    
    if not sampled_indices:
        print(f"  {clip_id}: 没有可处理的帧")
        return None
    
    print(f"  {clip_id}: 处理 {len(sampled_indices)}/{total_frames} 帧")
    
    results = []
    for idx in tqdm(sampled_indices, desc=f"  {clip_id[:8]}", leave=False):
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
                    img_idx = int(row[f'{cam}_f{t}_idx'])
                    img_path = f"{data_dir}/camera_images/{cam}/{img_idx:06d}.jpg"
                    img = Image.open(img_path).convert('RGB')
                    images.append(np.array(img))
            
            images = rearrange(np.stack(images), '(c t) h w ch -> c t ch h w', c=4, t=4)
            
            # 加载egomotion
            history = np.load(f"{data_dir}/egomotion/frame_{frame_id:06d}_history.npy", allow_pickle=False)
            future = np.load(f"{data_dir}/egomotion/frame_{frame_id:06d}_future_gt.npy", allow_pickle=False)
            
            hist_xyz_world = history[:, 5:8]
            hist_quat = history[:, 1:5]
            future_xyz_world = future[:, 1:4]
            
            t0_rot_inv = R.from_quat(hist_quat[-1]).inv()
            hist_xyz = t0_rot_inv.apply(hist_xyz_world - hist_xyz_world[-1])
            hist_rot = (t0_rot_inv * R.from_quat(hist_quat)).as_matrix()
            future_xyz_local = t0_rot_inv.apply(future_xyz_world - hist_xyz_world[-1])
            
            image_frames = torch.from_numpy(images).float()
            hist_xyz_t = torch.from_numpy(hist_xyz).float().unsqueeze(0).unsqueeze(0)
            hist_rot_t = torch.from_numpy(hist_rot).float().unsqueeze(0).unsqueeze(0)
            future_xyz_t = torch.from_numpy(future_xyz_local).float().unsqueeze(0).unsqueeze(0)
            
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
                    data=model_inputs,
                    top_p=args.top_p, temperature=args.temp,
                    num_traj_samples=args.traj,
                    max_generation_length=args.max_len,
                    return_extra=True
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
            
            # 保存
            np.save(pred_file, pred_np)
            
            results.append({
                'clip_id': clip_id,
                'frame_id': frame_id,
                'ego_idx': int(row['ego_idx']),
                'ade': float(ade),
                'inference_time_ms': round(inference_time, 1),
                'cot_text': json.dumps(cot_texts)
            })
            
        except Exception as e:
            print(f"    Frame {frame_id} 错误: {e}")
    
    # 保存结果
    if results:
        result_csv = f"{result_dir}/inference_results_strict.csv"
        if os.path.exists(result_csv):
            existing_df = pd.read_csv(result_csv)
            new_df = pd.DataFrame(results)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df.drop_duplicates(subset=['frame_id'], keep='last', inplace=True)
            combined_df.sort_values('frame_id', inplace=True)
            combined_df.to_csv(result_csv, index=False)
        else:
            pd.DataFrame(results).to_csv(result_csv, index=False)
        
        # 统计
        avg_ade = np.mean([r['ade'] for r in results])
        avg_time = np.mean([r['inference_time_ms'] for r in results])
        print(f"  {clip_id}: 完成 {len(results)} 帧, ADE={avg_ade:.3f}m, Time={avg_time:.0f}ms")
    
    return results

def discover_clips(chunk_id, base_dir='/data01/vla/data'):
    """发现clips"""
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

def main():
    parser = argparse.ArgumentParser(description='批量Clip推理')
    parser.add_argument('--chunk', type=int, default=0)
    parser.add_argument('--max_clips', type=int, default=None)
    parser.add_argument('--num_frames', type=int, default=1000, help='每clip处理的帧数 (0=全部)')
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--traj', type=int, default=1)
    parser.add_argument('--top_p', type=float, default=0.98)
    parser.add_argument('--temp', type=float, default=0.6)
    parser.add_argument('--max_len', type=int, default=256)
    args = parser.parse_args()
    
    print("="*70)
    print("🚀 批量Clip推理 (单GPU顺序版)")
    print("="*70)
    frame_info = "全部" if args.num_frames == 0 else str(args.num_frames)
    print(f"配置: chunk={args.chunk}, {frame_info}帧/clip, step={args.step}")
    print()
    
    # 发现clips
    clips = discover_clips(args.chunk)
    print(f"📂 发现 {len(clips)} 个clips")
    
    if args.max_clips:
        clips = clips[:args.max_clips]
        print(f"⚠️  限制处理前 {args.max_clips} 个clips")
    
    if not clips:
        print("❌ 没有clips可处理")
        return
    
    # 加载模型
    print("\n🔄 加载模型...")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model = AlpamayoR1.from_pretrained(MODEL_PATH, dtype=torch.bfloat16).to("cuda")
    processor = helper.get_processor(model.tokenizer)
    print("✅ 模型加载完成")
    
    # 处理clips
    print(f"\n🔄 开始处理 {len(clips)} 个clips...")
    all_results = []
    start_time = time.time()
    
    for i, clip_info in enumerate(clips, 1):
        print(f"\n[{i}/{len(clips)}] ")
        results = process_single_clip(clip_info, model, processor, 
                                     args.num_frames, args.step, args)
        if results:
            all_results.extend(results)
    
    elapsed = time.time() - start_time
    
    # 总结
    print("\n" + "="*70)
    print("📊 处理完成!")
    print("="*70)
    print(f"总clips: {len(clips)}")
    print(f"总帧数: {len(all_results)}")
    print(f"总耗时: {elapsed/60:.1f} 分钟")
    
    if all_results:
        results_df = pd.DataFrame(all_results)
        print(f"平均ADE: {results_df['ade'].mean():.4f}m")
        print(f"平均推理时间: {results_df['inference_time_ms'].mean():.1f}ms")
        
        # 保存汇总
        summary_dir = "/data01/vla/cot_collection"
        os.makedirs(summary_dir, exist_ok=True)
        summary = {
            'chunk': args.chunk,
            'clips_processed': len(clips),
            'total_frames': len(all_results),
            'elapsed_seconds': elapsed,
            'avg_ade': float(results_df['ade'].mean()),
            'avg_inference_time_ms': float(results_df['inference_time_ms'].mean())
        }
        with open(f"{summary_dir}/batch_summary_chunk{args.chunk}.json", 'w') as f:
            json.dump(summary, f, indent=2)

if __name__ == '__main__':
    main()
