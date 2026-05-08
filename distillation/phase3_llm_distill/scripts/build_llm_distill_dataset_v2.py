#!/usr/bin/env python3
"""
构建LLM蒸馏数据集 - 优化版本
分批处理，避免内存问题
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

# 路径配置
DATA_BASE = "/data01/mikelee/data"
INFER_RESULT_PATH = "/data01/mikelee/infer_result/infer_result_20260424_161448/infer_results_all.csv"
OUTPUT_DIR = "/gpfs-data/mikelee/llm_distillation_data"

# 相机配置（与Alpamayo1.5一致）
CAMERAS = [
    "camera_cross_left_120fov",
    "camera_front_wide_120fov", 
    "camera_cross_right_120fov",
    "camera_front_tele_30fov",
]

# 时间帧配置
TIME_FRAMES = [0, 1, 2, 3]


def build_distillation_dataset_chunk(
    infer_df,
    chunk_id,
    output_dir,
    data_base=DATA_BASE,
):
    """
    构建单个chunk的蒸馏数据集
    
    输出格式（JSON Lines）:
    {
        "chunk_id": int,
        "clip_id": str,
        "frame_id": int,
        "cot_text": str,
        "image_paths": [str],  # 16张图片的路径
        "history_traj_path": str,
    }
    """
    
    # 过滤当前chunk
    chunk_df = infer_df[infer_df['chunk_id'] == chunk_id]
    print(f"Chunk {chunk_id}: {len(chunk_df)} samples")
    
    if len(chunk_df) == 0:
        return 0
    
    # 构建数据集
    dataset = []
    skipped = 0
    
    for idx, row in tqdm(chunk_df.iterrows(), total=len(chunk_df), desc=f"Chunk {chunk_id}"):
        clip_id = row['clip_name']
        frame_id = row['frame_number']
        cot_text = row['cot_result']
        
        # 检查数据是否存在
        clip_dir = f"{data_base}/data_sample_chunk{chunk_id}/infer/{clip_id}/data"
        if not os.path.exists(clip_dir):
            skipped += 1
            continue
        
        # 读取index csv来确定图片索引
        index_csv = f"{clip_dir}/inference_index_strict.csv"
        frame_row = None
        
        if os.path.exists(index_csv):
            try:
                index_df = pd.read_csv(index_csv)
                frame_rows = index_df[index_df['frame_id'] == frame_id]
                if len(frame_rows) > 0:
                    frame_row = frame_rows.iloc[0]
            except Exception as e:
                pass
        
        # 收集图片路径
        image_paths = []
        valid = True
        
        if frame_row is not None:
            # 使用index csv中的索引
            for cam_name in CAMERAS:
                for t_idx in TIME_FRAMES:
                    col_name = f"{cam_name}_f{t_idx}_idx"
                    if col_name in frame_row.index:
                        img_idx = int(frame_row[col_name])
                        img_path = f"{clip_dir}/camera_images/{cam_name}/{img_idx:06d}_small.jpg"
                        if not os.path.exists(img_path):
                            img_path = f"{clip_dir}/camera_images/{cam_name}/{img_idx:06d}.jpg"
                        
                        if os.path.exists(img_path):
                            image_paths.append(img_path)
                        else:
                            valid = False
                            break
                    else:
                        valid = False
                        break
                if not valid:
                    break
        else:
            #  fallback：直接使用frame_id
            for cam_name in CAMERAS:
                img_path = f"{clip_dir}/camera_images/{cam_name}/{frame_id:06d}_small.jpg"
                if not os.path.exists(img_path):
                    img_path = f"{clip_dir}/camera_images/{cam_name}/{frame_id:06d}.jpg"
                
                if os.path.exists(img_path):
                    image_paths.append(img_path)
                else:
                    valid = False
                    break
            
            # 如果只有4张图，复制4份来模拟16张
            if valid and len(image_paths) == 4:
                image_paths = image_paths * 4
        
        if not valid or len(image_paths) != 16:
            skipped += 1
            continue
        
        # 历史轨迹路径
        hist_path = f"{clip_dir}/egomotion/frame_{frame_id:06d}_history.npy"
        
        # 构建样本
        sample = {
            "chunk_id": int(chunk_id),
            "clip_id": clip_id,
            "frame_id": int(frame_id),
            "cot_text": cot_text,
            "image_paths": image_paths,
            "history_traj_path": hist_path if os.path.exists(hist_path) else None,
        }
        
        dataset.append(sample)
    
    # 保存数据集
    if len(dataset) > 0:
        output_path = f"{output_dir}/distillation_dataset_chunk{chunk_id}.jsonl"
        with open(output_path, 'w') as f:
            for sample in dataset:
                f.write(json.dumps(sample) + '\n')
        
        print(f"Chunk {chunk_id}: Saved {len(dataset)} samples to {output_path}")
    
    return len(dataset)


def merge_datasets(output_dir, chunks):
    """合并所有chunk的数据集"""
    
    print("\nMerging datasets...")
    
    all_samples = []
    for chunk_id in chunks:
        chunk_file = f"{output_dir}/distillation_dataset_chunk{chunk_id}.jsonl"
        if os.path.exists(chunk_file):
            with open(chunk_file, 'r') as f:
                for line in f:
                    all_samples.append(json.loads(line))
    
    # 保存合并后的数据集
    output_path = f"{output_dir}/distillation_dataset.jsonl"
    with open(output_path, 'w') as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + '\n')
    
    # 保存元数据
    metadata = {
        "created_at": datetime.now().isoformat(),
        "total_samples": len(all_samples),
        "chunks": chunks,
        "source_csv": INFER_RESULT_PATH,
        "data_base": DATA_BASE,
    }
    
    metadata_path = f"{output_dir}/metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nMerged dataset saved: {output_path}")
    print(f"Total samples: {len(all_samples)}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Build LLM distillation dataset")
    parser.add_argument("--infer-csv", default=INFER_RESULT_PATH, help="Path to inference results CSV")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--data-base", default=DATA_BASE, help="Base data directory")
    parser.add_argument("--chunks", type=int, nargs='+', default=None, help="Process specific chunks")
    parser.add_argument("--merge-only", action="store_true", help="Only merge existing chunk files")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.merge_only:
        # 只合并已有的chunk文件
        chunk_files = sorted(Path(args.output_dir).glob("distillation_dataset_chunk*.jsonl"))
        chunks = [int(f.stem.replace("distillation_dataset_chunk", "")) for f in chunk_files]
        merge_datasets(args.output_dir, chunks)
        return
    
    # 加载推理结果
    print(f"Loading inference results from: {args.infer_csv}")
    infer_df = pd.read_csv(args.infer_csv)
    print(f"Total inference results: {len(infer_df)}")
    
    # 确定要处理的chunks
    if args.chunks:
        chunks = args.chunks
    else:
        chunks = sorted(infer_df['chunk_id'].unique())
    
    print(f"Processing chunks: {chunks}")
    
    # 处理每个chunk
    total_samples = 0
    for chunk_id in chunks:
        n_samples = build_distillation_dataset_chunk(
            infer_df=infer_df,
            chunk_id=chunk_id,
            output_dir=args.output_dir,
            data_base=args.data_base,
        )
        total_samples += n_samples
    
    # 合并数据集
    merge_datasets(args.output_dir, chunks)
    
    print(f"\nDataset building completed!")
    print(f"Total samples: {total_samples}")


if __name__ == "__main__":
    main()
