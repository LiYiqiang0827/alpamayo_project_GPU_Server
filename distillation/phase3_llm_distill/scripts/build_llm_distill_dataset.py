#!/usr/bin/env python3
"""
构建LLM蒸馏数据集
从infer_results_all.csv提取Teacher CoT输出，构建(input, target)训练对
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

# 添加Alpamayo源码路径
sys.path.insert(0, "/home/user/mikelee/alpamayo_project/alpamayo_1_5/alpamayo1.5-main/src")

from transformers import AutoTokenizer, AutoProcessor

# 路径配置
DATA_BASE = "/data01/mikelee/data"
INFER_RESULT_PATH = "/data01/mikelee/infer_result/infer_result_20260424_161448/infer_results_all.csv"
OUTPUT_DIR = "/gpfs-data/mikelee/llm_distillation_data"

# 扩展后的tokenizer路径
TOKENIZER_PATH = "/data01/mikelee/weight/alpamayo2B/tokenizer_final"

# 相机配置（与Alpamayo1.5一致）
CAMERAS = [
    "camera_cross_left_120fov",
    "camera_front_wide_120fov", 
    "camera_cross_right_120fov",
    "camera_front_tele_30fov",
]

# 时间帧配置（4个历史帧）
TIME_FRAMES = [0, 1, 2, 3]  # 对应t0, t-1, t-2, t-3（或类似的时序）


def load_tokenizer():
    """加载扩展后的tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    print(f"Tokenizer loaded, vocab size: {len(tokenizer)}")
    return tokenizer


def build_prompt_with_cot(tokenizer, cot_text, has_navigation=False, nav_text=None):
    """
    构建Alpamayo1.5格式的prompt，包含CoT输出作为assistant的回复
    
    这是用于蒸馏训练的目标格式：
    - User: 图片 + 历史轨迹 + 导航指令 + "output the chain-of-thought..."
    - Assistant: <|cot_start|>CoT文本<|cot_end|>
    
    训练时，模型学习预测assistant的回复（即CoT）
    """
    
    # System message
    system_msg = {
        "role": "system",
        "content": "You are a driving assistant that generates safe and accurate actions."
    }
    
    # User message: 包含图片占位符和轨迹占位符
    # 注意：实际训练时，图片和轨迹会在模型forward时注入
    user_content = []
    
    # 添加图片占位符（16张图）
    for i in range(16):
        user_content.append({"type": "image", "image": None})  # 占位，实际会替换
    
    # 添加轨迹占位符
    traj_text = "<|traj_history_start|>" + "<|traj_history|>" * 48 + "<|traj_history_end|>"
    
    # 添加导航指令（如果有）
    if has_navigation and nav_text:
        route_text = f"<|route_start|>{nav_text}<|route_end|>"
    else:
        route_text = ""
    
    # 添加prompt文本
    prompt_text = f"{traj_text}{route_text}output the chain-of-thought reasoning of the driving process, then output the future trajectory."
    
    user_content.append({"type": "text", "text": prompt_text})
    
    user_msg = {
        "role": "user",
        "content": user_content
    }
    
    # Assistant message: CoT输出（这是训练目标！）
    assistant_msg = {
        "role": "assistant", 
        "content": f"<|cot_start|>{cot_text}<|cot_end|>"
    }
    
    messages = [system_msg, user_msg, assistant_msg]
    
    return messages


def load_frame_data(chunk_id, clip_id, frame_id, data_base=DATA_BASE):
    """
    加载指定帧的数据
    
    返回:
        images: list of PIL Images (16张)
        history_traj: np.array (16, 7) [x,y,z,qx,qy,qz,qw]
    """
    clip_dir = f"{data_base}/data_sample_chunk{chunk_id}/infer/{clip_id}/data"
    
    # 加载图片（使用预缩放的_small.jpg）
    images = []
    for cam_name in CAMERAS:
        cam_dir = f"{clip_dir}/camera_images/{cam_name}"
        # 需要根据frame_id找到对应的图片文件
        # 图片文件名格式: {frame:06d}_small.jpg
        img_path = f"{cam_dir}/{frame_id:06d}_small.jpg"
        if not os.path.exists(img_path):
            # 尝试不带_small后缀
            img_path = f"{cam_dir}/{frame_id:06d}.jpg"
        
        if os.path.exists(img_path):
            img = Image.open(img_path).convert('RGB')
            images.append(img)
        else:
            print(f"Warning: Image not found: {img_path}")
            # 使用空白图占位
            images.append(Image.new('RGB', (576, 320), color='black'))
    
    # 加载历史轨迹
    hist_path = f"{clip_dir}/egomotion/frame_{frame_id:06d}_history.npy"
    if os.path.exists(hist_path):
        history_traj = np.load(hist_path)  # (16, 7)
    else:
        print(f"Warning: History not found: {hist_path}")
        history_traj = np.zeros((16, 7), dtype=np.float32)
    
    return images, history_traj


def encode_trajectory_to_tokens(history_traj, tokenizer, traj_token_start_idx=151669):
    """
    将历史轨迹编码为token IDs
    
    简化版本：直接使用预计算的轨迹token（如果可用）
    或者使用DeltaTrajectoryTokenizer编码
    """
    # TODO: 实现轨迹编码
    # 暂时返回占位符token ID
    traj_tokens = [tokenizer.convert_tokens_to_ids("<|traj_history|>")] * 48
    return traj_tokens


def build_distillation_dataset(
    infer_csv_path,
    output_dir,
    data_base=DATA_BASE,
    max_samples=None,
    chunk_filter=None,
):
    """
    构建蒸馏数据集
    
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
    
    print(f"Loading inference results from: {infer_csv_path}")
    df = pd.read_csv(infer_csv_path)
    print(f"Total samples: {len(df)}")
    
    # 过滤chunk
    if chunk_filter is not None:
        df = df[df['chunk_id'].isin(chunk_filter)]
        print(f"After chunk filter: {len(df)}")
    
    # 限制样本数
    if max_samples is not None:
        df = df.head(max_samples)
        print(f"Limited to: {len(df)} samples")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建数据集
    dataset = []
    skipped = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Building dataset"):
        chunk_id = row['chunk_id']
        clip_id = row['clip_name']
        frame_id = row['frame_number']
        cot_text = row['cot_result']
        
        # 检查数据是否存在
        clip_dir = f"{data_base}/data_sample_chunk{chunk_id}/infer/{clip_id}/data"
        if not os.path.exists(clip_dir):
            skipped += 1
            continue
        
        # 收集图片路径（4相机 × 4时间帧 = 16张图）
        image_paths = []
        valid = True
        
        # 需要读取inference_index_strict.csv来确定每个时间帧对应的图片索引
        index_csv = f"{clip_dir}/inference_index_strict.csv"
        frame_idx = None
        if os.path.exists(index_csv):
            try:
                index_df = pd.read_csv(index_csv)
                # 找到对应frame_id的行
                frame_row = index_df[index_df['frame_id'] == frame_id]
                if len(frame_row) > 0:
                    frame_row = frame_row.iloc[0]
                else:
                    valid = False
            except Exception as e:
                print(f"Warning: Error reading index CSV: {e}")
                valid = False
        else:
            valid = False
        
        if valid:
            # 对每个相机，加载4个时间帧的图片
            for cam_name in CAMERAS:
                for t_idx in TIME_FRAMES:
                    # 从index_csv获取该相机该时间帧的图片索引
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
                        # 如果列不存在，尝试直接使用frame_id
                        img_path = f"{clip_dir}/camera_images/{cam_name}/{frame_id:06d}_small.jpg"
                        if not os.path.exists(img_path):
                            img_path = f"{clip_dir}/camera_images/{cam_name}/{frame_id:06d}.jpg"
                        
                        if os.path.exists(img_path):
                            image_paths.append(img_path)
                        else:
                            valid = False
                            break
                if not valid:
                    break
        
        # 如果没有index_csv或者失败了，尝试简化版本：只用frame_id
        if not valid:
            image_paths = []
            valid = True
            for cam_name in CAMERAS:
                # 只加载一张图（简化版本）
                img_path = f"{clip_dir}/camera_images/{cam_name}/{frame_id:06d}_small.jpg"
                if not os.path.exists(img_path):
                    img_path = f"{clip_dir}/camera_images/{cam_name}/{frame_id:06d}.jpg"
                
                if os.path.exists(img_path):
                    image_paths.append(img_path)
                else:
                    valid = False
                    break
            
            # 如果只有4张图，复制4份来模拟16张（临时方案）
            if valid and len(image_paths) == 4:
                image_paths = image_paths * 4  # 复制4次，得到16张
        
        if not valid:
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
    output_path = f"{output_dir}/distillation_dataset.jsonl"
    with open(output_path, 'w') as f:
        for sample in dataset:
            f.write(json.dumps(sample) + '\n')
    
    # 保存元数据
    metadata = {
        "created_at": datetime.now().isoformat(),
        "total_samples": len(dataset),
        "skipped_samples": skipped,
        "source_csv": infer_csv_path,
        "data_base": data_base,
        "chunk_filter": chunk_filter,
    }
    
    metadata_path = f"{output_dir}/metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nDataset built successfully!")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Skipped: {skipped}")
    print(f"  Output: {output_path}")
    print(f"  Metadata: {metadata_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Build LLM distillation dataset")
    parser.add_argument("--infer-csv", default=INFER_RESULT_PATH, help="Path to inference results CSV")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--data-base", default=DATA_BASE, help="Base data directory")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to process")
    parser.add_argument("--chunks", type=int, nargs='+', default=None, help="Filter by chunk IDs")
    
    args = parser.parse_args()
    
    build_distillation_dataset(
        infer_csv_path=args.infer_csv,
        output_dir=args.output_dir,
        data_base=args.data_base,
        max_samples=args.max_samples,
        chunk_filter=args.chunks,
    )


if __name__ == "__main__":
    main()
