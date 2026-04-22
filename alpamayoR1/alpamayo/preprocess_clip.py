#!/usr/bin/env python3
"""
Step 2 GPU加速版: 使用 NVIDIA GPU 硬件解码 + JPG 格式
支持命令行参数指定 clip_id
"""
import pandas as pd
import numpy as np
import os
import argparse
import av
from pathlib import Path
import json
from scipy.spatial.transform import Rotation as R
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def preprocess_clip(clip_id, chunk=0):
    """预处理单个 clip"""
    BASE_DIR = f'/data01/vla/data/data_sample_chunk{chunk}'
    INFER_DIR = f'{BASE_DIR}/infer/{clip_id}'
    DATA_DIR = f'{INFER_DIR}/data'
    CAMERA_IMG_DIR = f'{DATA_DIR}/camera_images'
    EGO_DIR = f'{DATA_DIR}/egomotion'
    
    # 相机配置
    cameras = [
        ('camera_cross_left_120fov', 0),
        ('camera_front_wide_120fov', 1),
        ('camera_cross_right_120fov', 2),
        ('camera_front_tele_30fov', 6)
    ]
    
    NUM_FRAMES_PER_CAM = 4
    HISTORY_STEPS = 16
    FUTURE_STEPS = 64
    TIME_STEP = 0.1
    JPG_QUALITY = 95
    
    print(f'=== 预处理 {clip_id} ===\n')
    print(f'设备: NVIDIA GPU 硬件解码 (h264_cuvid)')
    print(f'图片格式: JPG (质量{JPG_QUALITY})\n')
    
    # 创建目录
    os.makedirs(CAMERA_IMG_DIR, exist_ok=True)
    os.makedirs(EGO_DIR, exist_ok=True)
    
    def decode_video_gpu(cam_name):
        """使用 GPU 硬件解码视频"""
        cam_img_dir = f'{CAMERA_IMG_DIR}/{cam_name}'
        os.makedirs(cam_img_dir, exist_ok=True)
        
        # 检查是否已完成
        existing_files = len([f for f in os.listdir(cam_img_dir) if f.endswith('.jpg')])
        if existing_files >= 600:
            print(f'  {cam_name}: 已存在 {existing_files} 张图片，跳过')
            return cam_name, 0
        
        video_path = f'{BASE_DIR}/camera/{clip_id}.{cam_name}.mp4'
        print(f'  开始 GPU 解码 {cam_name}...')
        start_time = time.time()
        
        frame_count = 0
        try:
            container = av.open(video_path)
            stream = container.streams.video[0]
            
            # 尝试使用 GPU 解码器
            codec_name = stream.codec_context.codec.name
            if codec_name == 'h264':
                try:
                    stream.codec_context.codec = av.Codec('h264_cuvid', 'r')
                    print(f'    使用 GPU 硬件解码: h264_cuvid')
                except Exception as e:
                    print(f'    GPU 解码不可用，使用 CPU: {e}')
            
            for packet in container.demux(stream):
                for frame in packet.decode():
                    img = frame.to_ndarray(format='rgb24')
                    img_pil = Image.fromarray(img)
                    img_pil.save(f'{cam_img_dir}/{frame_count:06d}.jpg', 'JPEG', quality=JPG_QUALITY)
                    frame_count += 1
                    
            container.close()
            elapsed = time.time() - start_time
            print(f'    完成: {frame_count} 帧, {elapsed:.1f}s, {frame_count/elapsed:.1f} fps')
            return cam_name, frame_count
            
        except Exception as e:
            print(f'  错误 {cam_name}: {e}')
            return cam_name, 0
    
    # Step 1: 解码视频 (并行)
    print('Step 1/2: GPU解码视频...')
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(decode_video_gpu, cam[0]) for cam in cameras]
        results = [f.result() for f in futures]
    
    total_frames = sum(r[1] for r in results)
    print(f'\n视频解码完成: 共 {total_frames} 帧\n')
    
    # Step 2: 处理 egomotion
    print('Step 2/2: 处理 egomotion 数据...')
    ego_path = f'{BASE_DIR}/labels/egomotion/{clip_id}.egomotion.parquet'
    ego_df = pd.read_parquet(ego_path)
    
    # 保存索引表
    # 简化版：直接保存完整 egomotion
    infer_index = []
    for idx, row in ego_df.iterrows():
        if idx < HISTORY_STEPS or idx >= len(ego_df) - FUTURE_STEPS:
            continue
            
        infer_index.append({
            'infer_idx': len(infer_index),
            't0_timestamp': row['timestamp'],
            'ego_idx': idx,
            'cam_left_f0': f'camera_images/camera_cross_left_120fov/{max(0, (idx - HISTORY_STEPS) * 3):06d}.jpg',
            'cam_front_f0': f'camera_images/camera_front_wide_120fov/{max(0, (idx - HISTORY_STEPS) * 3):06d}.jpg',
            'cam_right_f0': f'camera_images/camera_cross_right_120fov/{max(0, (idx - HISTORY_STEPS) * 3):06d}.jpg',
            'cam_tele_f0': f'camera_images/camera_front_tele_30fov/{max(0, (idx - HISTORY_STEPS) * 3):06d}.jpg',
        })
    
    index_df = pd.DataFrame(infer_index)
    index_df.to_csv(f'{DATA_DIR}/inference_index.csv', index=False)
    print(f'索引表保存: {len(index_df)} 帧')
    
    # 保存 egomotion 数据
    for idx in range(HISTORY_STEPS, len(ego_df) - FUTURE_STEPS):
        prefix = f'ego_{idx:06d}'
        
        # 历史数据
        hist = ego_df.iloc[idx - HISTORY_STEPS:idx]
        hist_xyz = hist[['x', 'y', 'z']].values
        
        # 未来数据
        future = ego_df.iloc[idx:idx + FUTURE_STEPS]
        future_xyz = future[['x', 'y', 'z']].values
        
        np.save(f'{EGO_DIR}/{prefix}_history_world.npy', {'xyz': hist_xyz})
        np.save(f'{EGO_DIR}/{prefix}_future_gt.npy', {'xyz': future_xyz})
    
    print(f'Egomotion 保存完成: {len(ego_df) - HISTORY_STEPS - FUTURE_STEPS} 帧')
    print(f'\n预处理完成: {DATA_DIR}')

def main():
    parser = argparse.ArgumentParser(description='Step 2: GPU加速数据预处理')
    parser.add_argument('--clip', type=str, required=True, help='Clip ID')
    parser.add_argument('--chunk', type=int, default=0, help='Chunk number (default: 0)')
    args = parser.parse_args()
    
    preprocess_clip(args.clip, args.chunk)

if __name__ == '__main__':
    main()
