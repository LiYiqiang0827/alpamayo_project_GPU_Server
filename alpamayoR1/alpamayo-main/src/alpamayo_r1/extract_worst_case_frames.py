#!/usr/bin/env python3
"""
提取特定时间点的4个摄像头图像
"""
import os
import av
import numpy as np
import pandas as pd
from PIL import Image
from einops import rearrange

def extract_frame_at_time(video_path: str, target_time_ms: float):
    """从视频提取特定时间点的帧"""
    container = av.open(video_path)
    stream = container.streams.video[0]
    
    # 获取视频信息
    fps = stream.average_rate
    duration = stream.duration * stream.time_base if stream.duration else None
    
    # 计算目标帧索引
    target_frame = int(target_time_ms / 1000 * fps)
    
    frame_img = None
    for i, frame in enumerate(container.decode(video=0)):
        if i == target_frame:
            frame_img = frame.to_ndarray(format='rgb24')
            break
    
    container.close()
    return frame_img, target_frame

def main():
    base_dir = "/data01/vla/data_sample_chunk0"
    clip_id = "fa83bcb8-ea31-4dbb-b447-4fb458f5984b"  # 最差样本
    t0_us = 20_900_000  # t=20.9s (minADE=11.23m的时间点)
    
    print("=" * 70)
    print(f"提取最差场景的4摄像头图像")
    print("=" * 70)
    print(f"Clip ID: {clip_id}")
    print(f"时间点: t={t0_us/1e6:.1f}s")
    print()
    
    cameras = [
        ("camera_cross_left_120fov", "Left"),
        ("camera_front_wide_120fov", "Front"),
        ("camera_cross_right_120fov", "Right"),
        ("camera_front_tele_30fov", "Tele"),
    ]
    
    output_dir = "/data01/vla/worst_case_frames"
    os.makedirs(output_dir, exist_ok=True)
    
    print("提取图像...")
    for cam_name, label in cameras:
        video_path = f"{base_dir}/camera/{cam_name}/{clip_id}.{cam_name}.mp4"
        
        if not os.path.exists(video_path):
            print(f"  ❌ {label}: 视频不存在")
            continue
        
        try:
            # 计算相对于视频开始的时间
            # 需要找到视频第一帧的时间戳
            ts_path = video_path.replace('.mp4', '.timestamps.parquet')
            if os.path.exists(ts_path):
                df = pd.read_parquet(ts_path)
                first_ts = df['timestamp'].values[0]
                # 目标时间相对于第一帧的偏移
                target_offset_ms = (t0_us - first_ts) / 1000.0
            else:
                target_offset_ms = 20_900  # 假设从0开始
            
            frame, frame_idx = extract_frame_at_time(video_path, target_offset_ms)
            
            if frame is not None:
                # 保存图像
                output_path = f"{output_dir}/{label}_{clip_id}.png"
                img = Image.fromarray(frame)
                img.save(output_path)
                print(f"  ✅ {label}: {output_path} (frame {frame_idx})")
            else:
                print(f"  ❌ {label}: 无法提取帧")
                
        except Exception as e:
            print(f"  ❌ {label}: {str(e)[:50]}")
    
    print(f"\n图像已保存到: {output_dir}")
    print("=" * 70)

if __name__ == "__main__":
    main()
