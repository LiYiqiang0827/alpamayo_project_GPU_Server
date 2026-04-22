#!/usr/bin/env python3
"""
提取4个摄像头图像 - 选择视频中间的时间点
"""
import os
import av
import numpy as np
import pandas as pd
from PIL import Image

def extract_frame_at_time(video_path: str, target_time_us: int):
    """从视频提取最接近目标时间戳的帧"""
    # 读取时间戳文件
    ts_path = video_path.replace('.mp4', '.timestamps.parquet')
    df = pd.read_parquet(ts_path)
    timestamps = df['timestamp'].values
    
    # 找到最接近目标时间的帧
    idx = np.argmin(np.abs(timestamps - target_time_us))
    actual_time = timestamps[idx]
    
    # 提取该帧
    container = av.open(video_path)
    frame_img = None
    for i, frame in enumerate(container.decode(video=0)):
        if i == idx:
            frame_img = frame.to_ndarray(format='rgb24')
            break
    container.close()
    
    return frame_img, actual_time, idx

def main():
    base_dir = "/data01/vla/data_sample_chunk0"
    clip_id = "fa83bcb8-ea31-4dbb-b447-4fb458f5984b"  # 最差样本
    
    # 获取视频时间范围，选择中间的时间点
    ts_path = f"{base_dir}/camera/camera_front_wide_120fov/{clip_id}.camera_front_wide_120fov.timestamps.parquet"
    df = pd.read_parquet(ts_path)
    ts = df['timestamp'].values
    
    video_start = ts[0]
    video_end = ts[-1]
    
    # 选择视频中间且有足够历史/未来的时间点
    # 需要 t0 - 1.6s > start 且 t0 + 6.4s < end
    t0_us = video_start + 7_900_000  # t0 = start + 7.9s
    
    print("=" * 70)
    print(f"提取4摄像头图像 - 最差场景样本")
    print("=" * 70)
    print(f"Clip ID: {clip_id}")
    print(f"视频时间范围: [{video_start/1e6:.2f}s, {video_end/1e6:.2f}s]")
    print(f"选择时间点: t0 = {t0_us/1e6:.1f}s")
    print()
    
    cameras = [
        ("camera_cross_left_120fov", "Left_Cross"),
        ("camera_front_wide_120fov", "Front_Wide"),
        ("camera_cross_right_120fov", "Right_Cross"),
        ("camera_front_tele_30fov", "Front_Tele"),
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
            frame, actual_time, idx = extract_frame_at_time(video_path, t0_us)
            
            if frame is not None:
                # 保存图像
                output_path = f"{output_dir}/{label}_{clip_id}.png"
                img = Image.fromarray(frame)
                img.save(output_path)
                time_diff = (actual_time - t0_us) / 1000  # ms
                print(f"  ✅ {label}: frame {idx}, time_offset={time_diff:+.1f}ms, saved to {output_path}")
            else:
                print(f"  ❌ {label}: 无法提取帧")
                
        except Exception as e:
            print(f"  ❌ {label}: {str(e)[:50]}")
    
    print(f"\n所有图像已保存到: {output_dir}")
    print("=" * 70)

if __name__ == "__main__":
    main()
