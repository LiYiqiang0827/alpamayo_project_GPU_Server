#!/usr/bin/env python3
"""
Step 1 & 2: 建立时间对齐索引 + GPU解码
修复版：正确按10Hz采样egomotion数据
"""
import pandas as pd
import numpy as np
import os
import argparse
import av
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import time

def preprocess_clip_v3(clip_id, chunk=0):
    """修复版：正确按10Hz采样egomotion"""
    BASE_DIR = f'/data01/vla/data/data_sample_chunk{chunk}'
    INFER_DIR = f'{BASE_DIR}/infer/{clip_id}'
    DATA_DIR = f'{INFER_DIR}/data'
    CAMERA_IMG_DIR = f'{DATA_DIR}/camera_images'
    EGO_DIR = f'{DATA_DIR}/egomotion'
    
    cameras = [
        ('camera_cross_left_120fov', 'cam_left'),
        ('camera_front_wide_120fov', 'cam_front'),
        ('camera_cross_right_120fov', 'cam_right'),
        ('camera_front_tele_30fov', 'cam_tele')
    ]
    
    NUM_FRAMES_PER_CAM = 4
    HISTORY_STEPS = 16  # 1.6s @ 10Hz
    FUTURE_STEPS = 64   # 6.4s @ 10Hz
    TIME_STEP = 0.1     # 10Hz
    JPG_QUALITY = 95
    MAX_TIME_DIFF_MS = 50
    
    print(f'=== 预处理 {clip_id} (v3 - 修复采样率) ===\n')
    
    os.makedirs(CAMERA_IMG_DIR, exist_ok=True)
    os.makedirs(EGO_DIR, exist_ok=True)
    
    # Step 1: 读取时间戳
    print('Step 1: 读取时间戳...')
    
    ego_path = f'{BASE_DIR}/labels/egomotion/{clip_id}.egomotion.parquet'
    ego_df = pd.read_parquet(ego_path)
    print(f'  Egomotion: {len(ego_df)} 帧, ~{1e6/ego_df["timestamp"].diff().mean():.1f} Hz')
    
    cam_timestamps = {}
    for cam_name, _ in cameras:
        ts_path = f'{BASE_DIR}/camera/{clip_id}.{cam_name}.timestamps.parquet'
        ts_df = pd.read_parquet(ts_path)
        cam_timestamps[cam_name] = ts_df['timestamp'].values
        print(f'  {cam_name}: {len(ts_df)} 帧')
    
    # Step 2: 建立时间对齐索引
    print('\nStep 2: 建立时间对齐索引...')
    
    valid_frames = []
    for ego_idx in range(HISTORY_STEPS, len(ego_df) - FUTURE_STEPS):
        ego_ts = ego_df.iloc[ego_idx]['timestamp']
        
        cam_frames = {}
        valid = True
        
        for cam_name, cam_prefix in cameras:
            ts_array = cam_timestamps[cam_name]
            
            valid_ts = ts_array[ts_array <= ego_ts]
            if len(valid_ts) == 0:
                valid = False
                break
                
            closest_idx = len(valid_ts) - 1
            closest_ts = valid_ts[-1]
            time_diff_ms = (ego_ts - closest_ts) / 1000.0
            
            if time_diff_ms > MAX_TIME_DIFF_MS:
                valid = False
                break
            
            if closest_idx < 3:
                valid = False
                break
                
            cam_frames[cam_name] = {
                'frame_idx': closest_idx,
                'timestamp': closest_ts,
                'time_diff_ms': time_diff_ms
            }
        
        if valid:
            valid_frames.append({
                'infer_idx': len(valid_frames),
                'ego_idx': ego_idx,
                't0_timestamp': ego_ts,
                **{f'{cam_prefix}_f0': f'camera_images/{cam_name}/{cam_frames[cam_name]["frame_idx"]-3:06d}.jpg' for cam_name, cam_prefix in cameras},
                **{f'{cam_prefix}_f1': f'camera_images/{cam_name}/{cam_frames[cam_name]["frame_idx"]-2:06d}.jpg' for cam_name, cam_prefix in cameras},
                **{f'{cam_prefix}_f2': f'camera_images/{cam_name}/{cam_frames[cam_name]["frame_idx"]-1:06d}.jpg' for cam_name, cam_prefix in cameras},
                **{f'{cam_prefix}_f3': f'camera_images/{cam_name}/{cam_frames[cam_name]["frame_idx"]:06d}.jpg' for cam_name, cam_prefix in cameras},
            })
    
    print(f'  有效对齐帧: {len(valid_frames)} / {len(ego_df)}')
    
    index_df = pd.DataFrame(valid_frames)
    index_df.to_csv(f'{DATA_DIR}/inference_index.csv', index=False)
    
    # Step 3: GPU解码视频
    print('\nStep 3: GPU解码视频...')
    
    def decode_video_gpu(cam_name):
        cam_img_dir = f'{CAMERA_IMG_DIR}/{cam_name}'
        os.makedirs(cam_img_dir, exist_ok=True)
        
        existing = len([f for f in os.listdir(cam_img_dir) if f.endswith('.jpg')])
        if existing >= 600:
            print(f'  {cam_name}: 已存在 {existing} 张，跳过')
            return cam_name, existing
        
        video_path = f'{BASE_DIR}/camera/{clip_id}.{cam_name}.mp4'
        print(f'  解码 {cam_name}...')
        start_time = time.time()
        
        frame_count = 0
        try:
            container = av.open(video_path)
            stream = container.streams.video[0]
            
            try:
                stream.codec_context.codec = av.Codec('h264_cuvid', 'r')
            except:
                pass
            
            for packet in container.demux(stream):
                for frame in packet.decode():
                    img = Image.fromarray(frame.to_ndarray(format='rgb24'))
                    img.save(f'{cam_img_dir}/{frame_count:06d}.jpg', 'JPEG', quality=JPG_QUALITY)
                    frame_count += 1
            
            container.close()
            elapsed = time.time() - start_time
            print(f'    完成: {frame_count} 帧, {elapsed:.1f}s')
            return cam_name, frame_count
        except Exception as e:
            print(f'  错误 {cam_name}: {e}')
            return cam_name, 0
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(decode_video_gpu, [c[0] for c in cameras]))
    
    total_frames = sum(r[1] for r in results)
    print(f'\n视频解码完成: 共 {total_frames} 帧')
    
    # Step 4: 保存egomotion数据 (修复版 - 按10Hz插值)
    print('\nStep 4: 保存 egomotion 数据 (按10Hz插值)...')
    
    for frame_info in valid_frames:
        ego_idx = frame_info['ego_idx']
        t0_ts = frame_info['t0_timestamp']
        prefix = f'ego_{frame_info["infer_idx"]:06d}'
        
        # 计算10Hz的时间戳 - 修复：官方方式
        # 历史: t0-1.5s, t0-1.4s, ..., t0-0.1s, t0 (16个点，包含t0)
        hist_offsets = np.arange(-(HISTORY_STEPS-1) * int(TIME_STEP * 1e6), 
                                  int(TIME_STEP * 1e6 / 2), 
                                  int(TIME_STEP * 1e6)).astype(np.int64)
        hist_timestamps = t0_ts + hist_offsets
        
        # 未来: t0+0.1s, t0+0.2s, ..., t0+6.4s (64个点，不包含t0)
        future_offsets = np.arange(1 * int(TIME_STEP * 1e6), 
                                   (FUTURE_STEPS + 0.5) * int(TIME_STEP * 1e6), 
                                   int(TIME_STEP * 1e6)).astype(np.int64)
        future_timestamps = t0_ts + future_offsets
        
        # 插值获取历史数据
        hist_xyz_world = []
        hist_quat = []
        for ts in hist_timestamps:
            # 找到最接近的时间戳
            idx = (ego_df['timestamp'] - ts).abs().idxmin()
            row = ego_df.loc[idx]
            hist_xyz_world.append(row[['x', 'y', 'z']].values)
            hist_quat.append(row[['qx', 'qy', 'qz', 'qw']].values)
        
        hist_xyz_world = np.array(hist_xyz_world)
        hist_rot = np.array([R.from_quat(q).as_matrix() for q in hist_quat])
        
        # 插值获取未来数据
        future_xyz_world = []
        for ts in future_timestamps:
            idx = (ego_df['timestamp'] - ts).abs().idxmin()
            row = ego_df.loc[idx]
            future_xyz_world.append(row[['x', 'y', 'z']].values)
        future_xyz_world = np.array(future_xyz_world)
        
        # 获取t0时刻姿态
        t0_row = ego_df.iloc[ego_idx]
        t0_xyz = t0_row[['x', 'y', 'z']].values
        t0_quat = t0_row[['qx', 'qy', 'qz', 'qw']].values
        t0_rot = R.from_quat(t0_quat).as_matrix()
        t0_rot_inv = t0_rot.T
        
        # 转换到局部坐标系
        hist_xyz_local = (hist_xyz_world - t0_xyz) @ t0_rot_inv
        hist_rot_local = np.array([t0_rot_inv @ r for r in hist_rot])
        future_xyz_local = (future_xyz_world - t0_xyz) @ t0_rot_inv
        
        # 保存
        np.save(f'{EGO_DIR}/{prefix}_history_world.npy', {
            'xyz': hist_xyz_world,
            'rotation_matrix': hist_rot
        })
        np.save(f'{EGO_DIR}/{prefix}_history_local.npy', {
            'xyz': hist_xyz_local,
            'rotation_matrix': hist_rot_local
        })
        np.save(f'{EGO_DIR}/{prefix}_future_gt.npy', {'xyz': future_xyz_local})
    
    print(f'Egomotion 保存完成: {len(valid_frames)} 帧')
    print(f'\n预处理完成: {DATA_DIR}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip', type=str, required=True)
    parser.add_argument('--chunk', type=int, default=0)
    args = parser.parse_args()
    
    preprocess_clip_v3(args.clip, args.chunk)

if __name__ == '__main__':
    main()
