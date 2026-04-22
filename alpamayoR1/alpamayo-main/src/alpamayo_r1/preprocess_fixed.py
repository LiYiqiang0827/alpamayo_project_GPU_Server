#!/usr/bin/env python3
"""
Step 2 修正版: 正确处理图像帧时间对齐
- 图像帧: t0-0.3s, t0-0.2s, t0-0.1s, t0 (按时间戳找最近帧)
- egomotion: t0-1.5s ~ t0 (16步), t0+0.1s ~ t0+6.4s (64步)
"""
import pandas as pd
import numpy as np
import os
import av
from pathlib import Path
import json
from scipy.spatial.transform import Rotation as R
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

CLIP_ID = '01d3588e-bca7-4a18-8e74-c6cfe9e996db'
BASE_DIR = '/data01/vla/data/data_sample_chunk0'
INFER_DIR = f'{BASE_DIR}/infer/{CLIP_ID}'
DATA_DIR = f'{INFER_DIR}/data'
CAMERA_IMG_DIR = f'{DATA_DIR}/camera_images'
EGO_DIR = f'{DATA_DIR}/egomotion'

# 相机配置
cameras = [
    'camera_cross_left_120fov',
    'camera_front_wide_120fov',
    'camera_cross_right_120fov',
    'camera_front_tele_30fov'
]

NUM_FRAMES_PER_CAM = 4      # 4帧: t0-0.3, t0-0.2, t0-0.1, t0
IMAGE_TIME_OFFSETS = [0.3, 0.2, 0.1, 0.0]  # 秒 (相对于t0往前)
HISTORY_STEPS = 16          # 1.6秒历史 (10Hz)
FUTURE_STEPS = 64           # 6.4秒未来 (10Hz)
TIME_STEP = 0.1             # 10Hz
JPG_QUALITY = 95

print('=== Step 2 修正版: 预处理推理数据 (正确时间对齐) ===\n')

# 创建目录
os.makedirs(CAMERA_IMG_DIR, exist_ok=True)
os.makedirs(EGO_DIR, exist_ok=True)

# ============================================
# Part 1: 解码所有视频帧到图片
# ============================================
print('Part 1: 解码视频帧到图片...\n')

for cam_name in cameras:
    cam_img_dir = f'{CAMERA_IMG_DIR}/{cam_name}'
    os.makedirs(cam_img_dir, exist_ok=True)
    
    existing_files = len([f for f in os.listdir(cam_img_dir) if f.endswith('.jpg')])
    if existing_files >= 600:
        print(f'  {cam_name}: 已存在 {existing_files} 张图片，跳过')
        continue
    
    video_path = f'{BASE_DIR}/camera/{CLIP_ID}.{cam_name}.mp4'
    print(f'  解码 {cam_name}...')
    start_time = time.time()
    
    try:
        container = av.open(video_path)
        stream = container.streams.video[0]
        total_frames = stream.frames
        
        frame_count = 0
        for frame_idx, frame in enumerate(container.decode(video=0)):
            img = frame.to_ndarray(format='rgb24')
            img_pil = Image.fromarray(img)
            img_path = f'{cam_img_dir}/{frame_idx:06d}.jpg'
            img_pil.save(img_path, 'JPEG', quality=JPG_QUALITY)
            frame_count += 1
            
            if (frame_idx + 1) % 100 == 0:
                elapsed = time.time() - start_time
                fps = (frame_idx + 1) / elapsed
                print(f'    已解码 {frame_idx + 1}/{total_frames} 帧, {fps:.1f} fps')
        
        container.close()
        elapsed = time.time() - start_time
        print(f'  {cam_name}: 完成 {frame_count} 帧')
        
    except Exception as e:
        print(f'  {cam_name}: 解码失败 - {e}')

print('\nPart 1 完成！\n')

# ============================================
# Part 2: 加载所有相机的时间戳
# ============================================
print('Part 2: 加载相机时间戳...\n')

cam_timestamps = {}
for cam_name in cameras:
    ts_file = f'{BASE_DIR}/camera/{CLIP_ID}.{cam_name}.timestamps.parquet'
    ts_df = pd.read_parquet(ts_file)
    cam_timestamps[cam_name] = {
        'timestamps': ts_df['timestamp'].values,
        'frame_indices': ts_df.index.values
    }
    print(f'  {cam_name}: {len(ts_df)} 帧, 时间戳范围 [{ts_df["timestamp"].min()}, {ts_df["timestamp"].max()}]')

# ============================================
# Part 3: 生成时间对齐的索引
# ============================================
print('\nPart 3: 生成时间对齐的索引...\n')

# 读取原始索引
index_df = pd.read_csv(f'{DATA_DIR}/infer_data_index.csv')
print(f'1. 加载原始索引: {len(index_df)} 行')

# 加载egomotion
ego_file = f'{BASE_DIR}/labels/egomotion/{CLIP_ID}.egomotion.parquet'
ego_df = pd.read_parquet(ego_file)
ego_timestamps_arr = ego_df['timestamp'].values
ego_xyz = ego_df[['x', 'y', 'z']].values
ego_quat = ego_df[['qx', 'qy', 'qz', 'qw']].values
print(f'2. 加载 egomotion: {len(ego_df)} 点')

# 对每个推理帧，计算正确的图像帧号
print(f'\n3. 处理 {len(index_df)} 个推理帧的时间对齐...')

def find_closest_frame(cam_name, target_ts):
    """找最接近目标时间戳的帧号"""
    cam_ts = cam_timestamps[cam_name]['timestamps']
    cam_frames = cam_timestamps[cam_name]['frame_indices']
    
    # 找最接近的帧
    idx = np.argmin(np.abs(cam_ts - target_ts))
    return cam_frames[idx], cam_ts[idx]

inference_records = []

for idx in range(len(index_df)):
    row = index_df.iloc[idx]
    infer_idx = int(row['infer_idx'])
    t0_us = int(row['ego_timestamp'])
    
    # A. 找4个时间点的图像帧 (t0-0.3s, t0-0.2s, t0-0.1s, t0)
    image_paths = {}
    image_frame_nums = {}  # 记录帧号用于验证
    
    for cam_name in cameras:
        cam_images = []
        cam_frame_nums = []
        
        for offset_sec in IMAGE_TIME_OFFSETS:
            target_ts = t0_us - int(offset_sec * 1_000_000)  # 微秒
            frame_num, actual_ts = find_closest_frame(cam_name, target_ts)
            
            img_path = f'camera_images/{cam_name}/{frame_num:06d}.jpg'
            cam_images.append(img_path)
            cam_frame_nums.append({
                'frame_num': frame_num,
                'target_ts': target_ts,
                'actual_ts': actual_ts,
                'diff_ms': (actual_ts - target_ts) / 1000
            })
        
        image_paths[cam_name] = cam_images
        image_frame_nums[cam_name] = cam_frame_nums
    
    # B. 提取 egomotion 数据
    # 历史: t0-1.5s ~ t0 (16步, 10Hz)
    history_offsets_us = np.arange(
        -(HISTORY_STEPS - 1) * TIME_STEP * 1_000_000,
        TIME_STEP * 1_000_000 / 2,
        TIME_STEP * 1_000_000,
    ).astype(np.int64)
    history_timestamps = t0_us + history_offsets_us
    
    # 未来: t0+0.1s ~ t0+6.4s (64步, 10Hz)
    future_offsets_us = np.arange(
        TIME_STEP * 1_000_000,
        (FUTURE_STEPS + 0.5) * TIME_STEP * 1_000_000,
        TIME_STEP * 1_000_000,
    ).astype(np.int64)
    future_timestamps = t0_us + future_offsets_us
    
    # 插值 egomotion
    def interpolate_ego(target_ts):
        xyz = np.zeros((len(target_ts), 3))
        quat = np.zeros((len(target_ts), 4))
        for i in range(3):
            xyz[:, i] = np.interp(target_ts, ego_timestamps_arr, ego_xyz[:, i])
        for i in range(4):
            quat[:, i] = np.interp(target_ts, ego_timestamps_arr, ego_quat[:, i])
        norms = np.linalg.norm(quat, axis=1, keepdims=True)
        quat = quat / norms
        return xyz, quat
    
    hist_xyz, hist_quat = interpolate_ego(history_timestamps)
    fut_xyz, fut_quat = interpolate_ego(future_timestamps)
    
    # C. 坐标转换到 t0 坐标系
    t0_xyz = hist_xyz[-1].copy()
    t0_quat = hist_quat[-1].copy()
    t0_rot = R.from_quat(t0_quat)
    t0_rot_inv = t0_rot.inv()
    
    hist_xyz_local = t0_rot_inv.apply(hist_xyz - t0_xyz)
    hist_rot_local = (t0_rot_inv * R.from_quat(hist_quat)).as_matrix()
    
    fut_xyz_local = t0_rot_inv.apply(fut_xyz - t0_xyz)
    fut_rot_local = (t0_rot_inv * R.from_quat(fut_quat)).as_matrix()
    
    # D. 保存 Egomotion
    ego_filename = f'ego_{infer_idx:06d}'
    
    np.save(f'{EGO_DIR}/{ego_filename}_history_world.npy', {
        'xyz': hist_xyz, 'quat': hist_quat, 'timestamps': history_timestamps
    })
    np.save(f'{EGO_DIR}/{ego_filename}_history_local.npy', {
        'xyz': hist_xyz_local, 'rotation_matrix': hist_rot_local, 'timestamps': history_timestamps
    })
    np.save(f'{EGO_DIR}/{ego_filename}_future_gt.npy', {
        'xyz': fut_xyz_local, 'rotation_matrix': fut_rot_local, 'timestamps': future_timestamps
    })
    
    t0_info = {
        't0_timestamp': int(t0_us),
        't0_position_world': [float(x) for x in t0_xyz],
        't0_quaternion_world': [float(x) for x in t0_quat],
        't0_rotation_matrix': [[float(x) for x in row] for row in t0_rot.as_matrix()]
    }
    with open(f'{EGO_DIR}/{ego_filename}_t0.json', 'w') as f:
        json.dump(t0_info, f, indent=2)
    
    # 保存图像帧信息
    with open(f'{EGO_DIR}/{ego_filename}_image_frames.json', 'w') as f:
        json.dump({
            't0_timestamp': int(t0_us),
            'camera_frames': {k: [{kk: int(vv) if isinstance(vv, (np.integer, np.int64)) else float(vv) if isinstance(vv, (np.floating, np.float64)) else vv for kk, vv in item.items()} for item in v] for k, v in image_frame_nums.items()}
        }, f, indent=2)
    
    # E. 记录到索引表
    record = {
        'infer_idx': infer_idx,
        't0_timestamp': t0_us,
        'ego_file_prefix': ego_filename,
        # 图片路径 (按相机顺序)
        'cam_left_f0': image_paths['camera_cross_left_120fov'][0],  # t0-0.3s
        'cam_left_f1': image_paths['camera_cross_left_120fov'][1],  # t0-0.2s
        'cam_left_f2': image_paths['camera_cross_left_120fov'][2],  # t0-0.1s
        'cam_left_f3': image_paths['camera_cross_left_120fov'][3],  # t0
        'cam_front_f0': image_paths['camera_front_wide_120fov'][0],
        'cam_front_f1': image_paths['camera_front_wide_120fov'][1],
        'cam_front_f2': image_paths['camera_front_wide_120fov'][2],
        'cam_front_f3': image_paths['camera_front_wide_120fov'][3],
        'cam_right_f0': image_paths['camera_cross_right_120fov'][0],
        'cam_right_f1': image_paths['camera_cross_right_120fov'][1],
        'cam_right_f2': image_paths['camera_cross_right_120fov'][2],
        'cam_right_f3': image_paths['camera_cross_right_120fov'][3],
        'cam_tele_f0': image_paths['camera_front_tele_30fov'][0],
        'cam_tele_f1': image_paths['camera_front_tele_30fov'][1],
        'cam_tele_f2': image_paths['camera_front_tele_30fov'][2],
        'cam_tele_f3': image_paths['camera_front_tele_30fov'][3],
    }
    inference_records.append(record)
    
    if (idx + 1) % 100 == 0 or idx == len(index_df) - 1:
        print(f'  已处理 {idx + 1}/{len(index_df)}: Frame_{infer_idx:06d}')

# 保存索引表
index_table = pd.DataFrame(inference_records)
csv_path = f'{DATA_DIR}/inference_index_v2.csv'
index_table.to_csv(csv_path, index=False)

print(f'\n4. 索引表已保存: {csv_path}')
print(f'   总行数: {len(index_table)}')

# 验证时间对齐
print(f'\n5. 验证时间对齐...')
sample_idx = 0
sample_record = inference_records[sample_idx]
print(f'   示例 Frame {sample_record["infer_idx"]}:')
print(f'   t0_timestamp: {sample_record["t0_timestamp"]}')

with open(f'{EGO_DIR}/{sample_record["ego_file_prefix"]}_image_frames.json', 'r') as f:
    frame_info = json.load(f)
    for cam_name in cameras[:2]:  # 只看前2个相机
        print(f'   {cam_name}:')
        for i, frame_data in enumerate(frame_info['camera_frames'][cam_name]):
            print(f'     f{i} (t0-{IMAGE_TIME_OFFSETS[i]*1000:.0f}ms): frame={frame_data["frame_num"]}, diff={frame_data["diff_ms"]:.1f}ms')

print(f'\n=== 全部完成！===')
print(f'图像帧时间对齐: t0-300ms, t0-200ms, t0-100ms, t0')
print(f'数据位置: {DATA_DIR}/')
