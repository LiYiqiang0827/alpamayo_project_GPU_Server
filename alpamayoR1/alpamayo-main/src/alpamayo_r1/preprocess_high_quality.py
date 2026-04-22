#!/usr/bin/env python3
"""
Step 2 高质量版: 自动过滤边界帧
- 图像帧时间差 <= 33ms (1帧@30Hz)
- 4个摄像头都有有效数据
- 历史+未来egomotion完整
"""
import pandas as pd
import numpy as np
import os
import json
from scipy.spatial.transform import Rotation as R
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

CLIP_ID = '01d3588e-bca7-4a18-8e74-c6cfe9e996db'
BASE_DIR = '/data01/vla/data/data_sample_chunk0'
INFER_DIR = f'{BASE_DIR}/infer/{CLIP_ID}'
DATA_DIR = f'{INFER_DIR}/data'
CAMERA_IMG_DIR = f'{DATA_DIR}/camera_images'
EGO_DIR = f'{DATA_DIR}/egomotion'

# 配置
cameras = [
    'camera_cross_left_120fov',
    'camera_front_wide_120fov',
    'camera_cross_right_120fov',
    'camera_front_tele_30fov'
]

NUM_FRAMES_PER_CAM = 4
IMAGE_TIME_OFFSETS = [0.3, 0.2, 0.1, 0.0]  # 秒
MAX_TIME_DIFF_MS = 33  # 最大允许时间差 33ms (1帧@30Hz)

HISTORY_STEPS = 16
FUTURE_STEPS = 64
TIME_STEP = 0.1

print('=== Step 2 高质量版: 过滤边界帧 ===\n')

# 创建目录
os.makedirs(CAMERA_IMG_DIR, exist_ok=True)
os.makedirs(EGO_DIR, exist_ok=True)

# ============================================
# Part 1: 加载相机时间戳
# ============================================
print('Part 1: 加载相机时间戳...\n')

cam_timestamps = {}
for cam_name in cameras:
    ts_file = f'{BASE_DIR}/camera/{CLIP_ID}.{cam_name}.timestamps.parquet'
    ts_df = pd.read_parquet(ts_file)
    cam_timestamps[cam_name] = {
        'timestamps': ts_df['timestamp'].values,
        'frame_indices': ts_df.index.values,
        'min_ts': ts_df['timestamp'].min(),
        'max_ts': ts_df['timestamp'].max()
    }
    print(f'  {cam_name}: {len(ts_df)} 帧, 范围 [{ts_df["timestamp"].min()}, {ts_df["timestamp"].max()}]')

# ============================================
# Part 2: 加载egomotion确定有效范围
# ============================================
print('\nPart 2: 加载egomotion...')

ego_file = f'{BASE_DIR}/labels/egomotion/{CLIP_ID}.egomotion.parquet'
ego_df = pd.read_parquet(ego_file)
ego_timestamps_arr = ego_df['timestamp'].values
ego_min_ts = ego_timestamps_arr.min()
ego_max_ts = ego_timestamps_arr.max()

print(f'  Egomotion范围: [{ego_min_ts}, {ego_max_ts}] ({(ego_max_ts-ego_min_ts)/1e6:.1f}s)')

# 计算有效t0范围 (需要完整的16步历史 + 64步未来)
history_duration = (HISTORY_STEPS - 1) * TIME_STEP * 1_000_000  # 1.5s
future_duration = FUTURE_STEPS * TIME_STEP * 1_000_000  # 6.4s

valid_t0_min = max(ego_min_ts + history_duration, 
                   max(cam_timestamps[cam]['min_ts'] for cam in cameras) + int(0.3 * 1_000_000))
valid_t0_max = min(ego_max_ts - future_duration,
                   min(cam_timestamps[cam]['max_ts'] for cam in cameras))

print(f'\n  有效t0范围: [{valid_t0_min}, {valid_t0_max}]')
print(f'  时长: {(valid_t0_max - valid_t0_min)/1e6:.1f}s')

# ============================================
# Part 3: 生成高质量索引
# ============================================
print('\nPart 3: 生成高质量索引...\n')

# 加载原始索引
index_df = pd.read_csv(f'{DATA_DIR}/infer_data_index.csv')
print(f'1. 原始索引: {len(index_df)} 行')

# 加载egomotion数据
ego_xyz = ego_df[['x', 'y', 'z']].values
ego_quat = ego_df[['qx', 'qy', 'qz', 'qw']].values

def find_closest_frame(cam_name, target_ts):
    """找最接近目标时间戳的帧号和时间差"""
    cam_ts = cam_timestamps[cam_name]['timestamps']
    cam_frames = cam_timestamps[cam_name]['frame_indices']
    
    idx = np.argmin(np.abs(cam_ts - target_ts))
    frame_num = cam_frames[idx]
    actual_ts = cam_ts[idx]
    diff_ms = (actual_ts - target_ts) / 1000  # 转为ms
    
    return frame_num, actual_ts, diff_ms

def is_valid_frame(row):
    """检查帧是否满足高质量条件"""
    t0_us = int(row['ego_timestamp'])
    
    # 1. 检查t0在有效范围内
    if t0_us < valid_t0_min or t0_us > valid_t0_max:
        return False, 't0 out of valid range'
    
    # 2. 检查4个相机的图像帧时间差
    for cam_name in cameras:
        for offset_sec in IMAGE_TIME_OFFSETS:
            target_ts = t0_us - int(offset_sec * 1_000_000)
            _, _, diff_ms = find_closest_frame(cam_name, target_ts)
            
            if abs(diff_ms) > MAX_TIME_DIFF_MS:
                return False, f'{cam_name} time diff {diff_ms:.1f}ms > {MAX_TIME_DIFF_MS}ms'
    
    return True, 'ok'

# 过滤高质量帧
print('2. 过滤高质量帧...')
valid_frames = []
invalid_reasons = {}

for idx in range(len(index_df)):
    row = index_df.iloc[idx]
    is_valid, reason = is_valid_frame(row)
    
    if is_valid:
        valid_frames.append(idx)
    else:
        invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1

print(f'   高质量帧: {len(valid_frames)}/{len(index_df)}')
print(f'   过滤原因统计:')
for reason, count in sorted(invalid_reasons.items(), key=lambda x: -x[1])[:5]:
    print(f'     - {reason}: {count}')

valid_df = index_df.iloc[valid_frames].copy()

# ============================================
# Part 4: 生成高质量推理数据
# ============================================
print(f'\n4. 处理 {len(valid_df)} 个高质量帧...')

inference_records = []

for idx in range(len(valid_df)):
    row = valid_df.iloc[idx]
    infer_idx = int(row['infer_idx'])
    t0_us = int(row['ego_timestamp'])
    
    # A. 找4个时间点的图像帧
    image_paths = {}
    all_diffs = []
    
    for cam_name in cameras:
        cam_images = []
        
        for offset_sec in IMAGE_TIME_OFFSETS:
            target_ts = t0_us - int(offset_sec * 1_000_000)
            frame_num, actual_ts, diff_ms = find_closest_frame(cam_name, target_ts)
            
            img_path = f'camera_images/{cam_name}/{frame_num:06d}.jpg'
            cam_images.append(img_path)
            all_diffs.append(abs(diff_ms))
        
        image_paths[cam_name] = cam_images
    
    max_diff = max(all_diffs)
    
    # B. 提取 egomotion
    history_offsets_us = np.arange(
        -(HISTORY_STEPS - 1) * TIME_STEP * 1_000_000,
        TIME_STEP * 1_000_000 / 2,
        TIME_STEP * 1_000_000,
    ).astype(np.int64)
    history_timestamps = t0_us + history_offsets_us
    
    future_offsets_us = np.arange(
        TIME_STEP * 1_000_000,
        (FUTURE_STEPS + 0.5) * TIME_STEP * 1_000_000,
        TIME_STEP * 1_000_000,
    ).astype(np.int64)
    future_timestamps = t0_us + future_offsets_us
    
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
    
    # C. 坐标转换
    t0_xyz = hist_xyz[-1].copy()
    t0_quat = hist_quat[-1].copy()
    t0_rot = R.from_quat(t0_quat)
    t0_rot_inv = t0_rot.inv()
    
    hist_xyz_local = t0_rot_inv.apply(hist_xyz - t0_xyz)
    hist_rot_local = (t0_rot_inv * R.from_quat(hist_quat)).as_matrix()
    
    fut_xyz_local = t0_rot_inv.apply(fut_xyz - t0_xyz)
    fut_rot_local = (t0_rot_inv * R.from_quat(fut_quat)).as_matrix()
    
    # D. 保存数据
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
    
    # E. 记录
    record = {
        'infer_idx': infer_idx,
        't0_timestamp': t0_us,
        'max_image_diff_ms': round(max_diff, 1),
        'ego_file_prefix': ego_filename,
        'cam_left_f0': image_paths['camera_cross_left_120fov'][0],
        'cam_left_f1': image_paths['camera_cross_left_120fov'][1],
        'cam_left_f2': image_paths['camera_cross_left_120fov'][2],
        'cam_left_f3': image_paths['camera_cross_left_120fov'][3],
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
    
    if (idx + 1) % 50 == 0 or idx == len(valid_df) - 1:
        print(f'  已处理 {idx + 1}/{len(valid_df)}')

# 保存索引
index_table = pd.DataFrame(inference_records)
csv_path = f'{DATA_DIR}/inference_index_high_quality.csv'
index_table.to_csv(csv_path, index=False)

print(f'\n=== 完成！===')
print(f'高质量索引: {csv_path}')
print(f'总帧数: {len(index_table)}')
print(f'过滤比例: {(len(index_df) - len(index_table)) / len(index_df) * 100:.1f}%')

# 统计时间差
print(f'\n图像帧时间差统计:')
print(f'  max_image_diff_ms: min={index_table["max_image_diff_ms"].min():.1f}ms, max={index_table["max_image_diff_ms"].max():.1f}ms, mean={index_table["max_image_diff_ms"].mean():.1f}ms')
