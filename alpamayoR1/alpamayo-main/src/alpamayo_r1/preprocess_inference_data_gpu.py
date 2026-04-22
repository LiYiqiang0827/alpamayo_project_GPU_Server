#!/usr/bin/env python3
"""
Step 2 GPU加速版: 使用 NVIDIA GPU 硬件解码 + JPG 格式
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
    ('camera_cross_left_120fov', 0),
    ('camera_front_wide_120fov', 1),
    ('camera_cross_right_120fov', 2),
    ('camera_front_tele_30fov', 6)
]

NUM_FRAMES_PER_CAM = 4
HISTORY_STEPS = 16
FUTURE_STEPS = 64
TIME_STEP = 0.1
JPG_QUALITY = 95  # JPG 质量

print('=== Step 2 GPU加速版: 预处理推理数据 ===\n')
print(f'设备: NVIDIA GPU 硬件解码 (h264_cuvid)')
print(f'图片格式: JPG (质量{JPG_QUALITY}，节省75%空间)\n')

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
    
    # 清理旧文件（如果有）
    for f in os.listdir(cam_img_dir):
        if f.endswith('.png'):
            os.remove(f'{cam_img_dir}/{f}')
    
    video_path = f'{BASE_DIR}/camera/{CLIP_ID}.{cam_name}.mp4'
    print(f'  开始 GPU 解码 {cam_name}...')
    start_time = time.time()
    
    try:
        # 使用 GPU 硬件解码器
        container = av.open(video_path)
        
        # 尝试使用 GPU 解码器
        stream = container.streams.video[0]
        
        # 检查是否支持硬件解码
        codec_name = stream.codec_context.codec.name
        if codec_name == 'h264':
            # 尝试使用 h264_cuvid (NVIDIA GPU 解码)
            try:
                stream.codec_context.codec = av.Codec('h264_cuvid', 'r')
                print(f'    使用 GPU 硬件解码: h264_cuvid')
            except Exception as e:
                print(f'    GPU 解码不可用，使用 CPU: {e}')
        
        frame_count = 0
        for frame_idx, frame in enumerate(container.decode(video=0)):
            # 转换为RGB并保存为JPG
            img = frame.to_ndarray(format='rgb24')
            img_pil = Image.fromarray(img)
            img_path = f'{cam_img_dir}/{frame_idx:06d}.jpg'
            img_pil.save(img_path, 'JPEG', quality=JPG_QUALITY)
            frame_count += 1
            
            if (frame_idx + 1) % 100 == 0:
                elapsed = time.time() - start_time
                fps = (frame_idx + 1) / elapsed
                print(f'    已解码 {frame_idx + 1} 帧, {fps:.1f} fps')
        
        container.close()
        elapsed = time.time() - start_time
        print(f'  {cam_name}: 完成 {frame_count} 帧, 耗时 {elapsed:.1f}s, 平均 {frame_count/elapsed:.1f} fps')
        return cam_name, frame_count
        
    except Exception as e:
        print(f'  {cam_name}: 解码失败 - {e}')
        return cam_name, 0

# ============================================
# Part 1: GPU 并行解码所有视频
# ============================================
print('Part 1: GPU 并行解码视频...\n')

start_time = time.time()

# 使用线程池并行解码4个相机
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(decode_video_gpu, cam_name): cam_name 
               for cam_name, _ in cameras}
    
    for future in as_completed(futures):
        cam_name, count = future.result()

total_elapsed = time.time() - start_time
print(f'\nPart 1 完成！总耗时 {total_elapsed:.1f}s\n')

# 检查所有相机是否完成
for cam_name, _ in cameras:
    cam_img_dir = f'{CAMERA_IMG_DIR}/{cam_name}'
    jpg_count = len([f for f in os.listdir(cam_img_dir) if f.endswith('.jpg')])
    print(f'  {cam_name}: {jpg_count} 张 JPG')

# ============================================
# Part 2: 生成 Egomotion 数据和索引表
# ============================================
print('\nPart 2: 生成 Egomotion 数据和索引表...\n')

# 1. 读取索引文件
index_df = pd.read_csv(f'{DATA_DIR}/infer_data_index.csv')
print(f'1. 加载索引文件: {len(index_df)} 行')

# 2. 筛选有效行
valid_mask = (
    (index_df['camera_cross_left_120fov_frame'] >= 3) &
    (index_df['camera_front_wide_120fov_frame'] >= 3) &
    (index_df['camera_cross_right_120fov_frame'] >= 3) &
    (index_df['camera_front_tele_30fov_frame'] >= 3)
)
valid_df = index_df[valid_mask].copy()
print(f'2. 有效推理帧: {len(valid_df)}')

# 3. 加载egomotion
ego_file = f'{BASE_DIR}/labels/egomotion/{CLIP_ID}.egomotion.parquet'
ego_df = pd.read_parquet(ego_file)
ego_timestamps = ego_df['timestamp'].values
ego_xyz = ego_df[['x', 'y', 'z']].values
ego_quat = ego_df[['qx', 'qy', 'qz', 'qw']].values
print(f'3. 加载 egomotion: {len(ego_df)} 点')

# 4. 生成推理数据
print(f'\n4. 处理 {len(valid_df)} 个推理帧...')

inference_records = []

def process_inference_frame(args):
    """处理单个推理帧"""
    idx, row = args
    infer_idx = int(row['infer_idx'])
    t0_us = int(row['ego_timestamp'])
    
    # 图片路径 (JPG格式)
    image_paths = {}
    for cam_name, cam_idx in cameras:
        current_frame = int(row[f'{cam_name}_frame'])
        cam_images = []
        for offset in range(NUM_FRAMES_PER_CAM):
            frame_num = current_frame - offset
            img_path = f'camera_images/{cam_name}/{frame_num:06d}.jpg'
            cam_images.append(img_path)
        image_paths[cam_name] = cam_images
    
    # 插值 egomotion
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
            xyz[:, i] = np.interp(target_ts, ego_timestamps, ego_xyz[:, i])
        for i in range(4):
            quat[:, i] = np.interp(target_ts, ego_timestamps, ego_quat[:, i])
        norms = np.linalg.norm(quat, axis=1, keepdims=True)
        quat = quat / norms
        return xyz, quat
    
    hist_xyz, hist_quat = interpolate_ego(history_timestamps)
    fut_xyz, fut_quat = interpolate_ego(future_timestamps)
    
    # 坐标转换
    t0_xyz = hist_xyz[-1].copy()
    t0_quat = hist_quat[-1].copy()
    t0_rot = R.from_quat(t0_quat)
    t0_rot_inv = t0_rot.inv()
    
    hist_xyz_local = t0_rot_inv.apply(hist_xyz - t0_xyz)
    hist_rot_local = (t0_rot_inv * R.from_quat(hist_quat)).as_matrix()
    fut_xyz_local = t0_rot_inv.apply(fut_xyz - t0_xyz)
    fut_rot_local = (t0_rot_inv * R.from_quat(fut_quat)).as_matrix()
    
    # 保存 Egomotion
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
        't0_timestamp': t0_us,
        't0_position_world': t0_xyz.tolist(),
        't0_quaternion_world': t0_quat.tolist(),
        't0_rotation_matrix': t0_rot.as_matrix().tolist()
    }
    with open(f'{EGO_DIR}/{ego_filename}_t0.json', 'w') as f:
        json.dump(t0_info, f, indent=2)
    
    # 返回记录
    return {
        'infer_idx': infer_idx,
        't0_timestamp': t0_us,
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

# 并行处理推理帧
from tqdm import tqdm

print(f'使用多线程处理 {len(valid_df)} 个推理帧...')
with ThreadPoolExecutor(max_workers=8) as executor:
    results = list(tqdm(
        executor.map(process_inference_frame, valid_df.iterrows()),
        total=len(valid_df),
        desc='处理推理帧'
    ))

inference_records = results

# 保存索引表
index_table = pd.DataFrame(inference_records)
csv_path = f'{DATA_DIR}/inference_index.csv'
index_table.to_csv(csv_path, index=False)

print(f'\n5. 索引表已保存: {csv_path}')
print(f'   总行数: {len(index_table)}')

# 6. 统计信息
print(f'\n=== 全部完成！===')
print(f'数据位置: {DATA_DIR}/')
print(f'总推理帧数: {len(valid_df)}')

# 检查图片总大小
import subprocess
result = subprocess.run(['du', '-sh', CAMERA_IMG_DIR], capture_output=True, text=True)
print(f'图片总大小: {result.stdout.strip()}')

print(f'\n预期节省空间: ~75% (JPG vs PNG)')
