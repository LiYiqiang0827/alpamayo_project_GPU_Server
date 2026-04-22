#!/usr/bin/env python3
"""
Step 2 改进版: 先解码所有视频帧，再生成推理数据
"""
import pandas as pd
import numpy as np
import os
import av
import shutil
from pathlib import Path
import json
from scipy.spatial.transform import Rotation as R
from PIL import Image

CLIP_ID = '01d3588e-bca7-4a18-8e74-c6cfe9e996db'
BASE_DIR = '/data01/vla/data/data_sample_chunk0'
INFER_DIR = f'{BASE_DIR}/infer/{CLIP_ID}'
DATA_DIR = f'{INFER_DIR}/data'
CAMERA_IMG_DIR = f'{INFER_DIR}/camera_images'

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

print('=== Step 2 (改进版): 预处理推理数据 ===\n')

# ============================================
# Part 1: 解码所有视频帧到图片
# ============================================
print('Part 1: 解码视频帧到图片...\n')

for cam_name, cam_idx in cameras:
    # 创建相机图片文件夹
    cam_img_dir = f'{CAMERA_IMG_DIR}/{cam_name}'
    os.makedirs(cam_img_dir, exist_ok=True)
    
    # 检查是否已解码过
    existing_files = len([f for f in os.listdir(cam_img_dir) if f.endswith('.png')])
    if existing_files > 0:
        print(f'  {cam_name}: 已存在 {existing_files} 张图片，跳过解码')
        continue
    
    video_path = f'{BASE_DIR}/camera/{CLIP_ID}.{cam_name}.mp4'
    print(f'  解码 {cam_name}...')
    
    container = av.open(video_path)
    stream = container.streams.video[0]
    total_frames = stream.frames
    
    frame_count = 0
    for frame_idx, frame in enumerate(container.decode(video=0)):
        # 转换为RGB并保存
        img = frame.to_ndarray(format='rgb24')
        img_pil = Image.fromarray(img)
        img_path = f'{cam_img_dir}/{frame_idx:06d}.png'
        img_pil.save(img_path)
        frame_count += 1
        
        if (frame_idx + 1) % 100 == 0:
            print(f'    已解码 {frame_idx + 1}/{total_frames} 帧')
    
    container.close()
    print(f'  {cam_name}: 完成，共 {frame_count} 帧')

print('\nPart 1 完成！\n')

# ============================================
# Part 2: 生成 Frame 文件夹
# ============================================
print('Part 2: 生成 Frame 推理数据...\n')

# 1. 读取索引文件
index_df = pd.read_csv(f'{DATA_DIR}/infer_data_index.csv')
print(f'1. 加载索引文件: {len(index_df)} 行')

# 2. 筛选所有相机帧号 >= 3 的行
valid_mask = (
    (index_df['camera_cross_left_120fov_frame'] >= 3) &
    (index_df['camera_front_wide_120fov_frame'] >= 3) &
    (index_df['camera_cross_right_120fov_frame'] >= 3) &
    (index_df['camera_front_tele_30fov_frame'] >= 3)
)
valid_df = index_df[valid_mask].copy()
print(f'2. 帧号 >= 3 的有效行: {len(valid_df)} (原 {len(index_df)})')

# 3. 加载egomotion数据
ego_file = f'{BASE_DIR}/labels/egomotion/{CLIP_ID}.egomotion.parquet'
ego_df = pd.read_parquet(ego_file)
ego_timestamps = ego_df['timestamp'].values
ego_xyz = ego_df[['x', 'y', 'z']].values
ego_quat = ego_df[['qx', 'qy', 'qz', 'qw']].values
print(f'3. 加载 egomotion: {len(ego_df)} 点')

# 4. 处理每个有效推理点
print(f'\n4. 开始生成 {len(valid_df)} 个 Frame 文件夹...')

for idx, row in valid_df.iterrows():
    infer_idx = int(row['infer_idx'])
    t0_us = int(row['ego_timestamp'])
    
    # 创建 Frame 文件夹
    frame_dir = f'{DATA_DIR}/Frame_{infer_idx:06d}'
    os.makedirs(frame_dir, exist_ok=True)
    
    # A. 复制/链接图像帧 (4相机 × 4帧)
    # 记录使用的图片路径
    image_refs = {}
    
    for cam_name, cam_idx in cameras:
        current_frame = int(row[f'{cam_name}_frame'])
        cam_img_dir = f'{CAMERA_IMG_DIR}/{cam_name}'
        
        cam_refs = []
        for offset in range(NUM_FRAMES_PER_CAM):
            frame_num = current_frame - offset
            src_path = f'{cam_img_dir}/{frame_num:06d}.png'
            dst_path = f'{frame_dir}/{cam_name}_f{offset}.png'
            
            # 使用软链接节省空间
            if os.path.exists(src_path):
                os.symlink(os.path.abspath(src_path), dst_path)
                cam_refs.append(f'{cam_name}/{frame_num:06d}.png')
            else:
                print(f'警告: 图片不存在 {src_path}')
                cam_refs.append(None)
        
        image_refs[cam_name] = cam_refs
    
    # B. 提取 egomotion 数据
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
    
    # 插值 egomotion
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
    
    # C. 坐标转换到 t0 坐标系
    t0_xyz = hist_xyz[-1].copy()
    t0_quat = hist_quat[-1].copy()
    t0_rot = R.from_quat(t0_quat)
    t0_rot_inv = t0_rot.inv()
    
    hist_xyz_local = t0_rot_inv.apply(hist_xyz - t0_xyz)
    hist_rot_local = (t0_rot_inv * R.from_quat(hist_quat)).as_matrix()
    
    fut_xyz_local = t0_rot_inv.apply(fut_xyz - t0_xyz)
    fut_rot_local = (t0_rot_inv * R.from_quat(fut_quat)).as_matrix()
    
    # D. 保存数据
    np.save(f'{frame_dir}/ego_history_world.npy', {
        'xyz': hist_xyz,
        'quat': hist_quat,
        'timestamps': history_timestamps
    })
    
    np.save(f'{frame_dir}/ego_history_local.npy', {
        'xyz': hist_xyz_local,
        'rotation_matrix': hist_rot_local,
        'timestamps': history_timestamps
    })
    
    np.save(f'{frame_dir}/ego_future_gt.npy', {
        'xyz': fut_xyz_local,
        'rotation_matrix': fut_rot_local,
        'timestamps': future_timestamps
    })
    
    # 保存图像引用信息
    with open(f'{frame_dir}/image_refs.json', 'w') as f:
        json.dump({
            't0_timestamp': t0_us,
            'camera_images': image_refs
        }, f, indent=2)
    
    # t0 信息
    t0_info = {
        't0_timestamp': t0_us,
        't0_position_world': t0_xyz.tolist(),
        't0_quaternion_world': t0_quat.tolist(),
        't0_rotation_matrix': t0_rot.as_matrix().tolist()
    }
    with open(f'{frame_dir}/t0_info.json', 'w') as f:
        json.dump(t0_info, f, indent=2)
    
    if (idx + 1) % 100 == 0 or idx == len(valid_df) - 1:
        print(f'  已处理 {idx + 1}/{len(valid_df)}: Frame_{infer_idx:06d}')

print(f'\n5. 预处理完成！共生成 {len(valid_df)} 个 Frame 文件夹')

# 5. 生成数据说明文件
readme_content = f"""# 推理数据组织说明

## 目录结构

### 解码后的相机图片
```
{INFER_DIR}/camera_images/
├── camera_cross_left_120fov/
│   ├── 000000.png
│   ├── 000001.png
│   └── ... (605 帧)
├── camera_front_wide_120fov/
│   └── ...
├── camera_cross_right_120fov/
│   └── ...
└── camera_front_tele_30fov/
    └── ...
```

### 推理数据帧
```
{DATA_DIR}/Frame_{{infer_idx:06d}}/
├── camera_cross_left_120fov_f0.png      # 软链接到 camera_images/
├── camera_cross_left_120fov_f1.png
├── camera_cross_left_120fov_f2.png
├── camera_cross_left_120fov_f3.png
├── camera_front_wide_120fov_f0.png
├── ... (共16张图片软链接)
├── image_refs.json                      # 图片引用信息
├── ego_history_world.npy                # 历史轨迹（世界坐标系）
├── ego_history_local.npy                # 历史轨迹（t0局部坐标系）
├── ego_future_gt.npy                    # 未来轨迹真值（t0局部坐标系）
└── t0_info.json                         # t0时刻信息
```

## 数据说明

### 图像帧
- 4个相机: camera_cross_left_120fov, camera_front_wide_120fov, camera_cross_right_120fov, camera_front_tele_30fov
- 每个相机4帧: f0(当前), f1(前1帧), f2(前2帧), f3(前3帧)
- 格式: PNG, RGB
- Frame文件夹中使用**软链接**指向 camera_images/，节省空间

### Egomotion 数据

**ego_history_world.npy**
- xyz: (16, 3) - 世界坐标系位置 [米]
- quat: (16, 4) - 世界坐标系四元数 [qx, qy, qz, qw]
- timestamps: (16,) - 时间戳 [微秒]
- 范围: t0-1.5s 到 t0

**ego_history_local.npy**
- xyz: (16, 3) - t0局部坐标系位置 [米]
- rotation_matrix: (16, 3, 3) - t0局部坐标系旋转矩阵
- timestamps: (16,) - 时间戳 [微秒]

**ego_future_gt.npy**
- xyz: (64, 3) - t0局部坐标系未来位置真值 [米]
- rotation_matrix: (64, 3, 3) - t0局部坐标系未来旋转真值
- timestamps: (64,) - 时间戳 [微秒]
- 范围: t0+0.1s 到 t0+6.4s

**image_refs.json**
- t0_timestamp: t0时间戳
- camera_images: 各相机使用的图片路径

**t0_info.json**
- t0_timestamp: t0时间戳 [微秒]
- t0_position_world: t0世界坐标位置 [米]
- t0_quaternion_world: t0世界坐标四元数
- t0_rotation_matrix: t0世界坐标旋转矩阵

## 坐标转换
世界坐标系 → t0局部坐标系:
```
xyz_local = R_t0⁻¹ @ (xyz_world - t0_xyz)
rot_local = R_t0⁻¹ @ R_world
```

## 统计信息
- 总Frame数: {len(valid_df)}
- 相机图片总数: ~2420 张 (4相机 × 605帧)
- 生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

with open(f'{DATA_DIR}/README.md', 'w') as f:
    f.write(readme_content)

print(f'6. 数据说明文件已生成: {DATA_DIR}/README.md')
