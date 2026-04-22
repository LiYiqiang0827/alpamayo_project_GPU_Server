#!/usr/bin/env python3
"""
Step 2: 预处理推理数据
从mp4提取图像帧，提取并转换egomotion数据
"""
import pandas as pd
import numpy as np
import os
import av
from pathlib import Path
import json
from scipy.spatial.transform import Rotation as R

CLIP_ID = '01d3588e-bca7-4a18-8e74-c6cfe9e996db'
BASE_DIR = '/data01/vla/data/data_sample_chunk0'
INFER_DIR = f'{BASE_DIR}/infer/{CLIP_ID}'
DATA_DIR = f'{INFER_DIR}/data'

# 相机配置
cameras = [
    ('camera_cross_left_120fov', 0),
    ('camera_front_wide_120fov', 1),
    ('camera_cross_right_120fov', 2),
    ('camera_front_tele_30fov', 6)
]

NUM_FRAMES_PER_CAM = 4  # 每个相机取4帧
HISTORY_STEPS = 16      # 1.6秒历史
FUTURE_STEPS = 64       # 6.4秒未来
TIME_STEP = 0.1         # 10Hz

print('=== Step 2: 预处理推理数据 ===\n')

# 1. 读取索引文件
index_df = pd.read_csv(f'{DATA_DIR}/infer_data_index.csv')
print(f'1. 加载索引文件: {len(index_df)} 行')

# 2. 筛选所有相机帧号 >= 3 的行 (确保能取到前3帧)
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

# 4. 加载视频文件 (使用av)
video_containers = {}
for cam_name, _ in cameras:
    video_path = f'{BASE_DIR}/camera/{CLIP_ID}.{cam_name}.mp4'
    container = av.open(video_path)
    video_containers[cam_name] = container
    stream = container.streams.video[0]
    total_frames = stream.frames
    print(f'4. 加载 {cam_name}: {total_frames} 帧')

# 5. 处理每个有效推理点
print(f'\n5. 开始预处理 {len(valid_df)} 个推理点...')

for idx, row in valid_df.iterrows():
    infer_idx = int(row['infer_idx'])
    t0_us = int(row['ego_timestamp'])
    
    # 创建 Frame 文件夹
    frame_dir = f'{DATA_DIR}/Frame_{infer_idx:06d}'
    os.makedirs(frame_dir, exist_ok=True)
    
    # A. 提取图像帧 (4相机 × 4帧)
    for cam_name, cam_idx in cameras:
        current_frame = int(row[f'{cam_name}_frame'])
        
        # 提取当前帧和往前3帧
        for offset in range(NUM_FRAMES_PER_CAM):
            frame_num = current_frame - offset
            
            container = video_containers[cam_name]
            
            # 使用av读取指定帧
            frame_idx = 0
            for frame in container.decode(video=0):
                if frame_idx == frame_num:
                    # 转换为numpy数组 (RGB)
                    img = frame.to_ndarray(format='rgb24')
                    # 保存为PNG
                    img_path = f'{frame_dir}/{cam_name}_f{offset}.png'
                    # 使用imageio或PIL保存，这里用numpy直接保存为npy先
                    np.save(f'{frame_dir}/{cam_name}_f{offset}.npy', img)
                    break
                frame_idx += 1
            
            # 重置容器指针 (重新打开)
            container = av.open(f'{BASE_DIR}/camera/{CLIP_ID}.{cam_name}.mp4')
            video_containers[cam_name] = container
    
    # B. 提取 egomotion 数据
    # 计算历史和未来时间戳
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
    
    t0_info = {
        't0_timestamp': t0_us,
        't0_position_world': t0_xyz.tolist(),
        't0_quaternion_world': t0_quat.tolist(),
        't0_rotation_matrix': t0_rot.as_matrix().tolist()
    }
    with open(f'{frame_dir}/t0_info.json', 'w') as f:
        json.dump(t0_info, f, indent=2)
    
    if (idx + 1) % 10 == 0 or idx == len(valid_df) - 1:
        print(f'  已处理 {idx + 1}/{len(valid_df)}: Frame_{infer_idx:06d}')

# 6. 释放视频资源
for container in video_containers.values():
    container.close()

print(f'\n6. 预处理完成！共生成 {len(valid_df)} 个 Frame 文件夹')

# 7. 生成数据说明文件
readme_content = f"""# 推理数据组织说明

## 目录结构
Frame_{{infer_idx:06d}}/
├── {{camera_name}}_f0.npy       # 当前帧图像 (H, W, 3) RGB
├── {{camera_name}}_f1.npy       # 前1帧图像
├── {{camera_name}}_f2.npy       # 前2帧图像
├── {{camera_name}}_f3.npy       # 前3帧图像
├── ego_history_world.npy        # 历史轨迹（世界坐标系）
├── ego_history_local.npy        # 历史轨迹（t0局部坐标系）
├── ego_future_gt.npy            # 未来轨迹真值（t0局部坐标系）
└── t0_info.json                 # t0时刻信息

## 数据说明

### 图像帧 (NPY)
- 4个相机: camera_cross_left_120fov, camera_front_wide_120fov, camera_cross_right_120fov, camera_front_tele_30fov
- 每个相机4帧: f0(当前), f1(前1帧), f2(前2帧), f3(前3帧)
- 数组形状: (H, W, 3), dtype=uint8, RGB格式

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
- 总帧数: {len(valid_df)}
- 生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

with open(f'{DATA_DIR}/README.md', 'w') as f:
    f.write(readme_content)

print(f'7. 数据说明文件已生成: {DATA_DIR}/README.md')
