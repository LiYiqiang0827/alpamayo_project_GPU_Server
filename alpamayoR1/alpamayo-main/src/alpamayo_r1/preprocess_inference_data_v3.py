#!/usr/bin/env python3
"""
Step 2 最终版: 中心化数据组织
- camera_images/: 解码后的相机图片
- egomotion/: 所有推理帧的轨迹数据
- inference_index.csv: 统一索引表
"""
import pandas as pd
import numpy as np
import os
import av
from pathlib import Path
import json
from scipy.spatial.transform import Rotation as R
from PIL import Image

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

print('=== Step 2 (最终版): 预处理推理数据 ===\n')

# 创建目录
os.makedirs(CAMERA_IMG_DIR, exist_ok=True)
os.makedirs(EGO_DIR, exist_ok=True)

# ============================================
# Part 1: 解码所有视频帧到图片
# ============================================
print('Part 1: 解码视频帧到图片...\n')

for cam_name, cam_idx in cameras:
    cam_img_dir = f'{CAMERA_IMG_DIR}/{cam_name}'
    os.makedirs(cam_img_dir, exist_ok=True)
    
    existing_files = len([f for f in os.listdir(cam_img_dir) if f.endswith('.png')])
    if existing_files >= 600:  # 假设605帧
        print(f'  {cam_name}: 已存在 {existing_files} 张图片，跳过')
        continue
    
    video_path = f'{BASE_DIR}/camera/{CLIP_ID}.{cam_name}.mp4'
    print(f'  解码 {cam_name}...')
    
    container = av.open(video_path)
    stream = container.streams.video[0]
    total_frames = stream.frames
    
    frame_count = 0
    for frame_idx, frame in enumerate(container.decode(video=0)):
        img = frame.to_ndarray(format='rgb24')
        img_pil = Image.fromarray(img)
        img_path = f'{cam_img_dir}/{frame_idx:06d}.png'
        img_pil.save(img_path)
        frame_count += 1
        
        if (frame_idx + 1) % 100 == 0:
            print(f'    已解码 {frame_idx + 1}/{total_frames}')
    
    container.close()
    print(f'  {cam_name}: 完成，共 {frame_count} 帧')

print('\nPart 1 完成！\n')

# ============================================
# Part 2: 生成 Egomotion 数据和索引表
# ============================================
print('Part 2: 生成 Egomotion 数据和索引表...\n')

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
print(f'2. 有效推理帧: {len(valid_df)} (原 {len(index_df)})')

# 3. 加载egomotion数据
ego_file = f'{BASE_DIR}/labels/egomotion/{CLIP_ID}.egomotion.parquet'
ego_df = pd.read_parquet(ego_file)
ego_timestamps = ego_df['timestamp'].values
ego_xyz = ego_df[['x', 'y', 'z']].values
ego_quat = ego_df[['qx', 'qy', 'qz', 'qw']].values
print(f'3. 加载 egomotion: {len(ego_df)} 点')

# 4. 生成每个推理帧的数据
print(f'\n4. 处理 {len(valid_df)} 个推理帧...')

inference_records = []

for idx, row in valid_df.iterrows():
    infer_idx = int(row['infer_idx'])
    t0_us = int(row['ego_timestamp'])
    
    # A. 记录图片路径
    image_paths = {}
    for cam_name, cam_idx in cameras:
        current_frame = int(row[f'{cam_name}_frame'])
        cam_images = []
        for offset in range(NUM_FRAMES_PER_CAM):
            frame_num = current_frame - offset
            img_path = f'camera_images/{cam_name}/{frame_num:06d}.png'
            cam_images.append(img_path)
        image_paths[cam_name] = cam_images
    
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
    
    # 插值
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
    
    # C. 坐标转换
    t0_xyz = hist_xyz[-1].copy()
    t0_quat = hist_quat[-1].copy()
    t0_rot = R.from_quat(t0_quat)
    t0_rot_inv = t0_rot.inv()
    
    hist_xyz_local = t0_rot_inv.apply(hist_xyz - t0_xyz)
    hist_rot_local = (t0_rot_inv * R.from_quat(hist_quat)).as_matrix()
    
    fut_xyz_local = t0_rot_inv.apply(fut_xyz - t0_xyz)
    fut_rot_local = (t0_rot_inv * R.from_quat(fut_quat)).as_matrix()
    
    # D. 保存 Egomotion 数据到文件
    ego_filename = f'ego_{infer_idx:06d}'
    
    # 历史世界坐标
    np.save(f'{EGO_DIR}/{ego_filename}_history_world.npy', {
        'xyz': hist_xyz,
        'quat': hist_quat,
        'timestamps': history_timestamps
    })
    
    # 历史局部坐标
    np.save(f'{EGO_DIR}/{ego_filename}_history_local.npy', {
        'xyz': hist_xyz_local,
        'rotation_matrix': hist_rot_local,
        'timestamps': history_timestamps
    })
    
    # 未来真值局部坐标
    np.save(f'{EGO_DIR}/{ego_filename}_future_gt.npy', {
        'xyz': fut_xyz_local,
        'rotation_matrix': fut_rot_local,
        'timestamps': future_timestamps
    })
    
    # t0 信息
    t0_info = {
        't0_timestamp': t0_us,
        't0_position_world': t0_xyz.tolist(),
        't0_quaternion_world': t0_quat.tolist(),
        't0_rotation_matrix': t0_rot.as_matrix().tolist()
    }
    with open(f'{EGO_DIR}/{ego_filename}_t0.json', 'w') as f:
        json.dump(t0_info, f, indent=2)
    
    # E. 记录到索引表
    record = {
        'infer_idx': infer_idx,
        't0_timestamp': t0_us,
        'ego_file_prefix': ego_filename,
        # 图片路径
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
    
    if (idx + 1) % 100 == 0 or idx == len(valid_df) - 1:
        print(f'  已处理 {idx + 1}/{len(valid_df)}: Frame_{infer_idx:06d}')

# 5. 保存索引表
index_table = pd.DataFrame(inference_records)
csv_path = f'{DATA_DIR}/inference_index.csv'
index_table.to_csv(csv_path, index=False)

print(f'\n5. 索引表已保存: {csv_path}')
print(f'   总行数: {len(index_table)}')
print(f'   列数: {len(index_table.columns)}')

# 6. 生成数据说明
readme_content = f"""# 推理数据组织说明 (中心化版本)

## 目录结构

```
{DATA_DIR}/
├── camera_images/                  # 解码后的相机图片
│   ├── camera_cross_left_120fov/
│   │   ├── 000000.png ~ 000604.png
│   ├── camera_front_wide_120fov/
│   ├── camera_cross_right_120fov/
│   └── camera_front_tele_30fov/
├── egomotion/                      # 所有推理帧的轨迹数据
│   ├── ego_000011_history_world.npy
│   ├── ego_000011_history_local.npy
│   ├── ego_000011_future_gt.npy
│   ├── ego_000011_t0.json
│   └── ... (共 {len(valid_df)} 组)
├── inference_index.csv             # 统一索引表 ⭐核心文件
├── infer_data_index.csv            # 原始时间戳索引
└── README.md                       # 本文件
```

## inference_index.csv 说明

这是**核心索引文件**，每行代表一个可推理的时间点。

### 列说明

**基础信息**
- `infer_idx`: 推理索引号 (0-N)
- `t0_timestamp`: 当前时刻时间戳 (微秒)
- `ego_file_prefix`: Egomotion 文件前缀 (如 `ego_000011`)

**图片路径** (相对于 data/ 目录)
- `cam_left_f0` ~ `cam_left_f3`: 左交叉相机 (当前, 前1, 前2, 前3帧)
- `cam_front_f0` ~ `cam_front_f3`: 前广角相机
- `cam_right_f0` ~ `cam_right_f3`: 右交叉相机
- `cam_tele_f0` ~ `cam_tele_f3`: 前长焦相机

### 使用示例

```python
import pandas as pd
import numpy as np
from PIL import Image

# 加载索引表
df = pd.read_csv('inference_index.csv')

# 获取第0个推理帧
row = df.iloc[0]

# 加载图片
img_left_f0 = Image.open(f\"data/{{row['cam_left_f0']}}\")
img_front_f0 = Image.open(f\"data/{{row['cam_front_f0']}}\")
# ...

# 加载egomotion
prefix = row['ego_file_prefix']
history = np.load(f'egomotion/{{prefix}}_history_local.npy', allow_pickle=True).item()
future = np.load(f'egomotion/{{prefix}}_future_gt.npy', allow_pickle=True).item()
```

## Egomotion 文件格式

**{{prefix}}_history_world.npy**
- xyz: (16, 3) - 世界坐标位置 [米]
- quat: (16, 4) - 世界坐标四元数
- timestamps: (16,) - 时间戳 [微秒]

**{{prefix}}_history_local.npy**
- xyz: (16, 3) - t0局部坐标位置 [米]
- rotation_matrix: (16, 3, 3) - 旋转矩阵
- timestamps: (16,) - 时间戳 [微秒]

**{{prefix}}_future_gt.npy**
- xyz: (64, 3) - t0局部坐标未来真值 [米]
- rotation_matrix: (64, 3, 3) - 旋转矩阵
- timestamps: (64,) - 时间戳 [微秒]

**{{prefix}}_t0.json**
- t0_timestamp, t0_position_world, t0_quaternion_world, t0_rotation_matrix

## 统计信息
- 总推理帧数: {len(valid_df)}
- 相机图片总数: ~2420 张
- 索引表行数: {len(index_table)}
- 生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

with open(f'{DATA_DIR}/README.md', 'w') as f:
    f.write(readme_content)

print(f'\n6. 数据说明已生成: {DATA_DIR}/README.md')
print(f'\n=== 全部完成！===')
print(f'数据位置: {DATA_DIR}/')
print(f'索引文件: {csv_path}')
