#!/usr/bin/env python3
"""
验证脚本：检查预处理数据完整性
验证 inference_index.csv 中的16张图片和3个egomotion文件
"""
import pandas as pd
import numpy as np
import os
from PIL import Image
from pathlib import Path

def verify_inference_data(clip_id='01d3588e-bca7-4a18-8e74-c6cfe9e996db'):
    """验证推理数据完整性"""
    
    BASE_DIR = '/data01/vla/data/data_sample_chunk0'
    INFER_DIR = f'{BASE_DIR}/infer/{clip_id}'
    DATA_DIR = f'{INFER_DIR}/data'
    
    print('=== 推理数据完整性验证 ===\n')
    
    # 1. 检查核心文件是否存在
    index_csv = f'{DATA_DIR}/inference_index.csv'
    if not os.path.exists(index_csv):
        print(f'❌ 错误: 索引文件不存在: {index_csv}')
        print('   请等待预处理脚本完成')
        return False
    
    print(f'✅ 索引文件存在: {index_csv}')
    
    # 2. 读取索引表
    df = pd.read_csv(index_csv)
    print(f'✅ 索引表行数: {len(df)}')
    
    # 3. 检查图片列
    image_cols = [col for col in df.columns if col.startswith('cam_')]
    print(f'✅ 图片路径列数: {len(image_cols)} (期望16列)')
    
    # 4. 验证图片存在性 (随机抽样)
    sample_size = min(10, len(df))
    sample_indices = np.random.choice(len(df), sample_size, replace=False)
    
    print(f'\n验证图片存在性 (随机抽样 {sample_size} 帧)...')
    missing_images = []
    for idx in sample_indices:
        row = df.iloc[idx]
        for col in image_cols:
            img_path = f'{DATA_DIR}/{row[col]}'
            if not os.path.exists(img_path):
                missing_images.append((idx, col, img_path))
    
    if missing_images:
        print(f'❌ 发现 {len(missing_images)} 个缺失图片:')
        for idx, col, path in missing_images[:5]:
            print(f'   行{idx}, 列{col}: {path}')
        return False
    else:
        print(f'✅ 抽样 {sample_size * 16} 张图片全部存在')
    
    # 5. 验证图片可读性 (进一步抽样)
    print(f'\n验证图片可读性...')
    check_count = min(5, len(df))
    for i in range(check_count):
        row = df.iloc[i]
        for col in ['cam_left_f0', 'cam_front_f0', 'cam_right_f0', 'cam_tele_f0']:
            img_path = f'{DATA_DIR}/{row[col]}'
            try:
                img = Image.open(img_path)
                img.verify()  # 验证图片完整性
            except Exception as e:
                print(f'❌ 图片损坏: {img_path}, 错误: {e}')
                return False
    print(f'✅ 抽样 {check_count * 4} 张图片可读性正常')
    
    # 6. 验证 egomotion 文件
    print(f'\n验证 Egomotion 文件...')
    ego_dir = f'{DATA_DIR}/egomotion'
    
    sample_indices = np.random.choice(len(df), sample_size, replace=False)
    missing_ego = []
    invalid_ego = []
    
    for idx in sample_indices:
        row = df.iloc[idx]
        prefix = row['ego_file_prefix']
        
        # 检查3个npy文件
        ego_files = [
            f'{ego_dir}/{prefix}_history_world.npy',
            f'{ego_dir}/{prefix}_history_local.npy',
            f'{ego_dir}/{prefix}_future_gt.npy'
        ]
        
        for ego_file in ego_files:
            if not os.path.exists(ego_file):
                missing_ego.append((idx, ego_file))
            else:
                # 验证npy文件可读
                try:
                    data = np.load(ego_file, allow_pickle=True).item()
                    # 检查关键字段
                    if 'xyz' not in data or 'timestamps' not in data:
                        invalid_ego.append((idx, ego_file, '缺少关键字段'))
                except Exception as e:
                    invalid_ego.append((idx, ego_file, str(e)))
        
        # 检查json文件
        json_file = f'{ego_dir}/{prefix}_t0.json'
        if not os.path.exists(json_file):
            missing_ego.append((idx, json_file))
    
    if missing_ego:
        print(f'❌ 发现 {len(missing_ego)} 个缺失egomotion文件:')
        for idx, path in missing_ego[:5]:
            print(f'   行{idx}: {path}')
        return False
    
    if invalid_ego:
        print(f'❌ 发现 {len(invalid_ego)} 个无效egomotion文件:')
        for idx, path, err in invalid_ego[:5]:
            print(f'   行{idx}: {path}, 错误: {err}')
        return False
    
    print(f'✅ 抽样 {sample_size} 个推理帧的egomotion文件全部正常')
    
    # 7. 验证数据内容
    print(f'\n验证数据内容格式...')
    test_idx = 0
    row = df.iloc[test_idx]
    prefix = row['ego_file_prefix']
    
    # 验证历史数据
    hist_world = np.load(f'{ego_dir}/{prefix}_history_world.npy', allow_pickle=True).item()
    hist_local = np.load(f'{ego_dir}/{prefix}_history_local.npy', allow_pickle=True).item()
    
    print(f'  历史数据 (世界坐标):')
    print(f'    xyz shape: {hist_world["xyz"].shape} (期望 (16, 3))')
    print(f'    quat shape: {hist_world["quat"].shape} (期望 (16, 4))')
    print(f'    timestamps shape: {hist_world["timestamps"].shape} (期望 (16,))')
    
    print(f'  历史数据 (局部坐标):')
    print(f'    xyz shape: {hist_local["xyz"].shape} (期望 (16, 3))')
    print(f'    rotation_matrix shape: {hist_local["rotation_matrix"].shape} (期望 (16, 3, 3))')
    
    # 验证未来数据
    future = np.load(f'{ego_dir}/{prefix}_future_gt.npy', allow_pickle=True).item()
    print(f'  未来真值:')
    print(f'    xyz shape: {future["xyz"].shape} (期望 (64, 3))')
    print(f'    rotation_matrix shape: {future["rotation_matrix"].shape} (期望 (64, 3, 3))')
    print(f'    timestamps shape: {future["timestamps"].shape} (期望 (64,))')
    
    # 8. 统计信息
    print(f'\n=== 验证通过！===\n')
    print(f'数据目录: {DATA_DIR}')
    print(f'总推理帧数: {len(df)}')
    print(f'图片路径列: {len(image_cols)} 列 (16张图片)')
    print(f'Egomotion 文件: 4个/帧 (3个npy + 1个json)')
    print(f'预期总文件数: {len(df) * 4} 个egomotion文件')
    
    # 实际统计
    ego_files = len([f for f in os.listdir(ego_dir) if f.endswith('.npy') or f.endswith('.json')])
    print(f'实际 egomotion 文件数: {ego_files}')
    
    return True

if __name__ == '__main__':
    import sys
    clip_id = sys.argv[1] if len(sys.argv) > 1 else '01d3588e-bca7-4a18-8e74-c6cfe9e996db'
    success = verify_inference_data(clip_id)
    sys.exit(0 if success else 1)
