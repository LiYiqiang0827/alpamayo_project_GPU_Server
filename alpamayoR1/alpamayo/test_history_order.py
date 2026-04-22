#!/usr/bin/env python3
"""测试：交换历史数据顺序，看是否改善ADE"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import copy
import numpy as np
import pandas as pd
import torch
import sys
sys.path.insert(0, '/home/user/mikelee/alpamayo-main/src')
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper
from PIL import Image
from einops import rearrange

CLIP_ID = '46003675-4b4e-4c0f-ae54-3f7622bddf6a'
DATA_DIR = f'/data01/vla/data/data_sample_chunk0/infer/{CLIP_ID}/data'
MODEL_PATH = "/data01/vla/models--nvidia--Alpamayo-R1-10B/snapshots/22fab1399111f50b52bfbe5d8b809f39bd4c2fe1"

CAMERA_ORDER = ['camera_cross_left_120fov', 'camera_front_wide_120fov', 
                'camera_cross_right_120fov', 'camera_front_tele_30fov']

cam_cols = {
    'camera_cross_left_120fov': ['cam_left_f0', 'cam_left_f1', 'cam_left_f2', 'cam_left_f3'],
    'camera_front_wide_120fov': ['cam_front_f0', 'cam_front_f1', 'cam_front_f2', 'cam_front_f3'],
    'camera_cross_right_120fov': ['cam_right_f0', 'cam_right_f1', 'cam_right_f2', 'cam_right_f3'],
    'camera_front_tele_30fov': ['cam_tele_f0', 'cam_tele_f1', 'cam_tele_f2', 'cam_tele_f3'],
}

def load_images_for_frame(row, data_dir):
    images = []
    for cam in CAMERA_ORDER:
        for col in cam_cols[cam]:
            img_path = f'{data_dir}/{row[col]}'
            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img)
            images.append(img_array)
    images = np.stack(images, axis=0)
    images = rearrange(images, '(c t) h w ch -> c t ch h w', c=4, t=4)
    return torch.from_numpy(images).float()

def load_egomotion_for_frame(row, data_dir, reverse_history=False):
    prefix = row['ego_file_prefix']
    ego_dir = f'{data_dir}/egomotion'
    hist_local = np.load(f'{ego_dir}/{prefix}_history_local.npy', allow_pickle=True).item()
    future_gt = np.load(f'{ego_dir}/{prefix}_future_gt.npy', allow_pickle=True).item()
    
    hist_xyz_np = hist_local['xyz']
    hist_rot_np = hist_local['rotation_matrix']
    
    if reverse_history:
        # 交换历史数据顺序（从t-1.6s到t0变成从t0到t-1.6s）
        hist_xyz_np = hist_xyz_np[::-1].copy()
        hist_rot_np = hist_rot_np[::-1].copy()
    
    hist_xyz = torch.from_numpy(hist_xyz_np).float().unsqueeze(0).unsqueeze(0)
    hist_rot = torch.from_numpy(hist_rot_np).float().unsqueeze(0).unsqueeze(0)
    future_xyz = torch.from_numpy(future_gt['xyz']).float().unsqueeze(0).unsqueeze(0)
    return hist_xyz, hist_rot, future_xyz

print('=== 测试历史数据顺序对ADE的影响 ===\n')

# 加载高质量索引
hq_df = pd.read_csv(f'{DATA_DIR}/inference_index_high_quality.csv')
hq_df['ego_file_prefix'] = hq_df['infer_idx'].apply(lambda x: f'ego_{x:06d}')

# 加载模型
print('加载模型...')
model = AlpamayoR1.from_pretrained(MODEL_PATH, dtype=torch.bfloat16).to("cuda")
processor = helper.get_processor(model.tokenizer)
print('模型加载完成!\n')

# 测试Frame 100
pos = 100
row = hq_df.iloc[pos]
infer_idx = int(row['infer_idx'])

print(f'Frame {infer_idx} (位置{pos}):')

for reverse in [False, True]:
    label = "原始顺序" if not reverse else "反转顺序"
    
    # 加载数据
    image_frames = load_images_for_frame(row, DATA_DIR)
    hist_xyz, hist_rot, future_xyz = load_egomotion_for_frame(row, DATA_DIR, reverse_history=reverse)
    
    # 推理
    messages = helper.create_message(image_frames.flatten(0, 1))
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        continue_final_message=True, return_dict=True, return_tensors="pt",
    )
    
    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": hist_xyz,
        "ego_history_rot": hist_rot,
    }
    model_inputs = helper.to_device(model_inputs, "cuda")
    torch.cuda.manual_seed_all(42)
    
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=copy.deepcopy(model_inputs),
            top_p=0.98, temperature=0.6,
            num_traj_samples=1,
            max_generation_length=256,
            return_extra=True,
        )
    
    # 计算ADE
    pred_xyz_np = pred_xyz.cpu().numpy()[0, 0, 0, :, :2]
    gt_xy = future_xyz.cpu().numpy()[0, 0, :, :2]
    diff = np.linalg.norm(pred_xyz_np - gt_xy, axis=1)
    ade = diff.mean()
    
    print(f"  {label}: ADE={ade:.3f}m")

print('\n完成!')
