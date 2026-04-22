#!/usr/bin/env python3
"""验证第二个clip的前几帧"""
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

def load_egomotion_for_frame(row, data_dir):
    prefix = row['ego_file_prefix']
    ego_dir = f'{data_dir}/egomotion'
    hist_local = np.load(f'{ego_dir}/{prefix}_history_local.npy', allow_pickle=True).item()
    future_gt = np.load(f'{ego_dir}/{prefix}_future_gt.npy', allow_pickle=True).item()
    
    hist_xyz = torch.from_numpy(hist_local['xyz']).float().unsqueeze(0).unsqueeze(0)
    hist_rot = torch.from_numpy(hist_local['rotation_matrix']).float().unsqueeze(0).unsqueeze(0)
    future_xyz = torch.from_numpy(future_gt['xyz']).float().unsqueeze(0).unsqueeze(0)
    return hist_xyz, hist_rot, future_xyz

print(f'=== 验证第二个clip: {CLIP_ID} ===\n')

# 加载高质量索引
print('1. 加载高质量索引...')
hq_df = pd.read_csv(f'{DATA_DIR}/inference_index_high_quality.csv')
hq_df['ego_file_prefix'] = hq_df['infer_idx'].apply(lambda x: f'ego_{x:06d}')
print(f'   高质量索引: {len(hq_df)} 帧')

# 加载模型
print('\n2. 加载模型...')
model = AlpamayoR1.from_pretrained(MODEL_PATH, dtype=torch.bfloat16).to("cuda")
processor = helper.get_processor(model.tokenizer)
print('   模型加载完成!')

# 测试前5帧
test_positions = [0, 1, 2, 3, 4]
print(f'\n3. 推理前5帧 (位置: {test_positions})...')
print(f'   对应infer_idx: {hq_df.iloc[test_positions]["infer_idx"].tolist()}')

results = []
for pos in test_positions:
    row = hq_df.iloc[pos]
    infer_idx = int(row['infer_idx'])
    
    # 加载数据
    image_frames = load_images_for_frame(row, DATA_DIR)
    hist_xyz, hist_rot, future_xyz = load_egomotion_for_frame(row, DATA_DIR)
    
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
    
    coc = extra.get("cot", [[[]]])[0]
    if isinstance(coc, np.ndarray):
        coc = coc.tolist()
    
    results.append({'pos': pos, 'infer_idx': infer_idx, 'ade': ade, 'coc': coc})
    print(f"   位置{pos} (infer_idx={infer_idx}): ADE={ade:.3f}m")

print('\n=== 验证结果 ===')
for r in results:
    status = "✅" if r['ade'] < 1.5 else ("⚠️" if r['ade'] < 3.0 else "❌")
    print(f"{status} 位置{r['pos']} (infer_idx={r['infer_idx']}): ADE={r['ade']:.3f}m")

print('\n✅ 第二个clip验证完成!')
