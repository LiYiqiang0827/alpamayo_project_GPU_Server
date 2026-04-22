#!/usr/bin/env python3
"""提取所有样本的CoC文本"""
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import glob
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from einops import rearrange
import scipy.spatial.transform as spt

# CoC文本存储
coc_results = []

def load_egomotion(ego_path: str):
    df = pd.read_parquet(ego_path)
    translations = df[['x', 'y', 'z']].values
    rotations = df[['qx', 'qy', 'qz', 'qw']].values
    timestamps = df['timestamp'].values
    return translations, rotations, timestamps

def interpolate_egomotion(translations, rotations, timestamps, target_timestamps):
    interp_xyz = np.zeros((len(target_timestamps), 3))
    for i in range(3):
        interp_xyz[:, i] = np.interp(target_timestamps, timestamps, translations[:, i])
    interp_quat = np.zeros((len(target_timestamps), 4))
    for i in range(4):
        interp_quat[:, i] = np.interp(target_timestamps, timestamps, rotations[:, i])
    norms = np.linalg.norm(interp_quat, axis=1, keepdims=True)
    interp_quat = interp_quat / norms
    return interp_xyz, interp_quat

def load_local_data(base_dir, clip_id, t0_us=5_100_000):
    ego_path = f"{base_dir}/labels/egomotion/{clip_id}.egomotion.parquet"
    translations, rotations, timestamps = load_egomotion(ego_path)
    
    history_timestamps = t0_us + np.arange(-15, 1, 1) * 100000
    future_timestamps = t0_us + np.arange(1, 65, 1) * 100000
    
    ego_history_xyz, _ = interpolate_egomotion(translations, rotations, timestamps, history_timestamps)
    ego_future_xyz, _ = interpolate_egomotion(translations, rotations, timestamps, future_timestamps)
    
    t0_xyz = ego_history_xyz[-1].copy()
    t0_quat = np.array([0, 0, 0, 1])
    t0_rot = spt.Rotation.from_quat(t0_quat)
    t0_rot_inv = t0_rot.inv()
    
    ego_history_xyz_local = t0_rot_inv.apply(ego_history_xyz - t0_xyz)
    ego_future_xyz_local = t0_rot_inv.apply(ego_future_xyz - t0_xyz)
    
    return {
        "ego_history_xyz": torch.from_numpy(ego_history_xyz_local).float().unsqueeze(0).unsqueeze(0),
        "ego_history_rot": torch.eye(3).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(1, 1, 16, 1, 1),
        "ego_future_xyz": torch.from_numpy(ego_future_xyz_local).float().unsqueeze(0).unsqueeze(0),
        "clip_id": clip_id,
    }

def run_inference_with_coc(model, processor, data, helper):
    """运行推理并返回CoC"""
    # 准备输入 - 使用简化的模拟图像数据
    batch_size = 1
    num_cameras = 4
    num_frames = 4
    
    # 创建虚拟图像帧 (因为我们需要运行推理获取CoC)
    dummy_frames = torch.randn(num_cameras * num_frames, 3, 224, 224)
    messages = helper.create_message(dummy_frames)
    
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    
    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    
    model_inputs = helper.to_device(model_inputs, "cuda")
    
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs,
                top_p=0.98,
                temperature=0.6,
                num_traj_samples=1,
                max_generation_length=256,
                return_extra=True,
            )
    
    # 计算 minADE
    gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
    pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
    diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
    min_ade = diff.min()
    
    coc_text = extra["cot"][0][0][0] if "cot" in extra else "N/A"
    
    return min_ade, coc_text

def main():
    print("=" * 70)
    print("Alpamayo R1 - Chain-of-Causation 文本提取")
    print("=" * 70)
    
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from alpamayo_r1 import helper
    
    print("\n加载模型...")
    model = AlpamayoR1.from_pretrained(
        "nvidia/Alpamayo-R1-10B",
        dtype=torch.bfloat16,
        local_files_only=True,
    ).to("cuda")
    processor = helper.get_processor(model.tokenizer)
    print("模型加载完成!")
    
    base_dir = "/data01/vla/data_sample_chunk0"
    ego_files = glob.glob(f"{base_dir}/labels/egomotion/*.parquet")
    clip_ids = [Path(f).stem.split('.')[0] for f in ego_files[:20]]
    
    print(f"\n提取 {len(clip_ids)} 个样本的 CoC...\n")
    
    results = []
    for i, clip_id in enumerate(clip_ids, 1):
        print(f"[{i:2d}/{len(clip_ids)}] {clip_id}...", end=" ")
        try:
            data = load_local_data(base_dir, clip_id)
            min_ade, coc = run_inference_with_coc(model, processor, data, helper)
            results.append({
                "clip_id": clip_id,
                "minADE": min_ade,
                "CoC": coc
            })
            print(f"✅ minADE={min_ade:.2f}m")
            
            # 每5个清理显存
            if i % 5 == 0:
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"❌ {str(e)[:50]}")
    
    # 输出汇总
    print("\n" + "=" * 70)
    print("Chain-of-Causation 推理文本汇总")
    print("=" * 70)
    
    for r in results:
        print(f"\n【样本 {r['clip_id'][:20]}... | minADE: {r['minADE']:.2f}m】")
        print("-" * 70)
        print(r['CoC'])
    
    print("\n" + "=" * 70)
    print("统计汇总")
    print("=" * 70)
    minADEs = [r['minADE'] for r in results]
    print(f"样本数: {len(results)}")
    print(f"平均 minADE: {np.mean(minADEs):.2f}m")
    print(f"中位数 minADE: {np.median(minADEs):.2f}m")
    print("=" * 70)

if __name__ == "__main__":
    main()
