#!/usr/bin/env python3
"""
测试摄像头顺序 - 验证左右是否反了
"""
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import pandas as pd
import numpy as np
import torch
import av
from pathlib import Path
import scipy.spatial.transform as spt
from einops import rearrange

def load_video_frames(video_path: str, num_frames: int = 4):
    container = av.open(video_path)
    stream = container.streams.video[0]
    total_frames = stream.frames if stream.frames else 100
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    for i, frame in enumerate(container.decode(video=0)):
        if i in indices:
            img = frame.to_ndarray(format='rgb24')
            frames.append(img)
        if len(frames) >= num_frames:
            break
    container.close()
    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
    frames = np.stack(frames)
    frames = rearrange(frames, "t h w c -> t c h w")
    return torch.from_numpy(frames).float()

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

def load_data_with_camera_order(base_dir: str, clip_id: str, t0_us: int, swap_left_right: bool = False):
    """加载数据，可选交换左右摄像头"""
    ego_path = f"{base_dir}/labels/egomotion/{clip_id}.egomotion.parquet"
    translations, rotations, timestamps = load_egomotion(ego_path)
    
    history_timestamps = t0_us + np.arange(-15, 1, 1) * 100000
    future_timestamps = t0_us + np.arange(1, 65, 1) * 100000
    
    ego_history_xyz, ego_history_quat = interpolate_egomotion(
        translations, rotations, timestamps, history_timestamps
    )
    ego_future_xyz, ego_future_quat = interpolate_egomotion(
        translations, rotations, timestamps, future_timestamps
    )
    
    t0_xyz = ego_history_xyz[-1].copy()
    t0_quat = ego_history_quat[-1].copy()
    t0_rot = spt.Rotation.from_quat(t0_quat)
    t0_rot_inv = t0_rot.inv()
    
    ego_history_xyz_local = t0_rot_inv.apply(ego_history_xyz - t0_xyz)
    ego_future_xyz_local = t0_rot_inv.apply(ego_future_xyz - t0_xyz)
    ego_history_rot_local = (t0_rot_inv * spt.Rotation.from_quat(ego_history_quat)).as_matrix()
    
    # 摄像头顺序
    if swap_left_right:
        # 交换 left 和 right
        cameras = [
            ("camera_cross_right_120fov", 0),  # 原本 left 的位置放 right
            ("camera_front_wide_120fov", 1),
            ("camera_cross_left_120fov", 2),   # 原本 right 的位置放 left
            ("camera_front_tele_30fov", 6),
        ]
        print("    [使用交换顺序: right, front, left, tele]")
    else:
        # 原始顺序
        cameras = [
            ("camera_cross_left_120fov", 0),
            ("camera_front_wide_120fov", 1),
            ("camera_cross_right_120fov", 2),
            ("camera_front_tele_30fov", 6),
        ]
        print("    [使用原始顺序: left, front, right, tele]")
    
    image_frames_list = []
    camera_indices_list = []
    
    for cam_name, cam_idx in cameras:
        video_path = f"{base_dir}/camera/{cam_name}/{clip_id}.{cam_name}.mp4"
        if os.path.exists(video_path):
            try:
                frames = load_video_frames(video_path, num_frames=4)
                image_frames_list.append(frames)
                camera_indices_list.append(cam_idx)
            except Exception as e:
                print(f"  Warning: Failed to load {cam_name}: {e}")
    
    if not image_frames_list:
        raise FileNotFoundError("No camera data found!")
    
    image_frames = torch.stack(image_frames_list, dim=0)
    camera_indices = torch.tensor(camera_indices_list, dtype=torch.int64)
    
    # 按索引排序
    sort_order = torch.argsort(camera_indices)
    image_frames = image_frames[sort_order]
    camera_indices = camera_indices[sort_order]
    
    return {
        "image_frames": image_frames,
        "camera_indices": camera_indices,
        "ego_history_xyz": torch.from_numpy(ego_history_xyz_local).float().unsqueeze(0).unsqueeze(0),
        "ego_history_rot": torch.from_numpy(ego_history_rot_local).float().unsqueeze(0).unsqueeze(0),
        "ego_future_xyz": torch.from_numpy(ego_future_xyz_local).float().unsqueeze(0).unsqueeze(0),
    }

def run_inference(model, processor, data, helper, num_traj_samples=6):
    frames = data["image_frames"].flatten(0, 1)
    messages = helper.create_message(frames)
    
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
                num_traj_samples=num_traj_samples,
                max_generation_length=256,
                return_extra=True,
            )
    
    gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
    pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
    diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
    min_ade = diff.min()
    coc_text = extra["cot"][0][0][0] if "cot" in extra else "N/A"
    
    return min_ade, coc_text

def main():
    print("=" * 70)
    print("摄像头顺序测试 - 验证左右是否反了")
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
    
    # 选择几个样本测试
    test_samples = [
        ("43ada34d-283f-46b9-ae38-cd41c576b889", "最佳样本"),
        ("fa83bcb8-ea31-4dbb-b447-4fb458f5984b", "最差样本(可能变道)"),
        ("ef4264ed-0fd2-4a64-9831-87e1aae28407", "良好样本"),
    ]
    
    print(f"\n测试 {len(test_samples)} 个样本...")
    print("对比: 原始顺序 [left, front, right, tele] vs 交换顺序 [right, front, left, tele]\n")
    
    results = []
    for clip_id, desc in test_samples:
        print(f"【{desc}】")
        print(f"  Clip: {clip_id}")
        
        try:
            # 测试原始顺序
            print("  原始顺序:")
            data_orig = load_data_with_camera_order(base_dir, clip_id, 5_100_000, swap_left_right=False)
            minADE_orig, coc_orig = run_inference(model, processor, data_orig, helper, num_traj_samples=6)
            print(f"    ✅ minADE: {minADE_orig:.2f}m")
            
            # 清理显存
            torch.cuda.empty_cache()
            
            # 测试交换顺序
            print("  交换顺序:")
            data_swap = load_data_with_camera_order(base_dir, clip_id, 5_100_000, swap_left_right=True)
            minADE_swap, coc_swap = run_inference(model, processor, data_swap, helper, num_traj_samples=6)
            print(f"    ✅ minADE: {minADE_swap:.2f}m")
            
            # 计算改善
            improvement = minADE_orig - minADE_swap
            better = "交换更好" if improvement < 0 else "原始更好"
            
            results.append({
                "clip_id": clip_id,
                "desc": desc,
                "minADE_orig": minADE_orig,
                "minADE_swap": minADE_swap,
                "improvement": improvement,
                "better": better,
            })
            
            print(f"  对比: {better} (差值: {improvement:+.2f}m)\n")
            
        except Exception as e:
            print(f"  ❌ 失败: {str(e)[:60]}\n")
        
        torch.cuda.empty_cache()
    
    # 输出汇总
    print("=" * 70)
    print("结果汇总")
    print("=" * 70)
    print(f"{'样本':<20} {'原始':<10} {'交换':<10} {'差值':<10} {'结论'}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['desc']:<20} {r['minADE_orig']:<10.2f} {r['minADE_swap']:<10.2f} {r['improvement']:+8.2f}m  {r['better']}")
    
    # 统计
    if results:
        swap_better = sum(1 for r in results if r['improvement'] < 0)
        orig_better = sum(1 for r in results if r['improvement'] > 0)
        print(f"\n统计: 交换更好 {swap_better} 个, 原始更好 {orig_better} 个")
        
        if swap_better > orig_better:
            print("⚠️ 结论: 左右摄像头可能真的反了！建议交换顺序。")
        else:
            print("✅ 结论: 原始顺序正确，无需交换。")
    
    print("=" * 70)

if __name__ == "__main__":
    main()
