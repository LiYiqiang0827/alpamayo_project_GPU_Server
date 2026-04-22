"""Load data from local downloaded dataset for model inference."""

import os
os.environ["HF_HUB_DISABLE_XNET"] = "1"

from typing import Any

import cv2
import numpy as np
import pandas as pd
import scipy.spatial.transform as spt
import torch
from einops import rearrange


LOCAL_DATA_ROOT = "/data01/vla"


def load_local_physical_aiavdataset(
    clip_id: str,
    t0_us: int,
    chunk_id: int = 0,
    num_history_steps: int = 16,
    num_future_steps: int = 64,
    time_step: float = 0.1,
    camera_features: list | None = None,
    num_frames: int = 4,
) -> dict[str, Any]:
    """Load data from local dataset for model inference."""

    chunk_dir = os.path.join(LOCAL_DATA_ROOT, f"data_sample_chunk{chunk_id}")

    if camera_features is None:
        camera_features = [
            "camera_cross_left_120fov",
            "camera_front_wide_120fov",
            "camera_cross_right_120fov",
            "camera_front_tele_30fov",
        ]

    camera_name_to_index = {
        "camera_cross_left_120fov": 0,
        "camera_front_wide_120fov": 1,
        "camera_cross_right_120fov": 2,
        "camera_rear_left_70fov": 3,
        "camera_rear_tele_30fov": 4,
        "camera_rear_right_70fov": 5,
        "camera_front_tele_30fov": 6,
    }

    egomotion_path = os.path.join(chunk_dir, "labels", "egomotion", f"{clip_id}.egomotion.parquet")
    import scipy.interpolate as interp
    
    ego_df = pd.read_parquet(egomotion_path)
    
    all_timestamps = ego_df["timestamp"].values
    ego_x = ego_df["x"].values
    ego_y = ego_df["y"].values
    ego_z = ego_df["z"].values
    ego_qx = ego_df["qx"].values
    ego_qy = ego_df["qy"].values
    ego_qz = ego_df["qz"].values
    ego_qw = ego_df["qw"].values
    
    interp_x = interp.interp1d(all_timestamps, ego_x, kind='linear', fill_value='extrapolate')
    interp_y = interp.interp1d(all_timestamps, ego_y, kind='linear', fill_value='extrapolate')
    interp_z = interp.interp1d(all_timestamps, ego_z, kind='linear', fill_value='extrapolate')
    interp_qx = interp.interp1d(all_timestamps, ego_qx, kind='linear', fill_value='extrapolate')
    interp_qy = interp.interp1d(all_timestamps, ego_qy, kind='linear', fill_value='extrapolate')
    interp_qz = interp.interp1d(all_timestamps, ego_qz, kind='linear', fill_value='extrapolate')
    interp_qw = interp.interp1d(all_timestamps, ego_qw, kind='linear', fill_value='extrapolate')
    
    history_offsets_us = np.arange(
        -(num_history_steps - 1) * time_step * 1_000_000,
        time_step * 1_000_000 / 2,
        time_step * 1_000_000,
    ).astype(np.int64)
    history_timestamps = t0_us + history_offsets_us

    future_offsets_us = np.arange(
        time_step * 1_000_000,
        (num_future_steps + 0.5) * time_step * 1_000_000,
        time_step * 1_000_000,
    ).astype(np.int64)
    future_timestamps = t0_us + future_offsets_us

    ego_history_xyz = np.column_stack([
        interp_x(history_timestamps),
        interp_y(history_timestamps),
        interp_z(history_timestamps)
    ])
    ego_history_quat = np.column_stack([
        interp_qx(history_timestamps),
        interp_qy(history_timestamps),
        interp_qz(history_timestamps),
        interp_qw(history_timestamps)
    ])

    ego_future_xyz = np.column_stack([
        interp_x(future_timestamps),
        interp_y(future_timestamps),
        interp_z(future_timestamps)
    ])
    ego_future_quat = np.column_stack([
        interp_qx(future_timestamps),
        interp_qy(future_timestamps),
        interp_qz(future_timestamps),
        interp_qw(future_timestamps)
    ])

    print(f"Loaded history: {ego_history_xyz.shape}, future: {ego_future_xyz.shape}")

    if len(history_timestamps) == 0:
        raise ValueError("No history timestamps found in egomotion data")

    t0_xyz = ego_history_xyz[-1].copy()
    t0_quat = ego_history_quat[-1].copy()
    t0_rot = spt.Rotation.from_quat(t0_quat.copy())
    t0_rot_inv = t0_rot.inv()

    ego_history_xyz_local = t0_rot_inv.apply(ego_history_xyz - t0_xyz)
    ego_future_xyz_local = t0_rot_inv.apply(ego_future_xyz - t0_xyz)

    ego_history_rot_local = (t0_rot_inv * spt.Rotation.from_quat(ego_history_quat)).as_matrix()
    ego_future_rot_local = (t0_rot_inv * spt.Rotation.from_quat(ego_future_quat)).as_matrix()

    ego_history_xyz_tensor = torch.from_numpy(ego_history_xyz_local).float().unsqueeze(0).unsqueeze(0)
    ego_history_rot_tensor = torch.from_numpy(ego_history_rot_local).float().unsqueeze(0).unsqueeze(0)
    ego_future_xyz_tensor = torch.from_numpy(ego_future_xyz_local).float().unsqueeze(0).unsqueeze(0)
    ego_future_rot_tensor = torch.from_numpy(ego_future_rot_local).float().unsqueeze(0).unsqueeze(0)

    image_frames_list = []
    camera_indices_list = []
    timestamps_list = []

    image_timestamps = np.array(
        [t0_us - (num_frames - 1 - i) * int(time_step * 1_000_000) for i in range(num_frames)],
        dtype=np.int64,
    )

    camera_dir = os.path.join(chunk_dir, "camera")

    for cam_feature in camera_features:
        # 先检查子目录
        cam_path = os.path.join(camera_dir, cam_feature, f"{clip_id}.{cam_feature}.mp4")
        ts_path = os.path.join(camera_dir, cam_feature, f"{clip_id}.{cam_feature}.timestamps.parquet")
        
        # 如果子目录没有，检查主目录
        if not os.path.exists(cam_path):
            cam_path = os.path.join(camera_dir, f"{clip_id}.{cam_feature}.mp4")
            ts_path = os.path.join(camera_dir, f"{clip_id}.{cam_feature}.timestamps.parquet")

        if not os.path.exists(cam_path):
            print(f"Warning: {cam_path} not found, skipping...")
            continue

        timestamps_df = pd.read_parquet(ts_path)
        if "timestamp_us" in timestamps_df.columns:
            ts_col = "timestamp_us"
        elif "timestamp" in timestamps_df.columns:
            ts_col = "timestamp"
        else:
            print(f"Warning: No timestamp column found in {ts_path}")
            continue

        available_cam_ts = timestamps_df[ts_col].values

        frame_indices = []
        for ts in image_timestamps:
            if len(available_cam_ts) == 0:
                continue
            closest_idx = (available_cam_ts - ts).argmin()
            frame_indices.append(closest_idx)

        if len(frame_indices) == 0:
            print(f"Warning: No frames could be selected for {cam_feature}")
            continue

        cap = cv2.VideoCapture(cam_path)
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        cap.release()

        if len(frames) == 0:
            print(f"Warning: No frames read from {cam_path}")
            continue

        frames_array = np.stack(frames)
        frames_tensor = torch.from_numpy(frames_array).float()
        frames_tensor = rearrange(frames_tensor, "t h w c -> t c h w")

        cam_idx = camera_name_to_index.get(cam_feature, 0)
        image_frames_list.append(frames_tensor)
        camera_indices_list.append(cam_idx)

        frame_ts = timestamps_df.iloc[frame_indices][ts_col].values
        timestamps_list.append(torch.from_numpy(frame_ts.astype(np.int64)))

    if len(image_frames_list) == 0:
        raise ValueError("No camera data could be loaded")

    image_frames = torch.stack(image_frames_list, dim=0)
    camera_indices = torch.tensor(camera_indices_list, dtype=torch.int64)
    all_timestamps = torch.stack(timestamps_list, dim=0)

    sort_order = torch.argsort(camera_indices)
    image_frames = image_frames[sort_order]
    camera_indices = camera_indices[sort_order]
    all_timestamps = all_timestamps[sort_order]

    camera_tmin = all_timestamps.min()
    relative_timestamps = (all_timestamps - camera_tmin).float() * 1e-6

    return {
        "image_frames": image_frames,
        "camera_indices": camera_indices,
        "ego_history_xyz": ego_history_xyz_tensor,
        "ego_history_rot": ego_history_rot_tensor,
        "ego_future_xyz": ego_future_xyz_tensor,
        "ego_future_rot": ego_future_rot_tensor,
        "relative_timestamps": relative_timestamps,
        "absolute_timestamps": all_timestamps,
        "t0_us": t0_us,
        "clip_id": clip_id,
    }


if __name__ == "__main__":
    clip_id = "01d3588e-bca7-4a18-8e74-c6cfe9e996db"
    print(f"Loading clip: {clip_id}")
    
    ego_df = pd.read_parquet(f"/data02/vla/data_sample_chunk0/labels/egomotion/{clip_id}.egomotion.parquet")
    ts = ego_df['timestamp'].values
    mid_idx = len(ts) // 2
    t0_us = int(ts[mid_idx])
    print(f"Using t0_us: {t0_us}")
    
    data = load_local_physical_aiavdataset(clip_id, t0_us=t0_us)
    print("Data loaded!")
    print(f"image_frames shape: {data['image_frames'].shape}")
    print(f"ego_history_xyz shape: {data['ego_history_xyz'].shape}")
