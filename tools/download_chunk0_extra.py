#!/usr/bin/env python3
"""Download calibration, egomotion.offline, obstacle.offline for chunk 0"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import hf_hub_download

BASE_DIR = "/data01/mikelee/data"
CHUNK_ID = 0

# Files to download
FILES = [
    # Calibration (6 files)
    "calibration/camera_intrinsics/camera_intrinsics.chunk_0000.parquet",
    "calibration/camera_intrinsics.offline/camera_intrinsics.offline.chunk_0000.parquet",
    "calibration/lidar_intrinsics.offline/lidar_intrinsics.offline.chunk_0000.parquet",
    "calibration/sensor_extrinsics/sensor_extrinsics.chunk_0000.parquet",
    "calibration/sensor_extrinsics.offline/sensor_extrinsics.offline.chunk_0000.parquet",
    "calibration/vehicle_dimensions/vehicle_dimensions.chunk_0000.parquet",
    # Labels offline
    "labels/egomotion.offline/egomotion.offline.chunk_0000.zip",
    "labels/obstacle.offline/obstacle.offline.chunk_0000.zip",
]

print(f"Downloading {len(FILES)} files for chunk {CHUNK_ID}...")
print(f"Base dir: {BASE_DIR}")

for fname in FILES:
    folder = "/".join(fname.split("/")[:-1])
    print(f"\n📥 {folder}/")
    try:
        path = hf_hub_download(
            repo_id="nvidia/PhysicalAI-Autonomous-Vehicles",
            filename=fname,
            repo_type="dataset",
            local_dir=BASE_DIR,
        )
        size = os.path.getsize(path)
        fname_short = fname.split("/")[-1]
        if size > 1024*1024:
            print(f"   ✅ {fname_short}: {size/1024/1024:.1f} MB")
        else:
            print(f"   ✅ {fname_short}: {size/1024:.1f} KB")
    except Exception as e:
        print(f"   ❌ Failed: {e}")

print("\n✅ Done!")
