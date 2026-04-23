#!/usr/bin/env python3
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
import os
import sys
from huggingface_hub import hf_hub_download


# Files to download for each chunk (chunked files)
# 4 cameras + 3 labels (egomotion, egomotion.offline, obstacle.offline)
CHUNK_FILES = [
    # Cameras (4)
    "camera/camera_front_wide_120fov/camera_front_wide_120fov.chunk_{:04d}.zip",
    "camera/camera_front_tele_30fov/camera_front_tele_30fov.chunk_{:04d}.zip",
    "camera/camera_cross_left_120fov/camera_cross_left_120fov.chunk_{:04d}.zip",
    "camera/camera_cross_right_120fov/camera_cross_right_120fov.chunk_{:04d}.zip",
    # Labels (3)
    "labels/egomotion/egomotion.chunk_{:04d}.zip",
    "labels/egomotion.offline/egomotion.offline.chunk_{:04d}.zip",
    "labels/obstacle.offline/obstacle.offline.chunk_{:04d}.zip",
]

# Calibration files (also chunked, one per chunk)
CALIBRATION_FILES = [
    "calibration/camera_intrinsics/camera_intrinsics.chunk_{:04d}.parquet",
    "calibration/camera_intrinsics.offline/camera_intrinsics.offline.chunk_{:04d}.parquet",
    "calibration/lidar_intrinsics.offline/lidar_intrinsics.offline.chunk_{:04d}.parquet",
    "calibration/sensor_extrinsics/sensor_extrinsics.chunk_{:04d}.parquet",
    "calibration/sensor_extrinsics.offline/sensor_extrinsics.offline.chunk_{:04d}.parquet",
    "calibration/vehicle_dimensions/vehicle_dimensions.chunk_{:04d}.parquet",
]


def parse_chunks(chunks_str: str) -> list[int]:
    """
    Parse chunk string into list of chunk IDs.
    
    Examples:
        "1" -> [1]
        "1,3,5" -> [1, 3, 5]
        "10:30" -> [10, 11, 12, ..., 30]
        "1,5,10:15" -> [1, 5, 10, 11, 12, 13, 14, 15]
    """
    chunk_ids = set()
    parts = chunks_str.split(",")
    
    for part in parts:
        part = part.strip()
        if ":" in part:
            # Range format: "10:30"
            start, end = part.split(":")
            start, end = int(start.strip()), int(end.strip())
            if start > end:
                raise ValueError(f"Invalid range: {part} (start > end)")
            chunk_ids.update(range(start, end + 1))
        else:
            # Single number
            chunk_ids.add(int(part))
    
    return sorted(list(chunk_ids))


def download_file(filename: str, base_dir: str, verbose: bool = True) -> dict:
    """
    Download a single file.
    
    Returns:
        dict with download statistics
    """
    folder_name = "/".join(filename.split("/")[:-1])
    
    if verbose:
        print(f"\n📥 {folder_name}/")
    
    stats = {
        "filename": filename,
        "bytes": 0,
        "error": None,
    }
    
    try:
        path = hf_hub_download(
            repo_id="nvidia/PhysicalAI-Autonomous-Vehicles",
            filename=filename,
            repo_type="dataset",
            local_dir=base_dir,
        )
        
        size = os.path.getsize(path)
        stats["bytes"] = size
        
        if verbose:
            fname = filename.split('/')[-1]
            if size > 1024*1024:
                print(f"   ✅ {fname}: {size/1024/1024:.1f} MB")
            else:
                print(f"   ✅ {fname}: {size/1024:.1f} KB")
            
    except Exception as e:
        stats["error"] = f"❌ Failed to download {filename}: {str(e)}"
        if verbose:
            print(f"   {stats['error']}")
    
    return stats


def download_chunk(chunk_id: int, base_dir: str, verbose: bool = True) -> dict:
    """
    Download all files for a single chunk (cameras + labels + calibration).
    
    Returns:
        dict with download statistics
    """
    # Format chunk ID to 4 digits
    chunk_str = f"{chunk_id:04d}"
    
    # Create directory: ~/alpamayo_project/data/data_sample_chunk{id}/
    chunk_dir = os.path.join(base_dir, f"data_sample_chunk{chunk_id}")
    os.makedirs(chunk_dir, exist_ok=True)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Downloading chunk {chunk_id} -> {chunk_dir}")
        print(f"{'='*60}")
    
    stats = {
        "chunk_id": chunk_id,
        "files": [],
        "total_bytes": 0,
        "errors": [],
    }
    
    # Download camera + label files
    for file_template in CHUNK_FILES:
        filename = file_template.format(chunk_id)
        
        folder_name = "/".join(filename.split("/")[:-1])
        
        if verbose:
            print(f"\n📥 {folder_name}/")
        
        try:
            path = hf_hub_download(
                repo_id="nvidia/PhysicalAI-Autonomous-Vehicles",
                filename=filename,
                repo_type="dataset",
                local_dir=chunk_dir,
            )
            
            size = os.path.getsize(path)
            stats["total_bytes"] += size
            stats["files"].append(filename)
            
            if verbose:
                fname = filename.split('/')[-1]
                if size > 1024*1024*1024:
                    print(f"   ✅ {fname}: {size/1024/1024/1024:.2f} GB")
                else:
                    print(f"   ✅ {fname}: {size/1024/1024:.1f} MB")
                
        except Exception as e:
            error_msg = f"❌ Failed to download {filename}: {str(e)}"
            stats["errors"].append(error_msg)
            if verbose:
                print(f"   {error_msg}")
    
    # Download calibration files (also chunked)
    for file_template in CALIBRATION_FILES:
        filename = file_template.format(chunk_id)
        
        folder_name = "/".join(filename.split("/")[:-1])
        
        if verbose:
            print(f"\n📥 {folder_name}/")
        
        try:
            path = hf_hub_download(
                repo_id="nvidia/PhysicalAI-Autonomous-Vehicles",
                filename=filename,
                repo_type="dataset",
                local_dir=chunk_dir,
            )
            
            size = os.path.getsize(path)
            stats["total_bytes"] += size
            stats["files"].append(filename)
            
            if verbose:
                fname = filename.split('/')[-1]
                print(f"   ✅ {fname}: {size/1024:.1f} KB")
                
        except Exception as e:
            error_msg = f"❌ Failed to download {filename}: {str(e)}"
            stats["errors"].append(error_msg)
            if verbose:
                print(f"   {error_msg}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Download NVIDIA PhysicalAI dataset chunks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 download_physicalai.py --chunks 1
    python3 download_physicalai.py --chunks 1,3,5
    python3 download_physicalai.py --chunks 10:30
    python3 download_physicalai.py --chunks 1,5,10:15 --base-dir /data/nvidia
        """
    )
    
    parser.add_argument(
        "--chunks", "-c",
        required=True,
        help="Chunk IDs or ranges. Examples: '1', '1,3,5', '10:30', '1,5,10:15'"
    )
    
    parser.add_argument(
        "--base-dir", "-d",
        default="/data01/mikelee/data",
        help="Base directory for downloads (default: /data01/mikelee/data)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Expand user home directory
    base_dir = os.path.expanduser(args.base_dir)
    
    # Parse chunk IDs
    try:
        chunk_ids = parse_chunks(args.chunks)
    except ValueError as e:
        print(f"❌ Error parsing chunks: {e}", file=sys.stderr)
        sys.exit(1)
    
    files_per_chunk = len(CHUNK_FILES) + len(CALIBRATION_FILES)
    
    print(f"🎯 Chunks to download: {chunk_ids}")
    print(f"📁 Base directory: {base_dir}")
    print(f"📦 Files per chunk: {files_per_chunk}")
    print(f"   - Camera files: {len(CHUNK_FILES)}")
    print(f"   - Calibration files: {len(CALIBRATION_FILES)}")
    
    # Verify hf is logged in
    from huggingface_hub import whoami
    try:
        user = whoami()
        print(f"✅ Logged in as: {user['name']}")
    except Exception:
        print("❌ Not logged in. Run 'huggingface-cli login' first.")
        sys.exit(1)
    
    # Download each chunk
    total_bytes = 0
    total_files = 0
    all_errors = []
    
    for chunk_id in chunk_ids:
        stats = download_chunk(chunk_id, base_dir, verbose=args.verbose)
        total_bytes += stats["total_bytes"]
        total_files += len(stats["files"])
        all_errors.extend(stats["errors"])
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    print(f"   Chunks downloaded: {len(chunk_ids)}")
    print(f"   Files downloaded: {total_files}")
    print(f"   Total size: {total_bytes/1024/1024/1024:.2f} GB")
    print(f"   Base location: {base_dir}")
    
    if all_errors:
        print(f"\n⚠️  Errors encountered:")
        for err in all_errors[:10]:  # Show first 10 errors
            print(f"   {err}")
        if len(all_errors) > 10:
            print(f"   ... and {len(all_errors)-10} more errors")
        sys.exit(1)
    else:
        print(f"\n✅ All downloads completed successfully!")
    
    # Print folder structure
    print(f"\n📂 Folder structure:")
    for chunk_id in chunk_ids:
        chunk_dir = os.path.join(base_dir, f"data_sample_chunk{chunk_id}")
        print(f"   {os.path.basename(chunk_dir)}/")
        for f in os.listdir(chunk_dir):
            subpath = os.path.join(chunk_dir, f)
            if os.path.isdir(subpath):
                contents = os.listdir(subpath)
                for subf in contents[:2]:
                    print(f"      {f}/{subf}")
                if len(contents) > 2:
                    print(f"      ... and {len(contents)-2} more")
            else:
                print(f"      {f}")


if __name__ == "__main__":
    main()
