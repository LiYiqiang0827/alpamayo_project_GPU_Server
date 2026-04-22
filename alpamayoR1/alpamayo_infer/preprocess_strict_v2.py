#!/usr/bin/env python3
"""
Preprocess clips with strict timestamp alignment (no interpolation version)
Key changes from v4:
1. Image frames: f3 only looks backward (no future), f0-f2 can look both ways
2. Egomotion: Use discrete frames (every 10 frames for 100ms step) instead of interpolation
3. Strict validation: Any timestamp mismatch > threshold marks frame as invalid

Usage:
  python preprocess_strict.py --clip <clip_id> --chunk 0                    # 单clip
  python preprocess_strict.py --clip clip1,clip2,clip3 --chunk 0            # 多clip
  python preprocess_strict.py --clip 0 --chunk 0                            # 处理chunk0所有clips
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
import cv2
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing

# Configuration
HISTORY_STEPS = 16  # 1.6 seconds at 10Hz
FUTURE_STEPS = 64  # 6.4 seconds at 10Hz
TIME_STEP = 0.1  # 100ms
CAMERAS = [
    "camera_cross_left_120fov",
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_front_tele_30fov",
]
IMG_TIME_OFFSETS_MS = [300, 200, 100, 0]  # f0, f1, f2, f3
MAX_IMAGE_DIFF_MS = 33  # Maximum allowed time difference for images
MAX_EGO_DIFF_MS = (
    30  # Maximum allowed time difference for egomotion validation (was 10ms, now 30ms)
)


def preprocess_clip_strict(clip_id, data_root, output_root, force=False):
    """
    Preprocess a single clip with strict timestamp alignment
    """
    data_path = Path(data_root)
    output_path = Path(output_root) / clip_id

    print(f"\n{'=' * 60}")
    print(f"Processing clip: {clip_id}")
    print(f"{'=' * 60}")
    print(f"Data root: {data_path}")
    print(f"Output: {output_path}")

    # Check if already processed
    if (output_path / "data" / "inference_index_strict.csv").exists():
        if force:
            print(f"🗑️  Force mode: removing existing output...")
            import shutil

            shutil.rmtree(output_path, ignore_errors=True)
            print(f"✅ Removed existing output")
        else:
            print(f"⏭️  Already processed, skipping...")
            return output_path, "skipped"

    # Load egomotion from labels/egomotion/
    ego_path = data_path / "labels" / "egomotion" / f"{clip_id}.egomotion.parquet"
    if not ego_path.exists():
        print(f"ERROR: Egomotion not found at {ego_path}")
        return None, "no_egomotion"

    ego_df = pd.read_parquet(ego_path)

    print(
        f"Egomotion data: {len(ego_df)} frames, {ego_df['timestamp'].iloc[-1] / 1e6:.2f}s duration"
    )

    # Load camera timestamps from camera/
    cam_timestamps = {}
    for cam in CAMERAS:
        ts_path = data_path / "camera" / f"{clip_id}.{cam}.timestamps.parquet"
        if ts_path.exists():
            cam_timestamps[cam] = pd.read_parquet(ts_path)
            print(f"  {cam}: {len(cam_timestamps[cam])} frames")
        else:
            print(f"  WARNING: {cam} timestamps not found at {ts_path}")
            return None, "missing_camera_timestamps"

    # Find common time range for all cameras
    cam_ranges = {}
    for cam in CAMERAS:
        cam_ranges[cam] = (
            cam_timestamps[cam]["timestamp"].iloc[0],
            cam_timestamps[cam]["timestamp"].iloc[-1],
        )

    # Common range: max of starts, min of ends
    common_start = max([cam_ranges[c][0] for c in CAMERAS])
    common_end = min([cam_ranges[c][1] for c in CAMERAS])

    print(
        f"Common camera time range: {common_start / 1e6:.3f}s - {common_end / 1e6:.3f}s"
    )

    # Filter egomotion to common range
    ego_valid = ego_df[
        (ego_df["timestamp"] >= common_start) & (ego_df["timestamp"] <= common_end)
    ].copy()

    print(f"Egomotion in common range: {len(ego_valid)} frames")

    # Build valid inference index
    valid_frames = []

    # Statistics counters
    stats = {
        "total_checked": 0,
        "filtered_by_history_time": 0,  # 原因1: Egomotion历史时间差>10ms
        "filtered_by_future_time": 0,  # 原因2: Egomotion未来时间差>10ms
        "filtered_by_camera": 0,  # 原因3: 相机帧时间差>33ms
    }

    for idx in range(len(ego_valid)):
        ego_idx = ego_valid.index[idx]
        ego_ts = ego_valid.iloc[idx]["timestamp"]

        stats["total_checked"] += 1

        # === Validate Egomotion History (包含 t0 本身) ===
        history_indices = []
        history_valid = True

        # i=0 是 t0 本身，i=1..15 是历史 100ms..1500ms
        for i in range(0, HISTORY_STEPS):
            target_ts = ego_ts - i * TIME_STEP * 1e6  # i=0: t0, i=1: t0-100ms, ...
            # Find nearest egomotion frame
            idx = (ego_df["timestamp"] - target_ts).abs().idxmin()
            closest_ts = ego_df.loc[idx, "timestamp"]
            diff_ms = abs(target_ts - closest_ts) / 1000.0

            if diff_ms > MAX_EGO_DIFF_MS:  # > 10ms
                history_valid = False
                break

            history_indices.append(idx)

        history_indices.reverse()  # Now ordered from oldest to newest (t0-1500ms ... t0)

        if not history_valid:
            stats["filtered_by_history_time"] += 1
            continue

        # === Validate Egomotion Future (New Method: find nearest by timestamp) ===
        future_indices = []
        future_valid = True

        for i in range(1, FUTURE_STEPS + 1):
            target_ts = ego_ts + i * TIME_STEP * 1e6  # 100ms intervals
            # Find nearest egomotion frame
            idx = (ego_df["timestamp"] - target_ts).abs().idxmin()
            closest_ts = ego_df.loc[idx, "timestamp"]
            diff_ms = abs(target_ts - closest_ts) / 1000.0

            if diff_ms > MAX_EGO_DIFF_MS:  # > 10ms
                future_valid = False
                break

            future_indices.append(idx)

        if not future_valid:
            stats["filtered_by_future_time"] += 1
            continue

        # === Validate Camera Frames ===
        cam_valid = True
        cam_frames = {cam: [] for cam in CAMERAS}

        for i, offset_ms in enumerate(IMG_TIME_OFFSETS_MS):
            target_ts = ego_ts - offset_ms * 1000

            for cam in CAMERAS:
                cam_ts_df = cam_timestamps[cam]

                if i == 3:  # f3 (t=0): only look backward (<= target_ts)
                    # Find latest frame <= target_ts
                    valid_frames_cam = cam_ts_df[cam_ts_df["timestamp"] <= target_ts]
                    if len(valid_frames_cam) == 0:
                        cam_valid = False
                        break

                    closest_idx = valid_frames_cam.index[-1]
                    closest_ts = valid_frames_cam.iloc[-1]["timestamp"]
                    diff_ms = abs(target_ts - closest_ts) / 1000.0

                    if diff_ms > MAX_IMAGE_DIFF_MS:
                        cam_valid = False
                        break

                    cam_frames[cam].append((closest_idx, closest_ts, diff_ms))

                else:  # f0, f1, f2: can look both ways
                    # Find closest frame (both before and after)
                    idx = (cam_ts_df["timestamp"] - target_ts).abs().idxmin()
                    closest_ts = cam_ts_df.loc[idx, "timestamp"]
                    diff_ms = abs(target_ts - closest_ts) / 1000.0

                    if diff_ms > MAX_IMAGE_DIFF_MS:
                        cam_valid = False
                        break

                    cam_frames[cam].append((idx, closest_ts, diff_ms))

            if not cam_valid:
                break

        if not cam_valid:
            stats["filtered_by_camera"] += 1
            continue

        # All validations passed
        valid_frames.append(
            {
                "ego_idx": ego_idx,
                "ego_ts": ego_ts,
                "history_indices": history_indices,
                "future_indices": future_indices,
                "cam_frames": cam_frames,
            }
        )

    print(f"\nValid inference frames: {len(valid_frames)}")

    # Print filtering statistics
    print("\n=== Filtering Statistics ===")
    total_filtered = stats["total_checked"] - len(valid_frames)
    print(f"Total checked: {stats['total_checked']}")
    print(f"Total filtered: {total_filtered}")
    print(
        f"  1. Egomotion history time diff (> {MAX_EGO_DIFF_MS}ms): {stats['filtered_by_history_time']} ({stats['filtered_by_history_time'] / stats['total_checked'] * 100:.1f}%)"
    )
    print(
        f"  2. Egomotion future time diff (> {MAX_EGO_DIFF_MS}ms): {stats['filtered_by_future_time']} ({stats['filtered_by_future_time'] / stats['total_checked'] * 100:.1f}%)"
    )
    print(
        f"  3. Camera frame alignment (> {MAX_IMAGE_DIFF_MS}ms): {stats['filtered_by_camera']} ({stats['filtered_by_camera'] / stats['total_checked'] * 100:.1f}%)"
    )

    # Find the top reason
    reasons = [
        ("Egomotion history time diff", stats["filtered_by_history_time"]),
        ("Egomotion future time diff", stats["filtered_by_future_time"]),
        ("Camera frame alignment", stats["filtered_by_camera"]),
    ]
    top_reason = max(reasons, key=lambda x: x[1])
    print(
        f"\n🔴 Top filtering reason: {top_reason[0]} ({top_reason[1]} frames, {top_reason[1] / stats['total_checked'] * 100:.1f}%)"
    )

    if len(valid_frames) == 0:
        print("ERROR: No valid frames found!")
        return None, "no_valid_frames"

    # Create output directory
    out_data_dir = output_path / "data"
    out_data_dir.mkdir(parents=True, exist_ok=True)

    # Save inference index
    index_data = []
    for i, frame in enumerate(valid_frames):
        row = {
            "frame_id": i,
            "ego_idx": frame["ego_idx"],
            "ego_ts": frame["ego_ts"],
            "ego_ts_sec": frame["ego_ts"] / 1e6,
        }

        # Add camera frame indices
        for cam in CAMERAS:
            for j, (idx, ts, diff) in enumerate(frame["cam_frames"][cam]):
                row[f"{cam}_f{j}_idx"] = idx
                row[f"{cam}_f{j}_ts"] = ts
                row[f"{cam}_f{j}_diff_ms"] = diff

        index_data.append(row)

    index_df = pd.DataFrame(index_data)
    index_df.to_csv(out_data_dir / "inference_index_strict.csv", index=False)
    print(f"Saved inference index: {len(index_df)} frames")

    # Print statistics
    print("\n=== Validation Statistics ===")
    for cam in CAMERAS:
        for j in range(4):
            diffs = index_df[f"{cam}_f{j}_diff_ms"].values
            print(
                f"{cam} f{j}: mean_diff={diffs.mean():.2f}ms, max_diff={diffs.max():.2f}ms"
            )

    # Save egomotion data for valid frames
    ego_out_dir = out_data_dir / "egomotion"
    ego_out_dir.mkdir(exist_ok=True)

    for i, frame in enumerate(valid_frames):
        # History (16 steps)
        hist_data = ego_df.iloc[frame["history_indices"]][
            ["timestamp", "qx", "qy", "qz", "qw", "x", "y", "z", "vx", "vy", "vz"]
        ].values

        # Future (64 steps) - ground truth
        future_data = ego_df.iloc[frame["future_indices"]][
            ["timestamp", "x", "y", "z"]
        ].values

        # Save
        np.save(ego_out_dir / f"frame_{i:06d}_history.npy", hist_data)
        np.save(ego_out_dir / f"frame_{i:06d}_future_gt.npy", future_data)

    print(f"\nSaved {len(valid_frames)} egomotion files")

    # Step 5: GPU decode images
    print("\nStep 5: GPU解码视频帧...")
    import av
    from PIL import Image

    JPG_QUALITY = 95

    def decode_video_gpu(cam_name):
        """GPU解码单个相机视频"""
        try:
            video_path = f"{data_root}/camera/{clip_id}.{cam_name}.mp4"
            cam_img_dir = f"{output_path}/data/camera_images/{cam_name}"
            os.makedirs(cam_img_dir, exist_ok=True)

            container = av.open(video_path)
            stream = container.streams.video[0]

            # 尝试使用GPU解码
            try:
                stream.codec_context.codec = av.Codec("h264_cuvid", "r")
            except:
                pass

            frame_count = 0
            for packet in container.demux(stream):
                for frame in packet.decode():
                    img = Image.fromarray(frame.to_ndarray(format="rgb24"))
                    img.save(
                        f"{cam_img_dir}/{frame_count:06d}.jpg",
                        "JPEG",
                        quality=JPG_QUALITY,
                    )
                    frame_count += 1

            container.close()
            print(f"  {cam_name}: {frame_count} frames")
            return cam_name, frame_count
        except Exception as e:
            print(f"  错误 {cam_name}: {e}")
            return cam_name, 0

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(decode_video_gpu, CAMERAS))

    total_frames = sum(r[1] for r in results)
    print(f"  完成: 共 {total_frames} 帧")

    # Save metadata
    metadata = {
        "clip_id": clip_id,
        "num_valid_frames": len(valid_frames),
        "history_steps": HISTORY_STEPS,
        "future_steps": FUTURE_STEPS,
        "time_step": TIME_STEP,
        "max_image_diff_ms": MAX_IMAGE_DIFF_MS,
        "max_ego_diff_ms": MAX_EGO_DIFF_MS,
        "validation_method": "strict_no_interpolation",
    }

    with open(output_path / "preprocess_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Preprocessing complete!")
    print(f"Output directory: {output_path}")

    return output_path, "success"


def get_all_clips_in_chunk(data_root):
    """获取chunk中所有的clip ID列表"""
    labels_dir = Path(data_root) / "labels" / "egomotion"
    if not labels_dir.exists():
        print(f"ERROR: Labels directory not found: {labels_dir}")
        return []

    clips = []
    for f in labels_dir.glob("*.egomotion.parquet"):
        clip_id = f.stem.replace(".egomotion", "")
        clips.append(clip_id)

    return sorted(clips)


def process_single_clip_wrapper(args):
    """多进程处理的包装函数"""
    clip_id, data_root, output_root, force = args
    try:
        result, status = preprocess_clip_strict(
            clip_id, data_root, output_root, force=force
        )
        return clip_id, status, str(result) if result else None
    except Exception as e:
        import traceback

        print(f"❌ Error processing {clip_id}: {e}")
        traceback.print_exc()
        return clip_id, "error", str(e)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="V5 严格预处理 - 支持批量处理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 处理单个clip
  python preprocess_strict.py --clip <clip_id> --chunk 0
  
  # 处理多个clip（逗号分隔）
  python preprocess_strict.py --clip clip1,clip2,clip3 --chunk 0
  
  # 处理chunk0的所有clips
  python preprocess_strict.py --clip 0 --chunk 0
  
  # 并行处理（推荐用于大量clips）
  python preprocess_strict.py --clip 0 --chunk 0 --parallel 6
  
  # 跳过已处理的clips（批量处理优化）
  python preprocess_strict.py --clip 0 --chunk 0 --parallel 6 --skip-existing
  
  # 强制重新处理（覆盖已有结果）
  python preprocess_strict.py --clip 0 --chunk 0 --parallel 6 --force
        """,
    )
    parser.add_argument(
        "--clip",
        type=str,
        required=True,
        help='Clip ID(s). 支持: 单clip, 逗号分隔列表, 或 "0"处理所有clips',
    )
    parser.add_argument("--chunk", type=int, default=0, help="Chunk number")
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="并行处理进程数（默认1，推荐设为GPU数量）",
    )
    parser.add_argument(
        "--skip-existing", action="store_true", help="跳过已处理的clips"
    )
    parser.add_argument(
        "--force", action="store_true", help="强制覆盖已存在的预处理结果"
    )
    args = parser.parse_args()

    data_root = f"/data01/vla/data/data_sample_chunk{args.chunk}"
    output_root = f"/data01/vla/data/data_sample_chunk{args.chunk}/infer"

    # 确定要处理的clips列表
    if args.clip == "0":
        # 自动获取所有clips
        print(f"🔍 发现模式：将处理 chunk{args.chunk} 的所有 clips...")
        clip_list = get_all_clips_in_chunk(data_root)
        if not clip_list:
            print("❌ 没有找到任何 clips！")
            sys.exit(1)
        print(f"📊 找到 {len(clip_list)} 个 clips")
    elif "," in args.clip:
        # 逗号分隔的列表
        clip_list = [c.strip() for c in args.clip.split(",")]
        print(f"📝 处理指定的 {len(clip_list)} 个 clips: {clip_list[:5]}...")
    else:
        # 单个clip
        clip_list = [args.clip]
        print(f"📝 处理单个 clip: {args.clip}")

    # 检查 force 和 skip-existing 是否冲突
    if args.force and args.skip_existing:
        print("❌ 错误：--force 和 --skip-existing 不能同时使用！")
        sys.exit(1)

    # 强制覆盖模式
    if args.force:
        print(f"⚠️  强制覆盖模式：将删除并重新处理所有 {len(clip_list)} 个 clips")

    # 检查是否需要跳过已处理的
    if args.skip_existing:
        clips_to_process = []
        for clip in clip_list:
            output_path = Path(output_root) / clip
            if (output_path / "data" / "inference_index_strict.csv").exists():
                print(f"⏭️  {clip}: 已处理，跳过")
            else:
                clips_to_process.append(clip)
        clip_list = clips_to_process
        print(f"📊 实际需要处理: {len(clip_list)} 个 clips")

    if len(clip_list) == 0:
        print("✅ 所有 clips 都已处理完成！")
        sys.exit(0)

    # 处理clips
    print(f"\n{'=' * 60}")
    print(f"开始预处理: {len(clip_list)} clips, chunk={args.chunk}")
    print(f"并行进程数: {args.parallel}")
    print(f"{'=' * 60}\n")

    results_summary = {
        "total": len(clip_list),
        "success": 0,
        "skipped": 0,
        "failed": 0,
        "errors": [],
    }

    if args.parallel > 1 and len(clip_list) > 1:
        # 并行处理
        print(f"🚀 使用 {args.parallel} 个进程并行处理...")

        # 准备参数
        process_args = [
            (clip, data_root, output_root, args.force) for clip in clip_list
        ]

        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            results = list(executor.map(process_single_clip_wrapper, process_args))

        # 汇总结果
        for clip_id, status, result in results:
            if status == "success":
                results_summary["success"] += 1
                print(f"✅ {clip_id}: 成功")
            elif status == "skipped":
                results_summary["skipped"] += 1
                print(f"⏭️  {clip_id}: 已跳过")
            else:
                results_summary["failed"] += 1
                results_summary["errors"].append(f"{clip_id}: {status} - {result}")
                print(f"❌ {clip_id}: 失败 ({status})")
    else:
        # 串行处理
        for i, clip_id in enumerate(clip_list, 1):
            print(f"\n{'=' * 60}")
            print(f"[{i}/{len(clip_list)}] Processing: {clip_id}")
            print(f"{'=' * 60}")

            clip_id, status, result = process_single_clip_wrapper(
                (clip_id, data_root, output_root, args.force)
            )

            if status == "success":
                results_summary["success"] += 1
                print(f"✅ 成功!")
            elif status == "skipped":
                results_summary["skipped"] += 1
                print(f"⏭️  已跳过")
            else:
                results_summary["failed"] += 1
                results_summary["errors"].append(f"{clip_id}: {status}")
                print(f"❌ 失败: {status}")

    # 打印汇总
    print(f"\n{'=' * 60}")
    print(f"📊 处理完成汇总")
    print(f"{'=' * 60}")
    print(f"总计: {results_summary['total']}")
    print(f"✅ 成功: {results_summary['success']}")
    print(f"⏭️  跳过: {results_summary['skipped']}")
    print(f"❌ 失败: {results_summary['failed']}")

    if results_summary["errors"]:
        print(f"\n❌ 错误详情:")
        for error in results_summary["errors"][:10]:  # 只显示前10个
            print(f"  - {error}")
        if len(results_summary["errors"]) > 10:
            print(f"  ... 还有 {len(results_summary['errors']) - 10} 个错误")

    print(f"\n✨ 全部完成！")

    # 退出码
    sys.exit(0 if results_summary["failed"] == 0 else 1)
