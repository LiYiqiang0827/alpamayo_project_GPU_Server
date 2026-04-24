#!/usr/bin/env python3
"""
Preprocess clips with strict timestamp alignment (no interpolation version)
支持多 chunk、多 clip 并行处理。

用法:
  python infer_data_gen.py --chunks 1 --clips all --parallel 4
  python infer_data_gen.py --chunks 1,2 --clips all --parallel 4
  python infer_data_gen.py --chunks 1 --clips uuid1,uuid2 --parallel 4
  python infer_data_gen.py --chunks 1 --clips uuid1 --parallel 1

新增功能:
  --pre_resize  解码后自动 resize 成 576x320 并保存为 _small.jpg (默认开启)
  --save_numpy  调试用：额外保存解码后的原始 numpy 数据 (默认关闭)
  --source_dir   源数据根目录 (默认 /gpfs-data/mikelee/data)
  --output_dir   输出根目录 (默认 /data01/mikelee/data)
"""

import os
import sys
import json
import argparse
import math
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

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
MAX_EGO_DIFF_MS = 30  # Maximum allowed time difference for egomotion validation

# Resize configuration (与 infer_resize_image.py 保持一致)
MIN_PIXELS = 163840
MAX_PIXELS = 196608
PATCH_SIZE = 16
MERGE_SIZE = 2
FACTOR = PATCH_SIZE * MERGE_SIZE  # 32
JPG_QUALITY_RESIZE = 95  # pre_resize 输出的 JPEG 质量


def smart_resize(height, width, factor=FACTOR,
                  min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS):
    """复现 Qwen2VLImageProcessor 的 smart_resize 算法"""
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"Aspect ratio too extreme")

    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    return h_bar, w_bar


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def get_all_clips_in_chunk(data_root):
    """获取 chunk 中所有的 clip ID 列表。"""
    labels_dir = Path(data_root) / "labels" / "egomotion"
    if not labels_dir.exists():
        return []
    clips = []
    for f in labels_dir.glob("*.egomotion.parquet"):
        clip_id = f.stem.replace(".egomotion", "")
        clips.append(clip_id)
    return sorted(clips)


def build_clip_frame(chunks_str, clips_str, source_dir):
    """
    根据 chunks 和 clips 构建 clip_frame DataFrame。
    """
    chunk_ids = []
    for part in chunks_str.split(","):
        part = part.strip()
        if not part.isdigit():
            raise ValueError(f"Invalid chunk ID: {part}")
        chunk_ids.append(int(part))
    chunk_ids = sorted(set(chunk_ids))

    clips_str = clips_str.strip()
    clip_frames = []

    for chunk_id in chunk_ids:
        data_root = f"{source_dir}/data_sample_chunk{chunk_id}"

        if clips_str.lower() == "all":
            clips_in_chunk = get_all_clips_in_chunk(data_root)
            if not clips_in_chunk:
                print(f"⚠️  chunk {chunk_id}: 目录下没有找到 clips: {data_root}")
                continue
            print(f"  chunk {chunk_id}: 发现 {len(clips_in_chunk)} 个 clips")
        elif "," in clips_str:
            clips_in_chunk = [c.strip() for c in clips_str.split(",")]
        else:
            clips_in_chunk = [clips_str]

        for clip_id in clips_in_chunk:
            clip_frames.append({"chunk_id": chunk_id, "clip_id": clip_id})

    if not clip_frames:
        print("❌ 没有找到任何 (chunk, clip) 组合！")
        return None

    df = pd.DataFrame(clip_frames)
    print(f"\n📋 clip_frame ({len(df)} rows):")
    print(df.to_string(index=False))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Core preprocessing (per-clip)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_clip_strict(chunk_id, clip_id, source_dir, output_dir,
                           pre_resize=True, save_numpy=False, jpg_quality=95, force=False):
    """
    Preprocess a single clip with strict timestamp alignment.
    Returns: (chunk_id, clip_id, status, message)
    """
    data_path = Path(source_dir) / f"data_sample_chunk{chunk_id}"
    output_path = Path(output_dir) / f"data_sample_chunk{chunk_id}" / "infer" / clip_id

    # Check if already processed
    if (output_path / "data" / "inference_index_strict.csv").exists():
        if force:
            import shutil
            shutil.rmtree(output_path, ignore_errors=True)
        else:
            return chunk_id, clip_id, "skipped", str(output_path)

    # Load egomotion
    ego_path = data_path / "labels" / "egomotion" / f"{clip_id}.egomotion.parquet"
    if not ego_path.exists():
        return chunk_id, clip_id, "error", f"Egomotion not found: {ego_path}"

    try:
        ego_df = pd.read_parquet(ego_path)
    except Exception as e:
        return chunk_id, clip_id, "error", f"Failed to read ego parquet: {e}"

    # Load camera timestamps
    cam_timestamps = {}
    for cam in CAMERAS:
        ts_path = data_path / "camera" / cam / f"{clip_id}.{cam}.timestamps.parquet"
        if ts_path.exists():
            cam_timestamps[cam] = pd.read_parquet(ts_path)
        else:
            return chunk_id, clip_id, "error", f"Timestamps not found: {ts_path}"

    # Find common time range
    cam_ranges = {
        cam: (cam_timestamps[cam]["timestamp"].iloc[0], cam_timestamps[cam]["timestamp"].iloc[-1])
        for cam in CAMERAS
    }
    common_start = max(cam_ranges[c][0] for c in CAMERAS)
    common_end = min(cam_ranges[c][1] for c in CAMERAS)

    # Filter egomotion to common range
    ego_valid = ego_df[
        (ego_df["timestamp"] >= common_start) & (ego_df["timestamp"] <= common_end)
    ].copy()

    # Build valid inference index
    valid_frames = []
    stats = {
        "total_checked": 0,
        "filtered_by_history_time": 0,
        "filtered_by_future_time": 0,
        "filtered_by_camera": 0,
    }

    for idx in range(len(ego_valid)):
        ego_idx = ego_valid.index[idx]
        ego_ts = ego_valid.iloc[idx]["timestamp"]
        stats["total_checked"] += 1

        # --- Validate Egomotion History ---
        history_indices = []
        history_valid = True
        for i in range(0, HISTORY_STEPS):
            target_ts = ego_ts - i * TIME_STEP * 1e6
            idx_loc = (ego_df["timestamp"] - target_ts).abs().idxmin()
            closest_ts = ego_df.loc[idx_loc, "timestamp"]
            diff_ms = abs(target_ts - closest_ts) / 1000.0
            if diff_ms > MAX_EGO_DIFF_MS:
                history_valid = False
                break
            history_indices.append(idx_loc)
        history_indices.reverse()

        if not history_valid:
            stats["filtered_by_history_time"] += 1
            continue

        # --- Validate Egomotion Future ---
        future_indices = []
        future_valid = True
        for i in range(1, FUTURE_STEPS + 1):
            target_ts = ego_ts + i * TIME_STEP * 1e6
            idx_loc = (ego_df["timestamp"] - target_ts).abs().idxmin()
            closest_ts = ego_df.loc[idx_loc, "timestamp"]
            diff_ms = abs(target_ts - closest_ts) / 1000.0
            if diff_ms > MAX_EGO_DIFF_MS:
                future_valid = False
                break
            future_indices.append(idx_loc)

        if not future_valid:
            stats["filtered_by_future_time"] += 1
            continue

        # --- Validate Camera Frames ---
        cam_valid = True
        cam_frames = {cam: [] for cam in CAMERAS}

        for i, offset_ms in enumerate(IMG_TIME_OFFSETS_MS):
            target_ts = ego_ts - offset_ms * 1000
            for cam in CAMERAS:
                cam_ts_df = cam_timestamps[cam]
                if i == 3:
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
                else:
                    idx_loc = (cam_ts_df["timestamp"] - target_ts).abs().idxmin()
                    closest_ts = cam_ts_df.loc[idx_loc, "timestamp"]
                    diff_ms = abs(target_ts - closest_ts) / 1000.0
                    if diff_ms > MAX_IMAGE_DIFF_MS:
                        cam_valid = False
                        break
                    cam_frames[cam].append((idx_loc, closest_ts, diff_ms))

            if not cam_valid:
                break

        if not cam_valid:
            stats["filtered_by_camera"] += 1
            continue

        valid_frames.append({
            "ego_idx": ego_idx,
            "ego_ts": ego_ts,
            "history_indices": history_indices,
            "future_indices": future_indices,
            "cam_frames": cam_frames,
        })

    if len(valid_frames) == 0:
        return chunk_id, clip_id, "error", "No valid frames found"

    # Create output directory
    out_data_dir = output_path / "data"
    out_data_dir.mkdir(parents=True, exist_ok=True)

    # Save inference index CSV
    index_data = []
    for i, frame in enumerate(valid_frames):
        row = {
            "frame_id": i,
            "ego_idx": frame["ego_idx"],
            "ego_ts": frame["ego_ts"],
            "ego_ts_sec": frame["ego_ts"] / 1e6,
        }
        for cam in CAMERAS:
            for j, (idx, ts, diff) in enumerate(frame["cam_frames"][cam]):
                row[f"{cam}_f{j}_idx"] = idx
                row[f"{cam}_f{j}_ts"] = ts
                row[f"{cam}_f{j}_diff_ms"] = diff
        index_data.append(row)

    index_df = pd.DataFrame(index_data)
    index_df.to_csv(out_data_dir / "inference_index_strict.csv", index=False)

    # Save egomotion data
    ego_out_dir = out_data_dir / "egomotion"
    ego_out_dir.mkdir(exist_ok=True)
    for i, frame in enumerate(valid_frames):
        hist_data = ego_df.iloc[frame["history_indices"]][
            ["timestamp", "qx", "qy", "qz", "qw", "x", "y", "z", "vx", "vy", "vz"]
        ].values
        future_data = ego_df.iloc[frame["future_indices"]][
            ["timestamp", "x", "y", "z"]
        ].values
        np.save(ego_out_dir / f"frame_{i:06d}_history.npy", hist_data)
        np.save(ego_out_dir / f"frame_{i:06d}_future_gt.npy", future_data)

    # Decode camera videos
    import av
    from PIL import Image

    def decode_video_with_resize(cam_name):
        try:
            video_path = f"{data_path}/camera/{cam_name}/{clip_id}.{cam_name}.mp4"
            cam_img_dir = f"{output_path}/data/camera_images/{cam_name}"
            os.makedirs(cam_img_dir, exist_ok=True)

            container = av.open(video_path)
            stream = container.streams.video[0]
            try:
                stream.codec_context.codec = av.Codec("h264_cuvid", "r")
            except Exception:
                pass

            frame_count = 0
            for packet in container.demux(stream):
                for frame in packet.decode():
                    img = Image.fromarray(frame.to_ndarray(format="rgb24"))

                    if save_numpy:
                        # 保存解码后的原始 numpy (H, W, C) uint8
                        npy_data = np.array(img, dtype=np.uint8)
                        np.save(f"{cam_img_dir}/{frame_count:06d}.npy", npy_data)

                    if pre_resize:
                        # smart_resize
                        orig_w, orig_h = img.size
                        target_h, target_w = smart_resize(orig_h, orig_w)
                        img_resized = img.resize((target_w, target_h), Image.BICUBIC)
                        img_resized.save(
                            f"{cam_img_dir}/{frame_count:06d}_small.jpg",
                            "JPEG",
                            quality=jpg_quality
                        )
                        img_resized.close()

                    img.close()
                    frame_count += 1

            container.close()
            return cam_name, frame_count
        except Exception as e:
            return cam_name, 0

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(decode_video_with_resize, CAMERAS))

    # Save metadata
    metadata = {
        "clip_id": clip_id,
        "chunk_id": chunk_id,
        "num_valid_frames": len(valid_frames),
        "history_steps": HISTORY_STEPS,
        "future_steps": FUTURE_STEPS,
        "time_step": TIME_STEP,
        "max_image_diff_ms": MAX_IMAGE_DIFF_MS,
        "max_ego_diff_ms": MAX_EGO_DIFF_MS,
        "validation_method": "strict_no_interpolation",
        "pre_resize": pre_resize,
        "save_numpy": save_numpy,
        "resize_jpeg_quality": jpg_quality if pre_resize else None,
        "camera_frames_decoded": dict(results),
        "filter_stats": {
            "total_checked": stats["total_checked"],
            "filtered_by_history_time": stats["filtered_by_history_time"],
            "filtered_by_future_time": stats["filtered_by_future_time"],
            "filtered_by_camera": stats["filtered_by_camera"],
        },
    }
    with open(output_path / "preprocess_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return chunk_id, clip_id, "success", str(output_path)


def process_one_clip(args):
    """多进程 wrapper"""
    chunk_id, clip_id, source_dir, output_dir, pre_resize, save_numpy, jpg_quality, force = args
    return preprocess_clip_strict(
        chunk_id, clip_id, source_dir, output_dir,
        pre_resize=pre_resize, save_numpy=save_numpy, jpg_quality=jpg_quality, force=force
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="严格预处理 - 多 chunk 多 clip 并行处理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # chunk1 的所有 clips，4进程并行，预 resize 开启
  python infer_data_gen.py --chunks 1 --clips all --parallel 4

  # 多个 chunk 的所有 clips
  python infer_data_gen.py --chunks 1,2 --clips all --parallel 4

  # 指定 clip 列表
  python infer_data_gen.py --chunks 1 --clips uuid1,uuid2 --parallel 4

  # 单个 clip
  python infer_data_gen.py --chunks 1 --clips uuid1 --parallel 1

  # 跳过已处理
  python infer_data_gen.py --chunks 1 --clips all --parallel 4 --skip-existing

  # 强制覆盖
  python infer_data_gen.py --chunks 1 --clips all --parallel 4 --force

  # 关闭 pre_resize（输出原始分辨率 jpg）
  python infer_data_gen.py --chunks 1 --clips all --parallel 4 --no-pre-resize

  # 开启 save_numpy（调试用，保存原始解码 numpy）
  python infer_data_gen.py --chunks 1 --clips all --parallel 4 --save-numpy

  # 自定义 JPEG 质量
  python infer_data_gen.py --chunks 1 --clips all --parallel 4 --jpg-quality 95
        """,
    )
    parser.add_argument(
        "--chunks",
        type=str,
        required=True,
        help="Chunk ID(s)，单个或逗号分隔，如: 1 或 1,2,3",
    )
    parser.add_argument(
        "--clips",
        type=str,
        required=True,
        help="Clip ID(s)，支持: all（该chunk下全部）, 逗号分隔列表, 或单个clip ID",
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default="/gpfs-data/mikelee/data",
        help="源数据根目录 (default: /gpfs-data/mikelee/data)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/data01/mikelee/data",
        help="输出根目录 (default: /data01/mikelee/data)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="并行进程数（每个进程处理一个 clip）",
    )
    parser.add_argument(
        "--pre-resize",
        action="store_true",
        default=True,
        help="解码后 resize 成 576x320 并保存为 _small.jpg (默认开启)",
    )
    parser.add_argument(
        "--no-pre-resize",
        action="store_true",
        help="关闭 pre_resize（输出原始分辨率 jpg）",
    )
    parser.add_argument(
        "--save-numpy",
        action="store_true",
        default=False,
        help="调试用：额外保存解码后的原始 numpy 数据 (默认关闭)",
    )
    parser.add_argument(
        "--jpg-quality",
        type=int,
        default=95,
        help="pre_resize 输出的 JPEG 质量 (默认: 95)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="跳过已处理的 clips",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制覆盖已存在的预处理结果",
    )
    args = parser.parse_args()

    # --no-pre-resize 覆盖 --pre-resize
    pre_resize = not args.no_pre_resize

    if args.force and args.skip_existing:
        print("❌ --force 和 --skip-existing 不能同时使用！")
        sys.exit(1)

    # ── 构建 clip_frame ──
    print(f"\n{'=' * 60}")
    print(f"🎯 构建 clip_frame")
    print(f"{'=' * 60}")
    clip_frame = build_clip_frame(args.chunks, args.clips, args.source_dir)
    if clip_frame is None or len(clip_frame) == 0:
        sys.exit(1)

    # ── skip-existing 过滤 ──
    if args.skip_existing:
        def is_done(row):
            out_path = Path(f"{args.output_dir}/data_sample_chunk{row['chunk_id']}/infer/{row['clip_id']}/data/inference_index_strict.csv")
            return out_path.exists()

        mask = clip_frame.apply(is_done, axis=1)
        skipped_df = clip_frame[mask]
        to_process_df = clip_frame[~mask]

        if len(skipped_df) > 0:
            print(f"\n⏭️  跳过 {len(skipped_df)} 个已处理的 clips:")
            for _, row in skipped_df.iterrows():
                print(f"     chunk {row['chunk_id']} / {row['clip_id']}")
        clip_frame = to_process_df

    if len(clip_frame) == 0:
        print("\n✅ 所有 clips 都已处理完成！")
        sys.exit(0)

    print(f"\n{'=' * 60}")
    print(f"🚀 开始处理: {len(clip_frame)} clips, 并行数: {args.parallel}")
    print(f"   源数据目录: {args.source_dir}")
    print(f"   输出目录:   {args.output_dir}")
    print(f"   pre_resize: {pre_resize}")
    print(f"   jpg_quality: {args.jpg_quality}")
    print(f"   save_numpy: {args.save_numpy}")
    print(f"{'=' * 60}\n")

    # ── 准备并行任务参数 ──
    task_args = [
        (int(row["chunk_id"]), str(row["clip_id"]),
         args.source_dir, args.output_dir,
         pre_resize, args.save_numpy, args.jpg_quality, args.force)
        for _, row in clip_frame.iterrows()
    ]

    results = {"success": 0, "skipped": 0, "error": 0, "errors": []}

    if args.parallel > 1 and len(task_args) > 1:
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = {executor.submit(process_one_clip, arg): arg for arg in task_args}
            for future in as_completed(futures):
                cid, clip_id, status, msg = future.result()
                if status == "success":
                    results["success"] += 1
                    print(f"  ✅ chunk{cid} {clip_id}: {msg}")
                elif status == "skipped":
                    results["skipped"] += 1
                    print(f"  ⏭️  chunk{cid} {clip_id}: 已跳过")
                else:
                    results["error"] += 1
                    results["errors"].append(f"chunk{cid} {clip_id}: {msg}")
                    print(f"  ❌ chunk{cid} {clip_id}: {msg}")
    else:
        for arg in task_args:
            cid, clip_id, status, msg = process_one_clip(arg)
            if status == "success":
                results["success"] += 1
                print(f"  ✅ chunk{cid} {clip_id}: {msg}")
            elif status == "skipped":
                results["skipped"] += 1
                print(f"  ⏭️  chunk{cid} {clip_id}: 已跳过")
            else:
                results["error"] += 1
                results["errors"].append(f"chunk{cid} {clip_id}: {msg}")
                print(f"  ❌ chunk{cid} {clip_id}: {msg}")

    # ── 汇总 ──
    print(f"\n{'=' * 60}")
    print(f"📊 处理完成")
    print(f"{'=' * 60}")
    print(f"  总计: {len(clip_frame)}")
    print(f"  ✅ 成功: {results['success']}")
    print(f"  ⏭️  跳过: {results['skipped']}")
    print(f"  ❌ 失败: {results['error']}")
    if results["errors"]:
        print(f"\n  错误详情 (前10):")
        for e in results["errors"][:10]:
            print(f"    - {e}")
        if len(results["errors"]) > 10:
            print(f"    ... 还有 {len(results['errors']) - 10} 个")

    print(f"\n✨ 全部完成！")
    sys.exit(0 if results["error"] == 0 else 1)


if __name__ == "__main__":
    main()
