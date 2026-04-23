#!/usr/bin/env python3
"""
清理每个 clip 中未被 inference_index_strict.csv 引用的图片，释放存储空间。
支持每隔 step 行采样后删除未引用图片。

用法:
  python infer_data_reduce.py --chunks 1 --clips all --parallel 4
  python infer_data_reduce.py --chunks 1,2 --clips all --parallel 4 --step 10
  python infer_data_reduce.py --chunks 1 --clips uuid1,uuid2 --parallel 2 --step 5

逻辑:
  1. 备份 inference_index_strict.csv → inference_index_origin.csv
  2. 按 step 采样：保留行号 0, step, 2*step, ... 的行，写回 inference_index_strict.csv
     （frame_id 列保持原值，不重新编号）
  3. 从采样后的 CSV 提取所有被引用的图片索引（4相机 × 4帧 = 16图/帧）
  4. 去重得到 referenced_imgs
  5. 遍历 camera_images/{cam}/ 下的所有实际 JPG 文件
  6. 删除不在 referenced_imgs 中的文件
"""

import argparse
import os
import shutil
import sys
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

CAMERAS = [
    "camera_cross_left_120fov",
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_front_tele_30fov",
]


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_all_clips_in_chunk(data_root):
    labels_dir = Path(data_root) / "labels" / "egomotion"
    if not labels_dir.exists():
        return []
    return sorted(f.stem.replace(".egomotion", "") for f in labels_dir.glob("*.egomotion.parquet"))


def build_clip_frame(chunks_str, clips_str, base_dir):
    chunk_ids = sorted(set(int(p.strip()) for p in chunks_str.split(",") if p.strip().isdigit()))
    clip_rows = []

    for cid in chunk_ids:
        data_root = f"{base_dir}/data_sample_chunk{cid}"
        if clips_str.lower() == "all":
            clips = get_all_clips_in_chunk(data_root)
        elif "," in clips_str:
            clips = [c.strip() for c in clips_str.split(",")]
        else:
            clips = [clips_str.strip()]
        for clip_id in clips:
            clip_rows.append({"chunk_id": cid, "clip_id": clip_id})

    if not clip_rows:
        return None
    df = pd.DataFrame(clip_rows)
    print(f"\n📋 clip_frame ({len(df)} rows):")
    print(df.to_string(index=False))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# per-clip reduction
# ─────────────────────────────────────────────────────────────────────────────

def reduce_clip(chunk_id, clip_id, base_dir, step):
    """
    对单个 clip:
    1. 备份 CSV
    2. 按 step 采样 CSV（保留行 0, step, 2*step, ...）
    3. 基于采样后 CSV 构建引用集合
    4. 删除未被引用的 JPG 文件

    Returns: (chunk_id, clip_id, status, deleted_count, saved_mb,
              original_frames, sampled_frames, message)
    """
    infer_dir = Path(base_dir) / f"data_sample_chunk{chunk_id}" / "infer" / clip_id / "data"
    csv_path = infer_dir / "inference_index_strict.csv"
    origin_path = infer_dir / "inference_index_origin.csv"
    cam_img_root = infer_dir / "camera_images"

    if not csv_path.exists():
        return chunk_id, clip_id, "skip", 0, 0.0, 0, 0, f"CSV 不存在: {csv_path}"

    # ── Step 1: 读取原始 CSV ──
    try:
        df_origin = pd.read_csv(csv_path)
    except Exception as e:
        return chunk_id, clip_id, "error", 0, 0.0, 0, 0, f"CSV 读取失败: {e}"

    original_frames = len(df_origin)

    # ── Step 2: 备份原始 CSV ──
    shutil.copy2(csv_path, origin_path)

    # ── Step 3: 按 step 采样（保留行 0, step, 2*step, ...）──
    sampled_df = df_origin.iloc[::step].reset_index(drop=True)
    sampled_frames = len(sampled_df)

    # 写回 inference_index_strict.csv（frame_id 列保持原值，不重新编号）
    sampled_df.to_csv(csv_path, index=False)

    # ── Step 4: 构建引用集合 ──
    referenced = {}  # cam -> set of referenced frame indices (int)
    for cam in CAMERAS:
        idx_set = set()
        for j in range(4):
            col = f"{cam}_f{j}_idx"
            if col in sampled_df.columns:
                idx_set.update(int(x) for x in sampled_df[col].tolist() if pd.notna(x))
        referenced[cam] = idx_set

    # ── Step 5: 删除未被引用的 JPG ──
    deleted_count = 0
    saved_bytes = 0

    for cam in CAMERAS:
        cam_dir = cam_img_root / cam
        if not cam_dir.exists():
            continue

        for img_file in cam_dir.iterdir():
            if img_file.suffix.lower() not in (".jpg", ".jpeg"):
                continue
            try:
                frame_idx = int(img_file.stem)  # filename: 000000.jpg -> frame index
            except ValueError:
                continue

            if frame_idx not in referenced[cam]:
                file_size = img_file.stat().st_size
                img_file.unlink()
                deleted_count += 1
                saved_bytes += file_size

    saved_mb = saved_bytes / (1024 * 1024)
    msg = (f"原帧数={original_frames}, 采样后={sampled_frames}, "
           f"删图={deleted_count}, 释放={saved_mb:.1f}MB")
    return chunk_id, clip_id, "success", deleted_count, saved_mb, original_frames, sampled_frames, msg


def process_one(args):
    """多进程 wrapper"""
    chunk_id, clip_id, base_dir, step = args
    return reduce_clip(chunk_id, clip_id, base_dir, step)


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="清理未被引用的图片，释放存储空间（支持 step 采样）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python infer_data_reduce.py --chunks 1 --clips all --parallel 4
  python infer_data_reduce.py --chunks 1,2 --clips all --parallel 6 --step 10
  python infer_data_reduce.py --chunks 1 --clips uuid1,uuid2 --parallel 2 --step 5
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
        default="all",
        help="Clip ID(s): all（默认，chunk下全部）, 逗号分隔列表, 或单个clip ID",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="/data01/mikelee/data",
        help="数据根目录 (default: /data01/mikelee/data)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="并行进程数",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=10,
        help="每隔 step 行保留一行（default: 10，即保留 0,9,18,... 行）",
    )
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print(f"🎯 构建 clip_frame")
    print(f"{'=' * 60}")
    clip_frame = build_clip_frame(args.chunks, args.clips, args.base_dir)
    if clip_frame is None or len(clip_frame) == 0:
        print("❌ 没有找到任何 (chunk, clip) 组合！")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"🚀 开始清理: {len(clip_frame)} clips, 并行数: {args.parallel}, step={args.step}")
    print(f"{'=' * 60}\n")

    task_args = [
        (int(row["chunk_id"]), str(row["clip_id"]), args.base_dir, args.step)
        for _, row in clip_frame.iterrows()
    ]

    total_deleted = 0
    total_saved_mb = 0.0
    total_original_frames = 0
    total_sampled_frames = 0
    errors = []

    if args.parallel > 1 and len(task_args) > 1:
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = {executor.submit(process_one, arg): arg for arg in task_args}
            for future in as_completed(futures):
                cid, clip_id, status, deleted, saved, orig_f, sample_f, msg = future.result()
                if status == "success":
                    total_deleted += deleted
                    total_saved_mb += saved
                    total_original_frames += orig_f
                    total_sampled_frames += sample_f
                    print(f"  ✅ chunk{cid} {clip_id}: {msg}")
                elif status == "skip":
                    print(f"  ⏭️  chunk{cid} {clip_id}: {msg}")
                else:
                    errors.append(f"chunk{cid} {clip_id}: {msg}")
                    print(f"  ❌ chunk{cid} {clip_id}: {msg}")
    else:
        for arg in task_args:
            cid, clip_id, status, deleted, saved, orig_f, sample_f, msg = process_one(arg)
            if status == "success":
                total_deleted += deleted
                total_saved_mb += saved
                total_original_frames += orig_f
                total_sampled_frames += sample_f
                print(f"  ✅ chunk{cid} {clip_id}: {msg}")
            elif status == "skip":
                print(f"  ⏭️  chunk{cid} {clip_id}: {msg}")
            else:
                errors.append(f"chunk{cid} {clip_id}: {msg}")
                print(f"  ❌ chunk{cid} {clip_id}: {msg}")

    print(f"\n{'=' * 60}")
    print(f"📊 清理完成")
    print(f"{'=' * 60}")
    print(f"  总 clips: {len(clip_frame)}")
    print(f"  原总帧数: {total_original_frames:,}")
    print(f"  采样后帧数: {total_sampled_frames:,} (step={args.step})")
    print(f"  删除图片总数: {total_deleted}")
    print(f"  释放空间: {total_saved_mb:.1f} MB ({total_saved_mb/1024:.2f} GB)")
    if errors:
        print(f"  错误: {len(errors)} 个")
        for e in errors[:10]:
            print(f"    - {e}")

    print(f"\n✨ 全部完成！")
    sys.exit(0 if not errors else 1)


if __name__ == "__main__":
    main()
