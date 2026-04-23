#!/usr/bin/env python3
"""
解压 PhysicalAI chunk 数据中的 camera mp4 和 labels parquet 文件。
解压完成后验证完整性，完整则删除已解压的 zip 文件。

用法:
    python unzip_data_files.py --chunks 6
    python unzip_data_files.py --chunks 6,7,8
    python unzip_data_files.py --chunks 9 --base-dir /data01/mikelee/data
    python unzip_data_files.py --chunks 9 --delete  # 解压验证后删除zip

处理范围:
    - camera/每个摄像头/*.zip  → 解压到 camera/摄像头/ (mp4)
    - labels/egomotion/*.zip  → 解压到 labels/egomotion/ (parquet)
      (不包括 labels/egomotion.offline/)
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

CHUNK_DIR_TEMPLATE = "data_sample_chunk{id}"  # 如 data_sample_chunk1, data_sample_chunk10
BASE_DIR = "/data01/mikelee/data"


def parse_chunks(chunks_str: str):
    """解析 '6,7,8' 或 '9' 格式的 chunk ID。"""
    chunk_ids = []
    for part in chunks_str.split(","):
        part = part.strip()
        if not part.isdigit():
            raise ValueError(f"Invalid chunk ID: {part}")
        chunk_ids.append(int(part))
    return sorted(set(chunk_ids))


def get_camera_zips(chunk_base: Path):
    """获取 camera 目录下的所有 zip 文件路径。"""
    camera_dir = chunk_base / "camera"
    if not camera_dir.exists():
        return []
    zips = []
    for cam_dir in camera_dir.iterdir():
        if cam_dir.is_dir():
            zips.extend(cam_dir.glob("*.zip"))
    return zips


def get_label_zips(chunk_base: Path):
    """获取 labels/egomotion 目录下的 zip 文件路径（不包括 offline）。"""
    egomotion_dir = chunk_base / "labels" / "egomotion"
    if not egomotion_dir.exists():
        return []
    return list(egomotion_dir.glob("*.zip"))


def unzip_file(zip_path: Path) -> tuple[str, bool, str]:
    """
    解压单个 zip 文件到 zip 所在目录。
    Returns: (zip_path_str, success, message)
    """
    target_dir = zip_path.parent
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(target_dir)
        return (str(zip_path), True, f"OK → {target_dir}")
    except zipfile.BadZipFile:
        return (str(zip_path), False, "Bad zip file")
    except Exception as e:
        return (str(zip_path), False, f"Error: {e}")


def verify_extraction(zip_path: Path) -> tuple[bool, str]:
    """
    验证解压是否完整。
    检查解压出来的目录里是否包含预期的文件。
    camera zip → 应有 .mp4 文件
    egomotion zip → 应有 .parquet 文件
    Returns: (complete, message)
    """
    target_dir = zip_path.parent
    zip_name = zip_path.stem  # 去掉 .zip

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            archived_names = zf.namelist()
    except Exception as e:
        return False, f"无法读取zip内容: {e}"

    if not archived_names:
        return False, "zip 内文件列表为空"

    # 取第一个文件判断类型（路径格式: uuid.camera_type.xxx 或 uuid.egomotion.xxx）
    first_file = archived_names[0]
    is_camera = ".camera_" in first_file or any(
        f".{cam}." in first_file
        for cam in ["front_wide", "front_tele", "cross_left", "cross_right"]
    )
    is_egomotion = ".egomotion." in first_file

    if is_camera:
        # camera zip → 检查是否有 mp4 文件
        expected_ext = ".mp4"
        actual_files = list(target_dir.glob(f"*{expected_ext}"))
        if len(actual_files) > 0:
            return True, f"验证通过 ({len(actual_files)} 个 mp4)"
        else:
            return False, f"未找到 mp4 文件"
    elif is_egomotion:
        # egomotion zip → 检查是否有 parquet 文件
        expected_ext = ".parquet"
        actual_files = list(target_dir.glob(f"*{expected_ext}"))
        if len(actual_files) > 0:
            return True, f"验证通过 ({len(actual_files)} 个 parquet)"
        else:
            return False, f"未找到 parquet 文件"
    else:
        # 兜底：检查解压出的文件数是否与 zip 内一致
        actual_count = sum(1 for _ in target_dir.iterdir())
        archived_count = len(archived_names)
        if actual_count >= archived_count:
            return True, f"验证通过 ({actual_count} 个文件)"
        else:
            return False, f"文件数不足: zip内{archived_count}个, 实际{actual_count}个"


def delete_zip(zip_path: Path) -> tuple[bool, str]:
    """删除指定的 zip 文件。"""
    try:
        zip_path.unlink()
        return True, f"已删除: {zip_path.name}"
    except Exception as e:
        return False, f"删除失败: {e}"


def process_chunk(chunk_id: int, base_dir: str, max_workers: int = 4, delete_after_verify: bool = False):
    """处理单个 chunk 的所有解压任务。"""
    chunk_name = CHUNK_DIR_TEMPLATE.format(id=chunk_id)
    chunk_base = Path(base_dir) / chunk_name

    if not chunk_base.exists():
        print(f"[chunk {chunk_id}] ❌ 目录不存在: {chunk_base}")
        return 0, 0, 0

    camera_zips = get_camera_zips(chunk_base)
    label_zips = get_label_zips(chunk_base)
    all_zips = camera_zips + label_zips

    if not all_zips:
        print(f"[chunk {chunk_id}] ⚠️  没有找到需要解压的 zip 文件")
        return 0, 0, 0

    print(f"[chunk {chunk_id}] 📦 找到 {len(camera_zips)} 个 camera zip + {len(label_zips)} 个 label zip，共 {len(all_zips)} 个")

    # ── 第一步：解压 ──
    print(f"[chunk {chunk_id}] 🔓 开始解压...")
    unzip_results = {}  # Path -> (success, message)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(unzip_file, z): z for z in all_zips}
        for future in as_completed(futures):
            _, ok, msg = future.result()
            zip_path = futures[future]  # 用 Path 对象作为 key（不是 str）
            unzip_results[zip_path] = (ok, msg)
            zip_name = zip_path.name
            if ok:
                print(f"  ✅ {zip_name}: {msg}")
            else:
                print(f"  ❌ {zip_name}: {msg}")

    # ── 第二步：验证完整性 ──
    print(f"[chunk {chunk_id}] 🔍 验证解压完整性...")
    verify_results = {}  # zip_path -> (complete, message)
    for zip_path in all_zips:
        ok, _ = unzip_results[zip_path]
        if not ok:
            verify_results[zip_path] = (False, "解压失败，跳过验证")
            continue
        complete, msg = verify_extraction(zip_path)
        verify_results[zip_path] = (complete, msg)
        status_icon = "✅" if complete else "❌"
        print(f"  {status_icon} {zip_path.name}: {msg}")

    # ── 第三步：删除已验证的 zip ──
    deleted_count = 0
    if delete_after_verify:
        print(f"[chunk {chunk_id}] 🗑️  删除已验证的 zip 文件...")
        for zip_path in all_zips:
            complete, _ = verify_results[zip_path]
            if not complete:
                print(f"  ⏭️  跳过 {zip_path.name} (验证未通过)")
                continue
            ok, msg = delete_zip(zip_path)
            if ok:
                print(f"  🗑️  {msg}")
                deleted_count += 1
            else:
                print(f"  ⚠️  {msg}")
    else:
        complete_count = sum(1 for c, _ in verify_results.values() if c)
        print(f"[chunk {chunk_id}] ℹ️  验证完成: {complete_count}/{len(all_zips)} 个完整 (--delete 未启用，不删除 zip)")

    success_count = sum(1 for ok, _ in unzip_results.values() if ok)
    fail_count = len(all_zips) - success_count
    return success_count, fail_count, deleted_count


def main():
    parser = argparse.ArgumentParser(description="解压 PhysicalAI chunk 数据文件")
    parser.add_argument(
        "--chunks",
        type=str,
        required=True,
        help="Chunk ID，单个或逗号分隔，如: 6 或 6,7,8 或 9",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=BASE_DIR,
        help=f"数据根目录 (default: {BASE_DIR})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="并发解压线程数 (default: 4)",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="解压验证完整后删除原始 zip 文件",
    )
    args = parser.parse_args()

    try:
        chunk_ids = parse_chunks(args.chunks)
    except ValueError as e:
        print(f"❌ 参数错误: {e}")
        sys.exit(1)

    print(f"🎯 处理 chunks: {chunk_ids}")
    print(f"📁 数据目录: {args.base_dir}")
    print(f"⚙️  并发线程: {args.workers}")
    if args.delete:
        print(f"🗑️  模式: 解压后删除 zip (仅已验证完整者)")
    print("=" * 60)

    total_success = 0
    total_fail = 0
    total_deleted = 0

    for cid in chunk_ids:
        s, f, d = process_chunk(cid, args.base_dir, args.workers, args.delete)
        total_success += s
        total_fail += f
        total_deleted += d

    print("=" * 60)
    print(f"📊 总计: ✅ {total_success} 个解压成功, ❌ {total_fail} 个失败, 🗑️ {total_deleted} 个 zip 已删除")

    if total_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
