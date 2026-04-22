#!/usr/bin/env python3
"""
一键自动化处理 Alpamayo Clip (V5 严格版本)
用法: python3 auto_process.py --clip <clip_id> [--chunk 0] [--num_frames 20] [--traj 1]
"""
import os
import sys
import argparse
import subprocess
import time

def run_cmd(cmd, desc, check=True):
    """运行bash命令"""
    print(f"\n{'='*60}")
    print(f"{desc}")
    print(f"{'='*60}")
    print(f"命令: {cmd}")
    start = time.time()
    result = subprocess.run(cmd, shell=True, executable='/bin/bash')
    elapsed = time.time() - start
    print(f"耗时: {elapsed:.1f}s")
    if check and result.returncode != 0:
        print(f"❌ 错误: 返回码 {result.returncode}")
        sys.exit(1)
    print(f"✅ 完成")
    return result.returncode

def main():
    parser = argparse.ArgumentParser(description='一键处理 Alpamayo Clip (V5 严格版本)')
    parser.add_argument('--clip', type=str, required=True, help='Clip ID')
    parser.add_argument('--chunk', type=int, default=0, help='Chunk number')
    parser.add_argument('--num_frames', type=int, default=20, help='推理帧数')
    parser.add_argument('--traj', type=int, default=1, help='轨迹数量 (1/3/6)')
    parser.add_argument('--skip-preprocess', action='store_true', help='跳过预处理（如果已预处理过）')
    parser.add_argument('--no-download', action='store_true', help='不下载视频到本地')
    args = parser.parse_args()
    
    clip = args.clip
    chunk = args.chunk
    
    print(f"🚀 处理 Clip: {clip} (V5 严格版本)")
    print(f"   推理帧数: {args.num_frames}, 轨迹数: {args.traj}")
    
    # 使用 miniconda3 的 Python (包含 cv2)
    PYTHON = "~/miniconda3/bin/python3"
    
    # Step 1: 严格预处理 (V5) - 使用 miniconda3
    if not args.skip_preprocess:
        run_cmd(f"{PYTHON} ~/alpamayo/preprocess_strict.py --clip {clip} --chunk {chunk}", 
                "Step 1: V5 严格数据预处理")
    else:
        print(f"\n⏭️  跳过预处理 (使用已有数据)")
    
    # Step 2: V5 严格推理 - 使用 .venv (需要 GPU 环境)
    run_cmd(f"source ~/mikelee/alpamayo-main/.venv/bin/activate && python3 ~/alpamayo/run_inference_new_strict.py --clip {clip} --num_frames {args.num_frames} --traj {args.traj}",
            f"Step 2: V5 严格推理 ({args.num_frames}帧, {args.traj}条轨迹)")
    
    # Step 3: 组合帧 (V5 版本) - 使用 miniconda3
    run_cmd(f"{PYTHON} ~/alpamayo/batch_create_combined_v5.py --clip {clip} --chunk {chunk}",
            "Step 3: 生成组合帧")
    
    # Step 4: 视频 (V5 版本) - 使用 miniconda3
    run_cmd(f"{PYTHON} ~/alpamayo/step4_create_video_v5.py --clip {clip} --chunk {chunk}",
            "Step 4: 生成视频")
    
    # 下载
    if not args.no_download:
        print(f"\n{'='*60}")
        print("下载视频到本地...")
        print(f"{'='*60}")
        dl = f"scp gpu-server:/data01/vla/data/data_sample_chunk{chunk}/infer/{clip}/result_strict/combined_video_{clip}.mp4 ~/alpamayo/"
        result = subprocess.run(dl, shell=True)
        if result.returncode == 0:
            print(f"✅ 完成: ~/alpamayo/combined_video_{clip}.mp4")
        else:
            print(f"⚠️  下载失败，视频保留在服务器: result_strict/combined_video_{clip}.mp4")
    else:
        print(f"\n⏭️  跳过下载")
        print(f"   视频位置: result_strict/combined_video_{clip}.mp4")
    
    print(f"\n{'='*60}")
    print("🎉 全部完成!")
    print(f"{'='*60}")
    return 0

if __name__ == '__main__':
    sys.exit(main())
