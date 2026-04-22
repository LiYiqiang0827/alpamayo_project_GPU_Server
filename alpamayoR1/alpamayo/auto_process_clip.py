#!/usr/bin/env python3
"""
一键自动化处理 Alpamayo Clip
用法: python3 auto_process_clip.py --clip <clip_id> [--chunk 0] [--step 10] [--traj 1]
"""
import os
import sys
import argparse
import subprocess
import time

def run_command(cmd, description):
    """运行命令并打印输出"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"命令: {cmd}")
    start = time.time()
    result = subprocess.run(cmd, shell=True)
    elapsed = time.time() - start
    print(f"耗时: {elapsed:.1f}s")
    if result.returncode != 0:
        print(f"❌ 错误: 命令返回码 {result.returncode}")
        return False
    print(f"✅ 完成")
    return True

def main():
    parser = argparse.ArgumentParser(description='一键处理 Alpamayo Clip')
    parser.add_argument('--clip', type=str, required=True, help='Clip ID')
    parser.add_argument('--chunk', type=int, default=0, help='Chunk number (default: 0)')
    parser.add_argument('--step', type=int, default=10, help='推理步长 (default: 10)')
    parser.add_argument('--traj', type=int, default=1, help='轨迹数量 (default: 1)')
    parser.add_argument('--skip-preprocess', action='store_true', help='跳过预处理')
    parser.add_argument('--skip-inference', action='store_true', help='跳过推理')
    parser.add_argument('--skip-video', action='store_true', help='跳过视频生成')
    args = parser.parse_args()
    
    clip_id = args.clip
    chunk = args.chunk
    
    print(f"🚀 开始处理 Clip: {clip_id}")
    print(f"   Chunk: {chunk}")
    print(f"   推理步长: {args.step}")
    print(f"   轨迹数: {args.traj}")
    
    # 激活环境
    activate_cmd = "source ~/mikelee/alpamayo-main/.venv/bin/activate"
    
    # Step 0-2: 预处理
    if not args.skip_preprocess:
        if not run_command(
            f"{activate_cmd} && cd ~/alpamayo && python3 preprocess_clip_fixed.py --clip {clip_id} --chunk {chunk}",
            "Step 0-2: 数据预处理"
        ):
            return 1
    
    # Step 推理
    if not args.skip_inference:
        if not run_command(
            f"{activate_cmd} && cd ~/alpamayo && python3 run_inference.py --clip {clip_id} --chunk {chunk} --step {args.step} --traj {args.traj}",
            f"Step 推理: 每{args.step}帧推理一次, {args.traj}条轨迹"
        ):
            return 1
    
    # Step 3: 生成组合帧
    if not run_command(
        f"{activate_cmd} && cd ~/alpamayo && python3 batch_create_combined_v2.py --clip {clip_id} --chunk {chunk}",
        "Step 3: 生成组合帧 (JPG)"
    ):
        return 1
    
    # Step 4: 生成视频
    if not args.skip_video:
        if not run_command(
            f"{activate_cmd} && cd ~/alpamayo && python3 step4_create_video_from_frames.py --clip {clip_id} --chunk {chunk}",
            "Step 4: 生成视频 (MP4)"
        ):
            return 1
    
    # 下载到本地
    print(f"\n{'='*60}")
    print("下载视频到本地...")
    print(f"{'='*60}")
    local_cmd = f"scp gpu-server:/data01/vla/data/data_sample_chunk{chunk}/infer/{clip_id}/result/combined_video_{clip_id}.mp4 ~/alpamayo/"
    print(f"命令: {local_cmd}")
    result = subprocess.run(local_cmd, shell=True)
    if result.returncode == 0:
        print(f"✅ 视频已下载到: ~/alpamayo/combined_video_{clip_id}.mp4")
    else:
        print(f"⚠️  下载失败，请手动运行: {local_cmd}")
    
    print(f"\n{'='*60}")
    print(f"🎉 全部完成! Clip: {clip_id}")
    print(f"{'='*60}")
    return 0

if __name__ == '__main__':
    sys.exit(main())
