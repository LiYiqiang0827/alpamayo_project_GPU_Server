#!/usr/bin/env python3
"""
Step 4: 从组合帧生成视频 - V5 版本
适配 result_strict 目录
"""
import os
import sys
import re
import numpy as np
from PIL import Image
from tqdm import tqdm
import av
import argparse

def main():
    parser = argparse.ArgumentParser(description='Step 4: 从组合帧生成视频 (V5)')
    parser.add_argument('--clip', type=str, required=True, help='Clip ID')
    parser.add_argument('--chunk', type=int, default=0, help='Chunk number (default: 0)')
    parser.add_argument('--fps', type=int, default=10, help='视频帧率 (default: 10)')
    args = parser.parse_args()
    
    clip_id = args.clip
    chunk = args.chunk
    fps = args.fps
    
    # V5 使用 result_strict 目录
    RESULT_DIR = f'/data01/vla/data/data_sample_chunk{chunk}/infer/{clip_id}/result_strict'
    FRAME_INPUT_DIR = f'{RESULT_DIR}/combined_frames'
    VIDEO_OUTPUT = f'{RESULT_DIR}/combined_video_{clip_id}.mp4'
    
    print(f'=== Step 4: 从组合帧生成视频 V5 ({clip_id}) ===\n')
    
    # 检查目录
    if not os.path.exists(FRAME_INPUT_DIR):
        print(f'❌ 错误: 找不到帧目录 {FRAME_INPUT_DIR}')
        print(f'   请先运行组合帧生成脚本')
        return 1
    
    # 获取所有 jpg 文件
    jpg_files = [f for f in os.listdir(FRAME_INPUT_DIR) if f.endswith('.jpg')]
    if not jpg_files:
        print(f'❌ 错误: {FRAME_INPUT_DIR} 目录下没有找到 jpg 文件')
        return 1
    
    # 按数字顺序排序
    jpg_files.sort(key=lambda x: int(re.search(r'(\d+)', x).group()))
    
    print(f'找到 {len(jpg_files)} 帧')
    
    # 读取第一帧获取视频尺寸
    first_frame = Image.open(f'{FRAME_INPUT_DIR}/{jpg_files[0]}')
    width, height = first_frame.size
    first_frame.close()
    
    print(f'视频尺寸: {width}x{height}, FPS: {fps}')
    
    # 使用 PyAV 创建视频
    container = av.open(VIDEO_OUTPUT, mode='w')
    stream = container.add_stream('libx264', rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = 'yuv420p'
    stream.options = {'crf': '23', 'preset': 'medium'}
    
    print('\n写入视频帧...')
    for jpg_file in tqdm(jpg_files):
        frame_path = f'{FRAME_INPUT_DIR}/{jpg_file}'
        img = Image.open(frame_path).convert('RGB')
        
        # PIL Image -> numpy array -> PyAV frame
        img_array = np.array(img)
        frame = av.VideoFrame.from_ndarray(img_array, format='rgb24')
        
        # 编码帧
        for packet in stream.encode(frame):
            container.mux(packet)
        
        img.close()
    
    # 刷新剩余帧
    for packet in stream.encode():
        container.mux(packet)
    
    container.close()
    
    # 统计
    video_size_mb = os.path.getsize(VIDEO_OUTPUT) / (1024 * 1024)
    total_jpg_size_mb = sum(os.path.getsize(f'{FRAME_INPUT_DIR}/{f}') for f in jpg_files) / (1024 * 1024)
    
    print(f'\n✅ 视频保存至: {VIDEO_OUTPUT}')
    print(f'   视频大小: {video_size_mb:.1f} MB')
    print(f'   源帧总大小: {total_jpg_size_mb:.1f} MB')
    if video_size_mb > 0:
        print(f'   压缩比: {total_jpg_size_mb/video_size_mb:.2f}x')
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
