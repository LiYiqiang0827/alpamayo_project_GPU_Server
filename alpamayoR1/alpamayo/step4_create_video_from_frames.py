#!/usr/bin/env python3
"""
Step 4: 从组合帧生成视频
- 输入: combined_frames_v2/ 目录下的 jpg 文件
- 输出: 单个 MP4 视频文件
使用 PyAV 替代 cv2
"""
import os
import re
import numpy as np
from PIL import Image
from tqdm import tqdm
import av

import argparse

def main():
    parser = argparse.ArgumentParser(description='Step 4: 从组合帧生成视频')
    parser.add_argument('--clip', type=str, required=True, help='Clip ID (e.g., 01d3588e-bca7-4a18-8e74-c6cfe9e996db)')
    parser.add_argument('--chunk', type=int, default=0, help='Chunk number (default: 0)')
    args = parser.parse_args()
    
    clip_id = args.clip
    chunk = args.chunk
    
    RESULT_DIR = f'/data01/vla/data/data_sample_chunk{chunk}/infer/{clip_id}/result'
    FRAME_INPUT_DIR = f'{RESULT_DIR}/combined_frames_v2'
    VIDEO_OUTPUT = f'{RESULT_DIR}/combined_video_{clip_id}.mp4'
    
    print(f'=== Step 4: 从组合帧生成视频 ({clip_id}) ===\n')
    
    # 获取所有 jpg 文件
    jpg_files = [f for f in os.listdir(FRAME_INPUT_DIR) if f.endswith('.jpg')]
    jpg_files.sort(key=lambda x: int(re.search(r'(\d+)', x).group()))
    
    if not jpg_files:
        print(f'错误: {FRAME_INPUT_DIR} 目录下没有找到 jpg 文件')
        return
    
    print(f'找到 {len(jpg_files)} 帧')
    
    # 读取第一帧获取视频尺寸
    first_frame = Image.open(f'{FRAME_INPUT_DIR}/{jpg_files[0]}')
    width, height = first_frame.size
    first_frame.close()
    
    # 视频参数
    fps = 10  # 10fps
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
    if video_size_mb < total_jpg_size_mb:
        print(f'   压缩比: {total_jpg_size_mb/video_size_mb:.2f}x (视频比源帧小)')
    else:
        print(f'   视频比源帧大 (可能质量设置较高)')

if __name__ == '__main__':
    main()
