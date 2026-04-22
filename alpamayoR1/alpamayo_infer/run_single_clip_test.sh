#!/bin/bash
# 运行 Alpamayo 1.5 单 clip 前 100 帧测试

CLIP_ID="01d3588e-bca7-4a18-8e74-c6cfe9e996db"

cd /home/user/alpamayo_infer

echo "运行 Alpamayo 1.5 推理测试..."
echo "Clip ID: $CLIP_ID"
echo "Frames: 100"
echo "GPUs: 1,2,3"

/home/user/miniconda3/envs/alpamayo_env/bin/python batch_inference_alpamayo1_5.py \
    --chunk 0 \
    --clip "$CLIP_ID" \
    --num_frames 100 \
    --gpus 1,2,3 \
    --batch_size 4 \
    2>&1
