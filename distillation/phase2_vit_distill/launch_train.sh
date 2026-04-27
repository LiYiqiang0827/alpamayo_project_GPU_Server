#!/bin/bash
# Launch script for Phase 2 ViT Distillation training
# Uses GPUs 1, 2, 3

set -e

SCRIPT_DIR=$(cd $(dirname $0) && pwd)
cd $SCRIPT_DIR

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate alpamayo_env

# Set CUDA devices (use only GPUs 1,2,3)
export CUDA_VISIBLE_DEVICES=1,2,3

# NCCL stability fixes - prevent timeout crashes during eval
export NCCL_TIMEOUT=1800000          # 30 minutes (was default ~10min)
export NCCL_DEBUG=WARN              # Reduce NCCL verbosity
export TORCH_NCCL_TRACE_BUFFER_SIZE=0  # Disable flight recorder overhead

# Also set elastic timeout higher for resumability
export TORCHELASTIC_MAX_RESTARTS=3
export TORCHELASTIC_MAX_RETRIES=3

# Clear old log (fresh start)
rm -f training.log

# Launch with torchrun (3 GPUs, 1 node)
nohup torchrun \
    --nnodes=1 \
    --nproc_per_node=3 \
    --rdzv_id=phase2_vit_4033 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=127.0.0.1:29500 \
    vit_distillation.py \
    --config config_train.json \
    > training.log 2>&1 &

echo "Training launched with PID $!"
echo "Log file: $SCRIPT_DIR/training.log"
echo "To monitor: tail -f $SCRIPT_DIR/training.log"
