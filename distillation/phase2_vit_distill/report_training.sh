#!/bin/bash
# Training progress reporter - run every 15 minutes
# Usage: bash report_training.sh

LOG_FILE="$HOME/mikelee/alpamayo_project/distillation/phase2_vit_distill/training.log"
OUTPUT_DIR="/gpfs-data/mikelee/distillation_output"

echo "========== 🐢 小胖龟训练汇报 $(date '+%H:%M') =========="
echo ""

# Check if training is running
RUNNING=$(ps aux | grep vit_distillation | grep -v grep | wc -l)
if [ "$RUNNING" -eq 0 ]; then
    echo "⚠️  训练进程未运行！"
    echo ""
fi

# Parse latest training metrics
echo "📊 最新训练状态:"
tail -1 "$LOG_FILE" 2>/dev/null | grep -E "Epoch.*Step.*Loss" || echo "  无最新训练记录"

# Parse latest validation
echo ""
echo "📈 最新验证状态:"
grep "Val Results" "$LOG_FILE" 2>/dev/null | tail -1 || echo "  暂无验证记录"

# Check best model info
echo ""
echo "🏆 Best Model 详情:"
BEST_CKPT=$(ls -t $OUTPUT_DIR/checkpoint_*/best_model/checkpoint_best.pt 2>/dev/null | head -1)
if [ -n "$BEST_CKPT" ]; then
    BEST_SIZE=$(du -h "$BEST_CKPT" | cut -f1)
    BEST_NAME=$(basename "$BEST_CKPT")
    BEST_DIR=$(dirname "$BEST_CKPT")
    
    # Extract epoch and step from the checkpoint using Python (with conda env)
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate alpamayo_env
    python3 -c "
import torch
ckpt = torch.load('$BEST_CKPT', map_location='cpu')
step = ckpt.get('step', 'N/A')
epoch = ckpt.get('epoch', 'N/A')
print(f'  文件: $BEST_NAME')
print(f'  大小: $BEST_SIZE')
print(f'  Epoch: {epoch}')
print(f'  Step: {step}')
print(f'  路径: $BEST_DIR')
"
    
    # Also show the val loss and epoch from log
    BEST_LOG=$(grep -B 2 "New best model" "$LOG_FILE" 2>/dev/null | grep "New best" | tail -1)
    if [ -n "$BEST_LOG" ]; then
        echo "  $BEST_LOG"
        # Try to extract epoch from the log line before "New best"
        BEST_STEP=$(echo "$BEST_LOG" | grep -oP 'step \K\d+')
        if [ -n "$BEST_STEP" ]; then
            EPOCH_LINE=$(grep "Epoch.*Step $BEST_STEP" "$LOG_FILE" 2>/dev/null | head -1)
            if [ -n "$EPOCH_LINE" ]; then
                EPOCH=$(echo "$EPOCH_LINE" | grep -oP 'Epoch \K\d+')
                echo "  对应: Epoch $EPOCH | Step $BEST_STEP"
            fi
        fi
    fi
else
    echo "  暂无 best model"
fi

# Check all checkpoints
echo ""
echo "💾 Checkpoint 列表:"
for ckpt in $(ls -t $OUTPUT_DIR/checkpoint_*/checkpoints/checkpoint_step_*.pt 2>/dev/null | head -3); do
    if [ -f "$ckpt" ]; then
        CKPT_SIZE=$(du -h "$ckpt" | cut -f1)
        CKPT_NAME=$(basename "$ckpt")
        echo "  - $CKPT_NAME ($CKPT_SIZE)"
    fi
done

echo ""
echo "============================================"
