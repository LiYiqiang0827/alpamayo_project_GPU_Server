#!/usr/bin/env python3
"""
训练曲线可视化脚本
实时监控和绘制 Alpamayo2B 蒸馏训练的损失曲线

用法:
    python plot_training_curves.py --log-file ddp_training_6gpu_bs1_v5.log
    python plot_training_curves.py --log-file ddp_training_6gpu_bs1_v5.log --output-dir ./plots
    python plot_training_curves.py --log-file ddp_training_6gpu_bs1_v5.log --interval 60  # 每60秒更新一次
"""

import argparse
import re
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import matplotlib
matplotlib.use('Agg')  # 非交互式后端，适合服务器运行
import matplotlib.pyplot as plt
import numpy as np


class TrainingLogParser:
    """解析训练日志文件"""
    
    # 匹配训练日志的正则表达式
    TRAIN_PATTERN = re.compile(
        r'Epoch\s+(\d+):\s+.*?(\d+)/([\d,]+)\s+\[.*?loss=([\d.]+),\s*kl=([\d.]+),\s*ce=([\d.]+),\s*lr=([\d.e+-]+)'
    )
    
    def __init__(self, log_file: str):
        self.log_file = Path(log_file)
        self.train_data: Dict[str, List] = {
            'steps': [],
            'epochs': [],
            'loss': [],
            'kl': [],
            'ce': [],
            'lr': []
        }
        self.last_position = 0
        
    def parse_new_lines(self) -> bool:
        """解析新增的行，返回是否有新数据"""
        if not self.log_file.exists():
            return False
            
        with open(self.log_file, 'r') as f:
            # 跳到上次读取的位置
            f.seek(self.last_position)
            new_lines = f.readlines()
            self.last_position = f.tell()
        
        if not new_lines:
            return False
            
        has_new_data = False
        for line in new_lines:
            # 解析训练数据
            train_match = self.TRAIN_PATTERN.search(line)
            if train_match:
                epoch = int(train_match.group(1))
                step = int(train_match.group(2).replace(',', ''))
                loss = float(train_match.group(4))
                kl = float(train_match.group(5))
                ce = float(train_match.group(6))
                lr = float(train_match.group(7))
                
                self.train_data['steps'].append(step)
                self.train_data['epochs'].append(epoch)
                self.train_data['loss'].append(loss)
                self.train_data['kl'].append(kl)
                self.train_data['ce'].append(ce)
                self.train_data['lr'].append(lr)
                has_new_data = True
        
        return has_new_data
    
    def get_summary(self) -> Dict:
        """获取训练摘要信息"""
        if not self.train_data['steps']:
            return {}
        
        steps = self.train_data['steps']
        loss = self.train_data['loss']
        
        return {
            'current_step': steps[-1],
            'current_epoch': self.train_data['epochs'][-1],
            'current_loss': loss[-1],
            'min_loss': min(loss),
            'max_loss': max(loss),
            'avg_loss_last_100': np.mean(loss[-100:]) if len(loss) >= 100 else np.mean(loss),
            'loss_trend': '下降' if len(loss) > 1 and loss[-1] < loss[0] else '上升',
        }


class TrainingVisualizer:
    """训练可视化器"""
    
    def __init__(self, output_dir: str = './plots'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
    def _smooth_curve(self, data: List[float], window: int = 50) -> List[float]:
        """平滑曲线"""
        if len(data) < window:
            return data
        smoothed = []
        for i in range(len(data)):
            start = max(0, i - window // 2)
            end = min(len(data), i + window // 2 + 1)
            smoothed.append(np.mean(data[start:end]))
        return smoothed
    
    def plot_loss_curve(self, parser: TrainingLogParser, save: bool = True) -> str:
        """绘制损失曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Alpamayo2B Distillation Training Curves', fontsize=16, fontweight='bold')
        
        steps = parser.train_data['steps']
        loss = parser.train_data['loss']
        kl = parser.train_data['kl']
        ce = parser.train_data['ce']
        lr = parser.train_data['lr']
        
        if not steps:
            return ""
        
        # 1. 总损失曲线
        ax1 = axes[0, 0]
        ax1.plot(steps, loss, alpha=0.3, color='blue', label='Raw Loss')
        if len(steps) > 50:
            smoothed_loss = self._smooth_curve(loss, window=50)
            ax1.plot(steps, smoothed_loss, color='red', linewidth=2, label='Smoothed Loss')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss')
        ax1.set_title('Total Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. KL Loss 曲线
        ax2 = axes[0, 1]
        ax2.plot(steps, kl, alpha=0.3, color='green', label='Raw KL')
        if len(steps) > 50:
            smoothed_kl = self._smooth_curve(kl, window=50)
            ax2.plot(steps, smoothed_kl, color='darkgreen', linewidth=2, label='Smoothed KL')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('KL Loss')
        ax2.set_title('KL Divergence Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. CE Loss 曲线
        ax3 = axes[1, 0]
        ax3.plot(steps, ce, alpha=0.3, color='orange', label='Raw CE')
        if len(steps) > 50:
            smoothed_ce = self._smooth_curve(ce, window=50)
            ax3.plot(steps, smoothed_ce, color='darkorange', linewidth=2, label='Smoothed CE')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('CE Loss')
        ax3.set_title('Cross Entropy Loss')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 学习率曲线
        ax4 = axes[1, 1]
        ax4.plot(steps, lr, color='purple', linewidth=2)
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Learning Rate')
        ax4.set_title('Learning Rate Schedule')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = self.output_dir / f'training_curves_step{steps[-1]}_{timestamp}.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            
            # 同时保存一个固定名称的最新版本
            latest_path = self.output_dir / 'training_curves_latest.png'
            plt.savefig(latest_path, dpi=150, bbox_inches='tight')
            
            plt.close()
            return str(output_path)
        else:
            plt.show()
            return ""
    
    def generate_html_report(self, parser: TrainingLogParser) -> str:
        """生成 HTML 报告"""
        summary = parser.get_summary()
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Alpamayo2B Training Monitor</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; text-align: center; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
        .metric {{ background-color: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #007bff; }}
        .metric-label {{ font-size: 14px; color: #666; margin-top: 5px; }}
        .plot-container {{ margin: 20px 0; text-align: center; }}
        .plot-container img {{ max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        .status {{ text-align: center; padding: 10px; border-radius: 5px; margin: 10px 0; }}
        .status.running {{ background-color: #d4edda; color: #155724; }}
        .status.stopped {{ background-color: #f8d7da; color: #721c24; }}
        .timestamp {{ text-align: center; color: #999; font-size: 12px; margin-top: 20px; }}
    </style>
    <meta http-equiv="refresh" content="60">
</head>
<body>
    <div class="container">
        <h1>🚀 Alpamayo2B Distillation Training Monitor</h1>
        
        <div class="status running">
            <strong>Status:</strong> Training in Progress | Step {summary.get('current_step', 'N/A')}
        </div>
        
        <div class="summary">
            <div class="metric">
                <div class="metric-value">{summary.get('current_loss', 0):.4f}</div>
                <div class="metric-label">Current Loss</div>
            </div>
            <div class="metric">
                <div class="metric-value">{summary.get('min_loss', 0):.4f}</div>
                <div class="metric-label">Min Loss</div>
            </div>
            <div class="metric">
                <div class="metric-value">{summary.get('avg_loss_last_100', 0):.4f}</div>
                <div class="metric-label">Avg Loss (Last 100)</div>
            </div>
            <div class="metric">
                <div class="metric-value">{summary.get('loss_trend', 'N/A')}</div>
                <div class="metric-label">Loss Trend</div>
            </div>
        </div>
        
        <div class="plot-container">
            <h2>Training Curves</h2>
            <img src="training_curves_latest.png" alt="Training Curves">
        </div>
        
        <div class="timestamp">
            Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>
"""
        
        html_path = self.output_dir / 'training_monitor.html'
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        return str(html_path)


def main():
    parser = argparse.ArgumentParser(description='Alpamayo2B Training Curve Visualizer')
    parser.add_argument('--log-file', type=str, required=True, help='Path to training log file')
    parser.add_argument('--output-dir', type=str, default='./plots', help='Directory to save plots')
    parser.add_argument('--interval', type=int, default=300, help='Update interval in seconds (default: 300)')
    parser.add_argument('--once', action='store_true', help='Run once and exit (no continuous monitoring)')
    
    args = parser.parse_args()
    
    log_parser = TrainingLogParser(args.log_file)
    visualizer = TrainingVisualizer(args.output_dir)
    
    print(f"📊 Training Visualizer started")
    print(f"   Log file: {args.log_file}")
    print(f"   Output dir: {args.output_dir}")
    print(f"   Update interval: {args.interval}s")
    print(f"   Mode: {'Once' if args.once else 'Continuous'}")
    print("-" * 50)
    
    while True:
        # 解析新数据
        has_new_data = log_parser.parse_new_lines()
        
        if has_new_data:
            # 绘制曲线
            plot_path = visualizer.plot_loss_curve(log_parser)
            
            # 生成 HTML 报告
            html_path = visualizer.generate_html_report(log_parser)
            
            # 打印摘要
            summary = log_parser.get_summary()
            if summary:
                print(f"\n📈 Step {summary['current_step']} | "
                      f"Loss: {summary['current_loss']:.4f} | "
                      f"Min: {summary['min_loss']:.4f} | "
                      f"Trend: {summary['loss_trend']}")
                print(f"   Plots saved: {plot_path}")
                print(f"   HTML report: {html_path}")
        
        if args.once:
            break
            
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
