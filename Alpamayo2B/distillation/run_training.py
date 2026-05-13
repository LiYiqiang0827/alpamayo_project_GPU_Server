import os
os.environ['TMPDIR'] = '/data01/tmp'
os.environ['TEMP'] = '/data01/tmp'
os.environ['TMP'] = '/data01/tmp'

import torch
from train_distillation_Alpamayo2B import DistillationTrainer, DEFAULT_CONFIG

# 更新配置
config = {
    **DEFAULT_CONFIG,
    'num_epochs': 1,
    'max_samples': 100,  # 先测试100个样本
    'save_steps': 50,
    'logging_steps': 10,
}

print('Starting training with config:')
for key, value in config.items():
    print(f'  {key}: {value}')

# 创建训练器
trainer = DistillationTrainer(config)

# 开始训练
trainer.train()

print('\nTraining complete!')
