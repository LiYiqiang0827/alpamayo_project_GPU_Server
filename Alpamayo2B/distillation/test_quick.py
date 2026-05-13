import os
os.environ['TMPDIR'] = '/data01/tmp'
os.environ['TEMP'] = '/data01/tmp'
os.environ['TMP'] = '/data01/tmp'

import torch
from transformers import AutoTokenizer, AutoProcessor

# 测试1: 数据集加载
print('=== Test 1: Dataset Loading ===')
from dataloader_Alpamayo2B import AlpamayoDistillationDataset

tokenizer = AutoTokenizer.from_pretrained('/data01/mikelee/weight/alpamayo2B/tokenizer_final', trust_remote_code=True)
processor = AutoProcessor.from_pretrained('/data01/mikelee/weight/alpamayo2B', trust_remote_code=True)

dataset = AlpamayoDistillationDataset(
    infer_result_csv='/gpfs-data/mikelee/infer_result/infer_result_20260507_195923/infer_results_all.csv',
    teacher_logits_dir='/gpfs-data/mikelee/infer_result/infer_result_20260507_195923/logits',
    tokenizer=tokenizer,
    processor=processor,
    temperature=2.0,
)

print(f'Dataset size: {len(dataset)}')
if len(dataset) > 0:
    sample = dataset[0]
    print('Sample keys:', sample.keys())
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f'  {key}: shape={value.shape}, dtype={value.dtype}')
    print('Test 1 PASSED')
else:
    print('Test 1 FAILED')

# 测试2: 模型前向传播
print('\n=== Test 2: Model Forward Pass ===')
from model_setup_Alpamayo2B import setup_model_for_distillation

model, tokenizer, processor = setup_model_for_distillation(
    model_path='/data01/mikelee/weight/alpamayo2B',
    tokenizer_path='/data01/mikelee/weight/alpamayo2B/tokenizer_final',
    device='cuda:1',
    dtype=torch.bfloat16,
)

batch_size = 1
seq_len = 10
vocab_size = len(tokenizer)

input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to('cuda:1')
attention_mask = torch.ones(batch_size, seq_len).to('cuda:1')

with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=None)

logits = outputs.logits
print(f'logits shape: {logits.shape}')
assert logits.shape == (batch_size, seq_len, vocab_size)
print('Test 2 PASSED')

# 测试3: 损失函数
print('\n=== Test 3: Loss Computation ===')
from train_distillation_Alpamayo2B import DistillationLoss

loss_fn = DistillationLoss(temperature=2.0, alpha=0.7, beta=0.3)

batch_size = 2
seq_len = 10
vocab_size = 155697

student_logits = torch.randn(batch_size, seq_len, vocab_size)
teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
labels = torch.randint(0, vocab_size, (batch_size, seq_len))
attention_mask = torch.ones(batch_size, seq_len)

total_loss, kl_loss, ce_loss = loss_fn(student_logits, teacher_logits, labels, attention_mask)
print(f'total_loss: {total_loss.item():.4f}, kl_loss: {kl_loss.item():.4f}, ce_loss: {ce_loss.item():.4f}')
assert total_loss.item() > 0
print('Test 3 PASSED')

print('\n=== All Tests Passed! ===')
