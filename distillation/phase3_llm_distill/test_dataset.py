#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')

from transformers import AutoTokenizer
from alpamayo_sft_dataset import AlpamayoSFTDataset

print('=== Loading Tokenizer ===')
tokenizer = AutoTokenizer.from_pretrained(
    '/data01/mikelee/weight/Cosmos-Reason2-2B',
    trust_remote_code=True
)
print(f'Vocab size: {len(tokenizer)}')

print()
print('=== Creating Dataset (on-demand mode) ===')
dataset = AlpamayoSFTDataset(
    infer_result_csv='/data01/mikelee/infer_result/infer_result_20260424_161448/infer_results_all.csv',
    tokenizer=tokenizer,
    data_root='/data01/mikelee/data',
    preload_all=False  # on-demand loading
)

print(f'Dataset size: {len(dataset)}')

print()
print('=== Testing __getitem__ ===')
sample = dataset[0]
print(f"Sample keys: {sample.keys()}")
print(f"Images shape: {sample['images'].shape}")
print(f"History shape: {sample['history'].shape}")
print(f"Labels shape: {sample['labels'].shape}")
print(f"CoC text: {sample['cot_text']}")

# Decode labels
decoded = tokenizer.decode(sample['labels'][:20])
print(f"Decoded first 20 tokens: {decoded}")

print()
print('=== Testing index lookup for specific frame ===')
# Test with frame_id = 10
sample2 = dataset[1]
print(f"Frame 10 - CoC text: {sample2['cot_text']}")

print()
print('All tests passed!')
