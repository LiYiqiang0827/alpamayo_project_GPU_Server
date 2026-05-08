#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')

from transformers import AutoTokenizer
from alpamayo_sft_dataset import AlpamayoSFTDataset, collate_fn, create_alpamayo_dataloader
from torch.utils.data import DataLoader

print('=== Loading Tokenizer ===')
tokenizer = AutoTokenizer.from_pretrained(
    '/data01/mikelee/weight/Cosmos-Reason2-2B',
    trust_remote_code=True
)
print(f'Vocab size: {len(tokenizer)}')

print()
print('=== Creating Dataset ===')
dataset = AlpamayoSFTDataset(
    infer_result_csv='/data01/mikelee/infer_result/infer_result_20260424_161448/infer_results_all.csv',
    tokenizer=tokenizer,
    data_root='/data01/mikelee/data',
    preload_all=False
)
print(f'Dataset size: {len(dataset)}')

print()
print('=== Creating DataLoader ===')
dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0,  # Use 0 for debugging
    collate_fn=collate_fn
)
print(f'DataLoader created')

print()
print('=== Testing one batch ===')
batch = next(iter(dataloader))
print(f"Batch keys: {batch.keys()}")
print(f"Images batch shape: {batch['images'].shape}")
print(f"History batch shape: {batch['history'].shape}")
print(f"Labels batch shape: {batch['labels'].shape}")

print()
print('=== Testing with num_workers=4 ===')
dataloader2 = DataLoader(
    dataset,
    batch_size=4,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn,
    pin_memory=True
)
batch2 = next(iter(dataloader2))
print(f"Batch2 images shape: {batch2['images'].shape}")
print(f"Batch2 passed!")

print()
print('All DataLoader tests passed!')
