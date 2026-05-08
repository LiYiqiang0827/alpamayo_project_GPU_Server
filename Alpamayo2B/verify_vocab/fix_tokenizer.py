import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

from transformers import AutoTokenizer
import json

# Load base Cosmos-2B tokenizer
print('=== Loading Base Cosmos-2B Tokenizer ===')
tokenizer = AutoTokenizer.from_pretrained(
    '/data01/mikelee/weight/Cosmos-Reason2-2B',
    trust_remote_code=True
)

print(f'Base vocab size: {len(tokenizer)}')

# Add trajectory tokens (matching Alpamayo1.5 format: <i0> without vertical bars)
print('\n=== Adding Trajectory Tokens ===')
traj_tokens = [f'<i{v}>' for v in range(4000)]
num_added = tokenizer.add_tokens(traj_tokens)
print(f'Added {num_added} trajectory tokens')
print(f'First 5: {traj_tokens[:5]}')
print(f'Last 5: {traj_tokens[-5:]}')

# Add special tokens (matching Alpamayo1.5)
print('\n=== Adding Special Tokens ===')
special_tokens = [
    '<|prompt_start|>',
    '<|prompt_end|>', 
    '<|image_start|>',
    '<|image_end|>',
    '<|traj_history_start|>',
    '<|traj_history_end|>',
    '<|cot_start|>',
    '<|cot_end|>',
    '<|traj_future_start|>',
    '<|traj_future_end|>',
    '<|traj_history|>',
    '<|traj_future|>',
    '<|image_pad|>',
    '<|route_start|>',
    '<|route_pad|>',
    '<|route_end|>',
    '<|question_start|>',
    '<|question_end|>',
    '<|answer_start|>',
    '<|answer_end|>',
]

# Add as special tokens
num_special = tokenizer.add_tokens(special_tokens, special_tokens=True)
print(f'Added {num_special} special tokens')

# Set traj_token_start_idx
tokenizer.traj_token_start_idx = tokenizer.convert_tokens_to_ids('<i0>')
print(f'\ntraj_token_start_idx: {tokenizer.traj_token_start_idx}')

# Verify
print(f'\n=== Verification ===')
print(f'Final vocab size: {len(tokenizer)}')
print(f'Expected: 155697 (151669 + 4000 + 28)')

# Check key tokens
print(f'\nKey token IDs:')
print(f'  <i0>: {tokenizer.convert_tokens_to_ids("<i0>")}')
print(f'  <i3999>: {tokenizer.convert_tokens_to_ids("<i3999>")}')
print(f'  <|cot_start|>: {tokenizer.convert_tokens_to_ids("<|cot_start|>")}')
print(f'  <|cot_end|>: {tokenizer.convert_tokens_to_ids("<|cot_end|>")}')
print(f'  <|traj_history_start|>: {tokenizer.convert_tokens_to_ids("<|traj_history_start|>")}')

# Save the corrected tokenizer
output_dir = '/home/user/cosmos_reason2_expanded/tokenizer_corrected'
os.makedirs(output_dir, exist_ok=True)
tokenizer.save_pretrained(output_dir)
print(f'\n=== Saved to {output_dir} ===')
