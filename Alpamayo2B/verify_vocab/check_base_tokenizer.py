import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

from transformers import AutoTokenizer
import json

# Load 2B base tokenizer
print('=== Cosmos-2B Base Tokenizer ===')
tokenizer_2b = AutoTokenizer.from_pretrained(
    '/data01/mikelee/weight/Cosmos-Reason2-2B',
    trust_remote_code=True
)

print(f'Vocab size: {len(tokenizer_2b)}')
print(f'Added tokens count: {len(tokenizer_2b.added_tokens_encoder)}')

# Show all added tokens with IDs
print('\nAll added tokens (2B):')
for token, idx in sorted(tokenizer_2b.added_tokens_encoder.items(), key=lambda x: x[1]):
    print(f'  {idx}: {token}')

# Check base vocab structure
print(f'\nBase vocab info:')
print(f'  vocab_size: {tokenizer_2b.vocab_size}')
print(f'  total tokens: {len(tokenizer_2b)}')
print(f'  added tokens: {len(tokenizer_2b.added_tokens_encoder)}')

# Check if Alpamayo1.5-10B config mentions the base model
config_path = '/data01/mikelee/weight/models--nvidia--Alpamayo-1.5-10B/config.json'
with open(config_path, 'r') as f:
    config = json.load(f)
print(f'\nAlpamayo1.5 base model: {config.get("vlm_name_or_path", "NOT FOUND")}')
print(f'Alpamayo1.5 vocab_size: {config.get("vocab_size", "NOT FOUND")}')
