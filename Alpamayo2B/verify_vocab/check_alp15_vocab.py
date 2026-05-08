#!/usr/bin/env python3
import json

# Check config
with open('/data01/mikelee/weight/models--nvidia--Alpamayo-1.5-10B/config.json') as f:
    config = json.load(f)

print("=== Alpamayo 1.5-10B Config ===")
print("vocab_size:", config.get('vocab_size'))
print("traj_vocab_size:", config.get('traj_vocab_size'))
print("traj_token_start_idx:", config.get('traj_token_start_idx'))
print("bos_token_id:", config.get('bos_token_id'))
print("eos_token_id:", config.get('eos_token_id'))
print()

# Check tokenizer.json if it exists
import os
tokenizer_json = '/data01/mikelee/weight/models--nvidia--Alpamayo-1.5-10B/tokenizer.json'
if os.path.exists(tokenizer_json):
    with open(tokenizer_json) as f:
        tok_data = json.load(f)
    vocab = tok_data.get('model', {}).get('vocab', {})
    print("tokenizer.json vocab size:", len(vocab))
else:
    print("No tokenizer.json found")

# Check special tokens from config
print()
print("=== Special Token IDs ===")
traj_ids = config.get('traj_token_ids', {})
print("traj_token_ids:", traj_ids)

# Summary
print()
print("=== Summary ===")
print("Base vocab (0 to traj_token_start_idx-1): 0 to", config.get('traj_token_start_idx') - 1)
print("Trajectory tokens:", config.get('traj_token_start_idx'), "to", config.get('vocab_size') - 1)
print("Number of trajectory tokens:", config.get('vocab_size') - config.get('traj_token_start_idx'))
