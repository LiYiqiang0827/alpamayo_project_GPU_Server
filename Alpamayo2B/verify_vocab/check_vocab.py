#!/usr/bin/env python3
from transformers import AutoTokenizer

t = AutoTokenizer.from_pretrained(
    '/data01/mikelee/weight/Cosmos-Reason2-2B',
    trust_remote_code=True
)

print("=== Cosmos Reason 2B Tokenizer ===")
print("Vocab size from len(tokenizer):", len(t))
print("eos_token:", t.eos_token, "id:", t.eos_token_id)
print("pad_token:", t.pad_token, "id:", t.pad_token_id)
print()

# Check tokenizer.json to see actual vocab
import json
with open('/data01/mikelee/weight/Cosmos-Reason2-2B/tokenizer.json') as f:
    tok_data = json.load(f)
    
print("tokenizer.json vocab size:", len(tok_data.get('model', {}).get('vocab', {})))
