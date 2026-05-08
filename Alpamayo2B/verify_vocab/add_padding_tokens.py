import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

from transformers import AutoTokenizer

# Load the corrected tokenizer
print('=== Loading Corrected Tokenizer ===')
tokenizer = AutoTokenizer.from_pretrained(
    '/home/user/cosmos_reason2_expanded/tokenizer_corrected',
    trust_remote_code=True
)

print(f'Current vocab size: {len(tokenizer)}')

# Add padding tokens to match Alpamayo1.5 exactly
padding_tokens = [f'<|_padding_{i}|>' for i in range(9)]
print(f'\nAdding padding tokens: {padding_tokens}')
num_added = tokenizer.add_tokens(padding_tokens, special_tokens=True)
print(f'Added {num_added} padding tokens')

# Verify final vocab size
print(f'\nFinal vocab size: {len(tokenizer)}')
print(f'Expected (Alpamayo1.5): 155697')
print(f'Match: {len(tokenizer) == 155697}')

# Verify all Alpamayo1.5 special tokens are present
print('\n=== Verifying All Alpamayo1.5 Special Tokens ===')
alpamayo15_tokens = [
    'prompt_start', 'prompt_end', 'image_start', '_padding_0', 'image_end',
    'traj_history_start', '_padding_1', 'traj_history_end', 'cot_start', 'cot_end',
    '_padding_2', '_padding_3', 'traj_future_start', '_padding_4', 'traj_future_end',
    'traj_history', 'traj_future', 'image_pad', '_padding_5', '_padding_6',
    '_padding_7', '_padding_8', 'route_start', 'route_pad', 'route_end',
    'question_start', 'question_end', 'answer_start', 'answer_end'
]

all_present = True
for token_name in alpamayo15_tokens:
    token = f'<|{token_name}|>'
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is not None and token_id != tokenizer.unk_token_id:
        print(f'  ✅ {token}: {token_id}')
    else:
        print(f'  ❌ {token}: MISSING')
        all_present = False

print(f'\nAll tokens present: {all_present}')

# Save the final corrected tokenizer
output_dir = '/home/user/cosmos_reason2_expanded/tokenizer_final'
os.makedirs(output_dir, exist_ok=True)
tokenizer.save_pretrained(output_dir)
print(f'\n=== Saved final tokenizer to {output_dir} ===')
