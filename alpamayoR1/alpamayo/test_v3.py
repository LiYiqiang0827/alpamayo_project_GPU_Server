import torch
import numpy as np
import sys
sys.path.insert(0, '/home/user/mikelee/alpamayo-main/src')
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper
import copy

model = AlpamayoR1.from_pretrained('/data01/vla/models--nvidia--Alpamayo-R1-10B/snapshots/22fab1399111f50b52bfbe5d8b809f39bd4c2fe1', dtype=torch.bfloat16).to('cuda')
processor = helper.get_processor(model.tokenizer)

clip = '01d3588e-bca7-4a18-8e74-c6cfe9e996db'
base = f'/data01/vla/data/data_sample_chunk0/infer/{clip}'

hist = np.load(f'{base}/data/egomotion/ego_000000_history_local.npy', allow_pickle=True).item()
future = np.load(f'{base}/data/egomotion/ego_000000_future_gt.npy', allow_pickle=True).item()

hist_xyz = torch.from_numpy(hist['xyz']).float().unsqueeze(0).unsqueeze(0)
hist_rot = torch.from_numpy(hist['rotation_matrix']).float().unsqueeze(0).unsqueeze(0)

from einops import rearrange

# ... existing code ...

images = torch.randn(4, 4, 3, 224, 224).cuda()  # 4 cameras x 4 frames
images = rearrange(images, 'c t ch h w -> (c t) ch h w')  # flatten to (16, 3, H, W)
messages = helper.create_message(images)
inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, continue_final_message=True, return_dict=True, return_tensors='pt')

model_inputs = {
    'tokenized_data': inputs,
    'ego_history_xyz': hist_xyz.cuda(),
    'ego_history_rot': hist_rot.cuda(),
}
model_inputs = helper.to_device(model_inputs, 'cuda')

with torch.autocast('cuda', dtype=torch.bfloat16):
    pred_xyz, extra = model.sample_trajectories_from_data_with_vlm_rollout(
        data=copy.deepcopy(model_inputs), num_traj_samples=1, top_p=0.98, temperature=0.6, max_generation_length=256
    )

pred = pred_xyz.cpu().numpy()[0, 0, 0, :, :2]
gt = future['xyz'][:, :2]
diff = np.linalg.norm(pred - gt, axis=1)
print(f'Clip1 (v3修复后) - ADE: {diff.mean():.3f}m')
print(f'预测终点: [{pred[-1,0]:.2f}, {pred[-1,1]:.2f}]')
print(f'真值终点: [{gt[-1,0]:.2f}, {gt[-1,1]:.2f}]')
