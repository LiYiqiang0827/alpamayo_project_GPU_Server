import torch
import numpy as np
import os

os.environ["HF_HUB_DISABLE_XNET"] = "1"

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper

print("Loading model...")
model = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16).to("cuda")
processor = helper.get_processor(model.tokenizer)

B = 1
num_cameras = 4
num_frames = 4
H, W = 224, 224

print(f"Creating mock data: {num_cameras} cameras, {num_frames} frames, {H}x{W}")

image_frames = torch.randint(0, 256, (num_cameras, num_frames, 3, H, W), dtype=torch.uint8)
messages = helper.create_message(image_frames.flatten(0, 1))

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=False,
    continue_final_message=True,
    return_dict=True,
    return_tensors="pt",
)

ego_history_xyz = torch.randn(B, 1, 16, 3)
ego_history_rot = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(B, 1, 16, 1, 1)

model_inputs = {
    "tokenized_data": inputs,
    "ego_history_xyz": ego_history_xyz,
    "ego_history_rot": ego_history_rot,
}

model_inputs = helper.to_device(model_inputs, "cuda")

print("Running inference...")
torch.cuda.manual_seed_all(42)
with torch.autocast("cuda", dtype=torch.bfloat16):
    pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
        data=model_inputs,
        top_p=0.98,
        temperature=0.6,
        num_traj_samples=1,
        max_generation_length=256,
        return_extra=True,
    )

print(f"Output pred_xyz shape: {pred_xyz.shape}")
print(f"Output pred_rot shape: {pred_rot.shape}")
print(f"Chain-of-Causation: {extra['cot'][0]}")
print("Inference completed successfully!")
