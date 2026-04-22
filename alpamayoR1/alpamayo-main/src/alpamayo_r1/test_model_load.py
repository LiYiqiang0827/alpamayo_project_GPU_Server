import torch
import os

os.environ["HF_HUB_DISABLE_XNET"] = "1"

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

print("Loading model from local cache...")
print("This may take a few minutes...")
model = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16)
print("Model loaded to CPU, moving to CUDA...")
model = model.to("cuda")
print("Model loaded successfully!")
print(f"Model type: {type(model)}")
print(f"Device: {model.device}")
print(f"Dtype: {model.dtype}")
