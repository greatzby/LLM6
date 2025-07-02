# check_checkpoint.py
import torch

checkpoint_path = 'out/composition_20250702_063926/ckpt_50000.pt'
ckpt = torch.load(checkpoint_path, map_location='cpu')

print("Checkpoint keys:", ckpt.keys())
print("\nModel state dict keys:")
if 'model' in ckpt:
    for key in sorted(ckpt['model'].keys()):
        print(f"  {key}: {ckpt['model'][key].shape}")
else:
    print("No 'model' key found")