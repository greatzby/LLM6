import torch
import matplotlib.pyplot as plt
import argparse
import glob
import os
from model import GPT, GPTConfig

# --- (This script uses Hooks and does not modify model.py) ---

class AttentionExtractor:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.captured_tensors = []
        self._hook_handle = None

    def _hook_fn(self, module, input_tensors, output_tensors):
        self.captured_tensors.append(input_tensors[0].clone().detach())

    def __enter__(self):
        target_layer = self.model
        for part in self.target_layer_name.split('.'):
            target_layer = getattr(target_layer, part)
        self._hook_handle = target_layer.register_forward_hook(self._hook_fn)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._hook_handle:
            self._hook_handle.remove()
        self.captured_tensors.clear()

def get_final_checkpoint_path(ratio, seed, checkpoint_dir="out_d92"):
    pattern = f"{checkpoint_dir}/composition_mix{ratio}_seed{seed}_*"
    dirs = glob.glob(pattern)
    if not dirs:
        raise FileNotFoundError(f"Error: No matching directory found for pattern: {pattern}")
    latest_dir = sorted(dirs)[-1]
    iteration = 50000
    expected_filename = f"ckpt_mix{ratio}_seed{seed}_iter{iteration}.pt"
    path = os.path.join(latest_dir, expected_filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Error: Expected checkpoint file '{expected_filename}' not found in {latest_dir}")
    return path

def load_model_from_path(model_path, device='cpu'):
    print(f"[*] Loading model: {model_path}")
    ckpt = torch.load(model_path, map_location=device)
    gptconf = GPTConfig(**ckpt['model_args'])
    model = GPT(gptconf)
    state_dict = ckpt['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    print("[*] Model loaded successfully.")
    return model

def visualize_attention(seed):
    vocab = {chr(ord('A')+i): i for i in range(6)}
    vocab.update({',': 6, '>': 7})
    input_text = "A,B,C,D,E,F,>,A,F,"
    input_ids = torch.tensor([[vocab[c] for c in input_text]], dtype=torch.long)
    
    path_0 = get_final_checkpoint_path(0, seed)
    path_20 = get_final_checkpoint_path(20, seed)
    model_0 = load_model_from_path(path_0)
    model_20 = load_model_from_path(path_20)

    target_layer_name = "transformer.h.0.attn.attn_dropout"

    with AttentionExtractor(model_0, target_layer_name) as extractor_0:
        model_0(input_ids)
        attn_map_0 = extractor_0.captured_tensors[0]

    with AttentionExtractor(model_20, target_layer_name) as extractor_20:
        model_20(input_ids)
        attn_map_20 = extractor_20.captured_tensors[0]

    attn_map_0 = attn_map_0.squeeze(0)
    attn_map_20 = attn_map_20.squeeze(0)

    n_head = attn_map_0.size(0)
    # REVISED PLOTTING LOGIC
    fig, axes = plt.subplots(2, n_head, figsize=(n_head * 4, 8), squeeze=False)
    fig.suptitle(f'Attention Pattern for Input: "{input_text}"\n(Predicting the next token)', fontsize=16)
    
    labels = list(input_text)
    
    # Common image object for colorbar
    im = None

    for i in range(n_head):
        # Plot for mix0 model
        ax0 = axes[0, i]
        ax0.imshow(attn_map_0[i].cpu().numpy(), cmap='viridis', vmin=0, vmax=1)
        ax0.set_title(f'mix0 - Head {i+1}')
        ax0.set_yticks(range(len(labels)))
        ax0.set_yticklabels(labels)
        if i == 0: ax0.set_ylabel('Query Position')
        
        # Plot for mix20 model
        ax1 = axes[1, i]
        im = ax1.imshow(attn_map_20[i].cpu().numpy(), cmap='viridis', vmin=0, vmax=1)
        ax1.set_title(f'mix20 - Head {i+1}')
        ax1.set_xticks(range(len(labels)))
        ax1.set_xticklabels(labels, rotation=90)
        ax1.set_xlabel('Key Position')
        if i == 0: ax1.set_ylabel('Query Position')

    # Adjust layout to make space for the colorbar
    fig.subplots_adjust(right=0.85, hspace=0.3, wspace=0.3)
    
    # Add a new axis for the colorbar to the right of the subplots
    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7]) # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax, label="Attention Weight")

    output_filename = f"attention_visualization_hooks_seed{seed}_EN.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\n✅ Attention visualization saved to: {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize and compare attention patterns of mix0 and mix20 models using Hooks.")
    parser.add_argument('--seed', type=int, default=42, help='Seed used for the experiment (e.g., 42)')
    args = parser.parse_args()
    visualize_attention(args.seed)