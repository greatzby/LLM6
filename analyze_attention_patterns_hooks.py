import torch
import matplotlib.pyplot as plt
import argparse
import glob
import os
from model import GPT, GPTConfig # 导入您未经修改的原始模型定义

# --- (此脚本使用Hooks，无需修改 model.py) ---

# 辅助类，用于优雅地管理钩子和捕获的数据
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
        raise FileNotFoundError(f"错误：未找到匹配的目录: {pattern}")
    latest_dir = sorted(dirs)[-1]
    iteration = 50000
    expected_filename = f"ckpt_mix{ratio}_seed{seed}_iter{iteration}.pt"
    path = os.path.join(latest_dir, expected_filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"错误：在目录 {latest_dir} 中未找到预期的最终 checkpoint 文件 '{expected_filename}'")
    return path

def load_model_from_path(model_path, device='cpu'):
    print(f"[*] 正在加载模型: {model_path}")
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
    print("[*] 模型加载成功。")
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
    fig, axes = plt.subplots(2, n_head, figsize=(n_head * 3, 6.5), sharex=True, sharey=True, squeeze=False) # <--- 修正1: 添加 squeeze=False 保证axes总是二维
    
    fig.suptitle(f'输入序列 "{input_text}" 的注意力模式 (预测下一个Token)\n(通过钩子非侵入式提取)', fontsize=16)
    labels = list(input_text)
    for i in range(n_head):
        # <--- 修正2: 现在因为有了squeeze=False, axes[0, i]总是安全的了 --->
        ax0 = axes[0, i]
        im0 = ax0.imshow(attn_map_0[i].cpu().numpy(), cmap='viridis', vmin=0, vmax=1)
        ax0.set_title(f'mix0 - 注意力头 {i+1}')
        ax0.set_yticks(range(len(labels))); ax0.set_yticklabels(labels)
        if i == 0: ax0.set_ylabel('查询位置 (Query)')
        
        ax1 = axes[1, i]
        im1 = ax1.imshow(attn_map_20[i].cpu().numpy(), cmap='viridis', vmin=0, vmax=1)
        ax1.set_title(f'mix20 - 注意力头 {i+1}')
        ax1.set_xticks(range(len(labels))); ax1.set_xticklabels(labels, rotation=90)
        ax1.set_xlabel('键位置 (Key)')
        if i == 0: ax1.set_ylabel('查询位置 (Query)')

    fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.6, label="注意力权重")
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    output_filename = f"attention_visualization_hooks_seed{seed}.png"
    plt.savefig(output_filename, dpi=300)
    print(f"\n✅ 注意力可视化图已保存至: {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用Hooks可视化并比较 mix0 和 mix20 模型的注意力模式。")
    parser.add_argument('--seed', type=int, default=42, help='指定实验用的随机种子 (例如: 42)')
    args = parser.parse_args()
    visualize_attention(args.seed)