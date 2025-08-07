import torch
import matplotlib.pyplot as plt
import argparse
import glob
from model import GPT, GPTConfig # 导入您未经修改的原始模型定义

# --- (此脚本使用Hooks，无需修改 model.py) ---

# 辅助类，用于优雅地管理钩子和捕获的数据
class AttentionExtractor:
    """
    一个上下文管理器，用于在特定层上临时注册钩子并捕获其输入。
    """
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.captured_tensors = []
        self._hook_handle = None

    def _hook_fn(self, module, input_tensors, output_tensors):
        # 钩子函数：当目标层被调用时，此函数会自动执行
        # 注意力权重是送入dropout层的输入，所以我们捕获input_tensors[0]
        # .clone().detach() 是为了安全地复制张量，不影响原始的计算图
        self.captured_tensors.append(input_tensors[0].clone().detach())

    def __enter__(self):
        # 进入上下文时，找到目标层并注册钩子
        target_layer = self.model
        for part in self.target_layer_name.split('.'):
            target_layer = getattr(target_layer, part)
        
        # 注册一个“前向钩子”，它会在目标层forward()执行后被调用
        self._hook_handle = target_layer.register_forward_hook(self._hook_fn)
        return self # 返回自身，以便在 with 语句中使用

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 退出上下文时，移除钩子，防止内存泄漏和意外行为
        if self._hook_handle:
            self._hook_handle.remove()
        self.captured_tensors.clear()

def get_final_checkpoint_path(ratio, seed, checkpoint_dir="out_d92"):
    # (此函数与前面脚本中的完全相同，为简洁省略，您可从上面复制)
    pattern = f"{checkpoint_dir}/composition_mix{ratio}_seed{seed}_*"
    dirs = glob.glob(pattern); 
    if not dirs: raise FileNotFoundError(f"错误：未找到匹配的目录: {pattern}")
    latest_dir = sorted(dirs)[-1]
    path = os.path.join(latest_dir, f"ckpt_mix{ratio}_seed{seed}_iter50000.pt")
    if not os.path.exists(path): raise FileNotFoundError(f"错误：未找到文件 '{path}'")
    return path

def load_model_from_path(model_path, device='cpu'):
    print(f"[*] 正在加载模型: {model_path}")
    ckpt = torch.load(model_path, map_location=device)
    gptconf = GPTConfig(**ckpt['model_args'])
    model = GPT(gptconf)
    state_dict = ckpt['model']
    # 修复PyTorch 2.0+编译模型后可能出现的键名前缀问题
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
    # 定义一个有代表性的组合例子: "A,F->C"
    # 任务: "A,B,C,D,E,F -> A,F,C"
    # 输入序列: "A,B,C,D,E,F,>,A,F," ('>'是提示符)
    # 我们想看模型在预测最后一个 "C" 时，注意力集中在哪里
    vocab = {chr(ord('A')+i): i for i in range(6)}
    vocab.update({',': 6, '>': 7})
    input_text = "A,B,C,D,E,F,>,A,F,"
    input_ids = torch.tensor([[vocab[c] for c in input_text]], dtype=torch.long)
    
    path_0 = get_final_checkpoint_path(0, seed)
    path_20 = get_final_checkpoint_path(20, seed)
    model_0 = load_model_from_path(path_0)
    model_20 = load_model_from_path(path_20)

    # 目标层：h.0中，注意力模块里的attn_dropout层。它的输入就是我们想要的注意力矩阵。
    target_layer_name = "transformer.h.0.attn.attn_dropout"

    # 使用钩子分别获取两个模型的注意力权重
    with AttentionExtractor(model_0, target_layer_name) as extractor_0:
        model_0(input_ids) # 正常执行前向传播，钩子会自动捕获数据
        attn_map_0 = extractor_0.captured_tensors[0]

    with AttentionExtractor(model_20, target_layer_name) as extractor_20:
        model_20(input_ids)
        attn_map_20 = extractor_20.captured_tensors[0]

    # 张量形状: (batch, n_head, seq_len, seq_len) -> 挤压掉 batch 维度
    attn_map_0 = attn_map_0.squeeze(0)
    attn_map_20 = attn_map_20.squeeze(0)

    # --- 可视化 ---
    n_head = attn_map_0.size(0)
    fig, axes = plt.subplots(2, n_head, figsize=(n_head * 3, 6.5), sharex=True, sharey=True)
    fig.suptitle(f'输入序列 "{input_text}" 的注意力模式 (预测下一个Token)\n(通过钩子非侵入式提取)', fontsize=16)
    labels = list(input_text)
    for i in range(n_head):
        # 绘制 mix0 模型
        ax0 = axes[0, i]
        im0 = ax0.imshow(attn_map_0[i].cpu().numpy(), cmap='viridis', vmin=0, vmax=1)
        ax0.set_title(f'mix0 - 注意力头 {i+1}')
        ax0.set_yticks(range(len(labels))); ax0.set_yticklabels(labels)
        if i == 0: ax0.set_ylabel('查询位置 (Query)')
        
        # 绘制 mix20 模型
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