import torch
import torch.nn.functional as F
import numpy as np
import argparse
import glob
import os
from model import GPT, GPTConfig

# --- 辅助模块与类 (词汇表已确认无误) ---

VOCAB = {chr(ord('A')+i): i for i in range(6)}
VOCAB.update({',': 6, '>': 7, '\n': 8})
INV_VOCAB = {i: c for c, i in VOCAB.items()}

def get_final_checkpoint_path(ratio, seed, checkpoint_dir="out_d92"):
    """查找指定配置的最终检查点路径"""
    pattern = f"{checkpoint_dir}/composition_mix{ratio}_seed{seed}_*"
    dirs = glob.glob(pattern)
    if not dirs:
        raise FileNotFoundError(f"错误: 找不到匹配的目录: {pattern}")
    latest_dir = sorted(dirs)[-1]
    iteration = 50000
    expected_filename = f"ckpt_mix{ratio}_seed{seed}_iter{iteration}.pt"
    path = os.path.join(latest_dir, expected_filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"错误: 检查点文件 '{expected_filename}' 未在 {latest_dir} 中找到")
    return path

def load_model_from_path(model_path, device='cpu'):
    """从路径加载模型"""
    print(f"[*] 正在加载模型: {model_path}")
    ckpt = torch.load(model_path, map_location=device)
    gptconf = GPTConfig(**ckpt['model_args'])
    model = GPT(gptconf)
    state_dict = ckpt['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    print("[*] 模型加载成功.")
    return model

class AttentionHook:
    """通用的注意力钩子类"""
    def __init__(self, model, layer_name="transformer.h.0.attn.attn_dropout"):
        self.model = model
        self.layer_name = layer_name
        self.captured_attention = None
        self.modification_fn = None
        self._hook_handle = None

    def _hook_fn(self, module, input_tensors, output_tensors):
        attention_weights = input_tensors[0]
        self.captured_attention = attention_weights.clone().detach()
        if self.modification_fn:
            return self.modification_fn(attention_weights)
        return output_tensors

    def set_modification(self, mod_fn):
        self.modification_fn = mod_fn

    def __enter__(self):
        target_layer = self.model
        for part in self.layer_name.split('.'):
            target_layer = getattr(target_layer, part)
        self._hook_handle = target_layer.register_forward_hook(self._hook_fn)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._hook_handle: self._hook_handle.remove()

def display_pred(pred_idx):
    """安全地显示预测字符"""
    char = INV_VOCAB.get(pred_idx, 'N/A')
    return repr(char)

# 这是关键的修正：定义了模型训练时看到的标准上下文
CONTEXT = "A,B,C\nD,E,F\n"

# --- 实验A：输出分析 ---
def analyze_predictions(model_0, model_20):
    print("\n--- 实验A：实际输出分析 (最终修正版) ---")
    print(f"标准上下文: {repr(CONTEXT)}")
    
    test_prompts = {
        "A,F -> C": (">,A,F,", 'C'),
        "B,E -> D": (">,B,E,", 'D'),
        "C,D -> E": (">,C,D,", 'E'),
        "A,C -> C": (">,A,C,", 'C'),
    }
    
    for name, (prompt, expected) in test_prompts.items():
        full_input = CONTEXT + prompt
        print(f"\n[*] 测试案例: {name} (完整输入: {repr(full_input)}) | 预期输出: {repr(expected)}")
        input_ids = torch.tensor([[VOCAB[c] for c in full_input]], dtype=torch.long)
        
        with torch.no_grad():
            logits_0, _ = model_0(input_ids)
            pred_idx_0 = torch.argmax(logits_0[0, -1, :]).item()
            
            logits_20, _ = model_20(input_ids)
            pred_idx_20 = torch.argmax(logits_20[0, -1, :]).item()
            
        print(f"  - Mix0 (无能模型) 预测: {display_pred(pred_idx_0)}")
        print(f"  - Mix20 (有能模型) 预测: {display_pred(pred_idx_20)}")

# --- 实验B：注意力权重定量分析 ---
def quantify_attention(model_20):
    print("\n--- 实验B：注意力权重定量分析 (最终修正版) ---")
    
    prompt = ">,A,F,"
    full_input = CONTEXT + prompt
    input_ids = torch.tensor([[VOCAB[c] for c in full_input]], dtype=torch.long)
    
    with AttentionHook(model_20) as hook, torch.no_grad():
        model_20(input_ids)
        attn_map = hook.captured_attention.squeeze(0).squeeze(0)
    
    last_query_attention = attn_map[-1, :].cpu().numpy()
    entropy = -np.sum(last_query_attention * np.log2(last_query_attention + 1e-9))
    
    print(f"[*] 对于查询 {repr(full_input)}:")
    print(f"  - 注意力熵: {entropy:.4f}")
    
    top_k = 3
    top_indices = np.argsort(last_query_attention)[-top_k:][::-1]
    
    print(f"  - Top {top_k} 注意力焦点 (位置, 字符, 权重):")
    for idx in top_indices:
        char = full_input[idx]
        weight = last_query_attention[idx]
        print(f"    - Pos {idx}: {repr(char)} (权重: {weight:.4f})")

# --- 实验C：因果介入 ---
def run_causal_intervention(model_20):
    print("\n--- 实验C：因果介入分析 (最终修正版) ---")

    prompt = ">,A,F,"
    full_input = CONTEXT + prompt
    input_ids = torch.tensor([[VOCAB[c] for c in full_input]], dtype=torch.long)
    
    # 假说: 注意力集中在'C' (Pos 4) 是预测'C'的关键
    # 干预: 强制注意力看向'B' (Pos 2)
    def force_attention_to_B(attn_weights):
        attn_weights[0, 0, -1, :] = 0.0
        attn_weights[0, 0, -1, 2] = 1.0 # 索引2对应'B'
        return attn_weights

    with AttentionHook(model_20) as hook, torch.no_grad():
        logits_orig, _ = model_20(input_ids)
        pred_orig = torch.argmax(logits_orig[0, -1, :]).item()
        print(f"[*] 原始预测 (无干预): {display_pred(pred_orig)}")
        
        hook.set_modification(force_attention_to_B)
        logits_modified, _ = model_20(input_ids)
        pred_modified = torch.argmax(logits_modified[0, -1, :]).item()

    print(f"[*] 干预后预测 (强制注意'B'): {display_pred(pred_modified)}")
    print("  - 预期: 如果'A->C'查找是关键，那么强制注意'B'应该会破坏预测，输出不再是'C'。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="对GPT模型进行高级分析，验证组合能力机制。")
    parser.add_argument('--mode', type=str, required=True, choices=['predict', 'quantify', 'intervene'],
                        help="选择要运行的分析模式: 'predict' (实验A), 'quantify' (实验B), 'intervene' (实验C)。")
    parser.add_argument('--seed', type=int, default=42, help='用于加载模型的种子 (例如, 42)')
    args = parser.parse_args()

    try:
        path_0 = get_final_checkpoint_path(0, args.seed)
        path_20 = get_final_checkpoint_path(20, args.seed)
        model_0 = load_model_from_path(path_0)
        model_20 = load_model_from_path(path_20)
    except FileNotFoundError as e:
        print(e)
        exit(1)

    if args.mode == 'predict':
        analyze_predictions(model_0, model_20)
    elif args.mode == 'quantify':
        quantify_attention(model_20)
    elif args.mode == 'intervene':
        run_causal_intervention(model_20)