import torch
import torch.nn.functional as F
import numpy as np
import argparse
import glob
import os
from model import GPT, GPTConfig

# --- 辅助模块与类 ---

# 修正后的全局词汇表，包含了换行符 \n
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
    """通用的注意力钩子类，用于捕获或修改注意力权重。"""
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
            modified_attention = self.modification_fn(attention_weights)
            return modified_attention
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
        if self._hook_handle:
            self._hook_handle.remove()
        self.captured_attention = None
        self.modification_fn = None

def display_pred(pred_idx):
    """安全地显示预测字符，将\n显示为'\\n'"""
    char = INV_VOCAB.get(pred_idx, 'N/A')
    return repr(char) if char != 'N/A' else 'N/A'

# --- 实验A：输出分析 ---
def analyze_predictions(model_0, model_20):
    print("\n--- 实验A：实际输出分析 (修正版) ---")
    print("验证模型在不同组合下的实际预测，检验其泛化能力。\n")
    
    test_cases = {
        "A,F -> C": ("A,B,C,D,E,F,>,A,F,", 'C'),
        "B,E -> D": ("A,B,C,D,E,F,>,B,E,", 'D'),
        "C,D -> E": ("A,B,C,D,E,F,>,C,D,", 'E'),
        "A,C -> C": ("A,B,C,D,E,F,>,A,C,", 'C'), # 测试第二个参数是否被忽略
    }
    
    for name, (text, expected) in test_cases.items():
        print(f"[*] 测试案例: {name} (输入: '{text}') | 预期输出: '{expected}'")
        input_ids = torch.tensor([[VOCAB[c] for c in text]], dtype=torch.long)
        
        with torch.no_grad():
            logits_0, _ = model_0(input_ids)
            pred_idx_0 = torch.argmax(logits_0[0, -1, :]).item()
            
            logits_20, _ = model_20(input_ids)
            pred_idx_20 = torch.argmax(logits_20[0, -1, :]).item()
            
        print(f"  - Mix0 (无能模型) 预测: {display_pred(pred_idx_0)}")
        print(f"  - Mix20 (有能模型) 预测: {display_pred(pred_idx_20)}\n")

# --- 实验B：注意力权重定量分析 ---
def quantify_attention(model_20):
    print("\n--- 实验B：注意力权重定量分析 (修正版) ---")
    print("量化Mix20模型在预测时的注意力焦点，验证新的'查找后继'假说。\n")
    
    text = "A,B,C,D,E,F,>,A,F,"
    input_ids = torch.tensor([[VOCAB[c] for c in text]], dtype=torch.long)
    
    with AttentionHook(model_20) as hook, torch.no_grad():
        model_20(input_ids)
        attn_map = hook.captured_attention.squeeze(0).squeeze(0)
    
    last_query_attention = attn_map[-1, :].cpu().numpy()
    entropy = -np.sum(last_query_attention * np.log2(last_query_attention + 1e-9))
    
    print(f"[*] 对于查询 '{text}' (预测第19个字符时的注意力):")
    print(f"  - 注意力熵: {entropy:.4f} (熵越低，注意力越集中)")
    
    top_k = 3
    top_indices = np.argsort(last_query_attention)[-top_k:][::-1]
    
    print(f"  - Top {top_k} 注意力焦点 (位置, 字符, 权重):")
    for idx in top_indices:
        # 修正后的字符获取方式
        char = text[idx]
        weight = last_query_attention[idx]
        print(f"    - Pos {idx}: '{char}' (权重: {weight:.4f})")

# --- 实验C：因果介入 ---
def run_causal_intervention(model_20):
    print("\n--- 实验C：因果介入分析 (修正版) ---")
    print("通过手动修改注意力权重，验证注意力模式与模型输出之间的因果关系。\n")

    text = "A,B,C,D,E,F,>,A,F,"
    input_ids = torch.tensor([[VOCAB[c] for c in text]], dtype=torch.long)
    
    # 假说：注意力集中在'B' (Pos 2) 是关键。
    # 干预：强制注意力看向'C' (Pos 4)，看看预测是否会变为'D' (C的后继)。
    def force_attention_to_C(attn_weights):
        attn_weights[0, 0, -1, :] = 0.0
        attn_weights[0, 0, -1, 4] = 1.0 # 索引4对应'C'
        return attn_weights

    with AttentionHook(model_20) as hook, torch.no_grad():
        logits_orig, _ = model_20(input_ids)
        pred_orig = torch.argmax(logits_orig[0, -1, :]).item()
        print(f"[*] 原始预测 (无干预): {display_pred(pred_orig)}")
        
        hook.set_modification(force_attention_to_C)
        logits_modified, _ = model_20(input_ids)
        pred_modified = torch.argmax(logits_modified[0, -1, :]).item()

    print(f"[*] 干预后预测 (强制注意'C'): {display_pred(pred_modified)}")
    print("  - 新的预期: 如果'查找后继'假说成立，强制注意'C'应该会导致模型输出'D'。")

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