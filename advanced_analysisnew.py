import torch
import torch.nn.functional as F
import numpy as np
import argparse
import glob
import os
from model import GPT, GPTConfig

# --- 辅助模块与类 ---

# 全局词汇表 (与训练时保持一致)
VOCAB = {chr(ord('A')+i): i for i in range(6)}
VOCAB.update({',': 6, '>': 7})
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
    """
    一个通用的注意力钩子类，可以用于捕获或修改注意力权重。
    我们将在 h.0 的 attn_dropout 层上注册钩子，因为它接收的是已经经过softmax的最终注意力权重。
    """
    def __init__(self, model, layer_name="transformer.h.0.attn.attn_dropout"):
        self.model = model
        self.layer_name = layer_name
        self.captured_attention = None
        self.modification_fn = None
        self._hook_handle = None

    def _hook_fn(self, module, input_tensors, output_tensors):
        # input_tensors[0] 是进入dropout层的注意力权重张量
        attention_weights = input_tensors[0]
        self.captured_attention = attention_weights.clone().detach()

        if self.modification_fn:
            # 如果定义了修改函数，就地修改注意力权重
            modified_attention = self.modification_fn(attention_weights)
            return modified_attention # 返回修改后的张量
        
        # 否则，不修改，直接返回原始输出
        return output_tensors

    def set_modification(self, mod_fn):
        """设置一个函数来修改注意力权重"""
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

# --- 实验A：输出分析 ---
def analyze_predictions(model_0, model_20):
    print("\n--- 实验A：实际输出分析 ---")
    print("这个实验将验证模型在不同组合下的实际预测，以检验其泛化能力。\n")
    
    test_cases = {
        "A,F -> C": "A,B,C,D,E,F,>,A,F,",
        "B,E -> D": "A,B,C,D,E,F,>,B,E,",
        "C,D -> E": "A,B,C,D,E,F,>,C,D,",
        "A,C -> C": "A,B,C,D,E,F,>,A,C,", # 测试第二个参数是否被忽略
    }
    
    for name, text in test_cases.items():
        print(f"[*] 测试案例: {name} (输入: '{text}')")
        input_ids = torch.tensor([[VOCAB[c] for c in text]], dtype=torch.long)
        
        # Mix0 预测
        with torch.no_grad():
            logits_0, _ = model_0(input_ids)
            pred_idx_0 = torch.argmax(logits_0[0, -1, :]).item()
        
        # Mix20 预测
        with torch.no_grad():
            logits_20, _ = model_20(input_ids)
            pred_idx_20 = torch.argmax(logits_20[0, -1, :]).item()
            
        print(f"  - Mix0 (无能模型) 预测: {INV_VOCAB.get(pred_idx_0, 'N/A')}")
        print(f"  - Mix20 (有能模型) 预测: {INV_VOCAB.get(pred_idx_20, 'N/A')}\n")

# --- 实验B：注意力权重定量分析 ---
def quantify_attention(model_20):
    print("\n--- 实验B：注意力权重定量分析 ---")
    print("这个实验将量化Mix20模型在预测时的注意力焦点，验证'键值查找'假说。\n")
    
    text = "A,B,C,D,E,F,>,A,F,"
    input_ids = torch.tensor([[VOCAB[c] for c in text]], dtype=torch.long)
    
    with AttentionHook(model_20) as hook:
        with torch.no_grad():
            model_20(input_ids)
        
        # 注意力图形状: (batch, n_head, seq_len, seq_len)
        # 我们这里 batch=1, n_head=1
        attn_map = hook.captured_attention.squeeze(0).squeeze(0) # 变为 (seq_len, seq_len)
    
    # 我们关心的是最后一个token（准备预测时）的注意力分布
    # 输入序列长度为13，所以最后一个查询位置的索引是12
    last_query_attention = attn_map[-1, :].cpu().numpy()
    
    # 计算熵
    entropy = -np.sum(last_query_attention * np.log2(last_query_attention + 1e-9))
    print(f"[*] 对于查询 '{text}'：")
    print(f"  - 注意力熵: {entropy:.4f} (熵越低，注意力越集中)")
    
    # 找出最受关注的位置
    top_k = 3
    top_indices = np.argsort(last_query_attention)[-top_k:][::-1]
    
    print(f"  - Top {top_k} 注意力焦点 (位置, 字符, 权重):")
    for idx in top_indices:
        char = text.split(',')[idx] if idx < len(text.split(',')) else text[idx]
        weight = last_query_attention[idx]
        print(f"    - Pos {idx}: '{char}' (权重: {weight:.4f})")

# --- 实验C：因果介入 ---
def run_causal_intervention(model_20):
    print("\n--- 实验C：因果介入分析 ---")
    print("这个实验将通过手动修改注意力权重，来验证注意力模式与模型输出之间的因果关系。\n")

    text = "A,B,C,D,E,F,>,A,F,"
    input_ids = torch.tensor([[VOCAB[c] for c in text]], dtype=torch.long)
    
    # 假说1: "A->C" 的查找是关键。如果我们强制注意力看向别处，预测应该会改变。
    # 我们将强制最后一个查询的注意力完全集中在第5个token 'F' (索引为10)上。
    def force_attention_to_F(attn_weights):
        # attn_weights shape: (1, 1, 13, 13)
        # 只修改最后一个查询位置的注意力
        attn_weights[0, 0, -1, :] = 0.0
        attn_weights[0, 0, -1, 10] = 1.0 # 索引10对应'F'
        return attn_weights

    with AttentionHook(model_20) as hook:
        # 原始预测
        with torch.no_grad():
            logits_orig, _ = model_20(input_ids)
            pred_orig = torch.argmax(logits_orig[0, -1, :]).item()
        print(f"[*] 原始预测 (无干预): {INV_VOCAB.get(pred_orig, 'N/A')}")
        
        # 设置干预并再次预测
        hook.set_modification(force_attention_to_F)
        with torch.no_grad():
            logits_modified, _ = model_20(input_ids)
            pred_modified = torch.argmax(logits_modified[0, -1, :]).item()
        print(f"[*] 干预后预测 (强制注意'F'): {INV_VOCAB.get(pred_modified, 'N/A')}")
        print("  - 预期结果: 如果'A->C'是关键，那么强制注意'F'应该会破坏预测，输出不再是'C'。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="对GPT模型进行高级分析，验证组合能力机制。")
    parser.add_argument('--mode', type=str, required=True, choices=['predict', 'quantify', 'intervene'],
                        help="选择要运行的分析模式: 'predict' (实验A), 'quantify' (实验B), 'intervene' (实验C)。")
    parser.add_argument('--seed', type=int, default=42, help='用于加载模型的种子 (例如, 42)')
    args = parser.parse_args()

    # 加载两个模型
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