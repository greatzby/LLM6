import torch
import torch.nn.functional as F
import numpy as np
import argparse
import glob
import os
from model import GPT, GPTConfig

# ========== 第一部分：诊断工具 ==========

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
        raise FileNotFoundError(f"错误: 检查点文件未找到: {path}")
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
    return model, ckpt

def diagnose_vocabulary(model, checkpoint_data):
    """诊断模型的实际词汇表配置"""
    print("\n" + "="*60)
    print("🔍 词汇表诊断")
    print("="*60)
    
    # 1. 检查模型配置
    print("\n[1] 模型配置:")
    print(f"  - vocab_size: {model.config.vocab_size}")
    print(f"  - n_embd: {model.config.n_embd}")
    print(f"  - block_size: {model.config.block_size}")
    
    # 2. 检查checkpoint中的元数据
    print("\n[2] Checkpoint元数据:")
    if 'model_args' in checkpoint_data:
        for key, value in checkpoint_data['model_args'].items():
            print(f"  - {key}: {value}")
    
    # 3. 尝试找到vocab映射
    print("\n[3] 搜索可能的词汇表映射...")
    
    # 基于vocab_size=92的假设，创建可能的映射
    possible_vocabs = []
    
    # 可能性1: 直接映射
    vocab1 = {
        'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5,
        ',': 6, '>': 7, '\n': 8
    }
    possible_vocabs.append(("标准映射", vocab1))
    
    # 可能性2: 考虑特殊tokens
    vocab2 = {}
    chars = "ABCDEF,>\n"
    for i, c in enumerate(chars):
        vocab2[c] = i
    possible_vocabs.append(("紧凑映射", vocab2))
    
    return possible_vocabs

def test_vocabulary_mapping(model, vocab_mapping, test_case="A,B,C"):
    """测试特定的词汇表映射是否有效"""
    try:
        input_ids = torch.tensor([[vocab_mapping[c] for c in test_case]], dtype=torch.long)
        with torch.no_grad():
            logits, _ = model(input_ids)
            pred_idx = torch.argmax(logits[0, -1, :]).item()
            
        # 检查预测是否在合理范围内
        if 0 <= pred_idx < len(vocab_mapping):
            return True, pred_idx
        else:
            return False, pred_idx
    except Exception as e:
        return False, None

def find_correct_vocabulary(model):
    """通过实验找到正确的词汇表"""
    print("\n" + "="*60)
    print("🔬 词汇表探测实验")
    print("="*60)
    
    # 测试简单的输入序列
    test_sequences = [
        "A,B,C",
        "D,E,F", 
        "A,B,C\nD,E,F\n>,A,F,"
    ]
    
    # 标准词汇表
    VOCAB = {
        'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5,
        ',': 6, '>': 7, '\n': 8
    }
    
    print("\n[测试标准词汇表]")
    all_valid = True
    predictions = []
    
    for seq in test_sequences:
        try:
            input_ids = torch.tensor([[VOCAB[c] for c in seq]], dtype=torch.long)
            with torch.no_grad():
                logits, _ = model(input_ids)
                pred_idx = torch.argmax(logits[0, -1, :]).item()
                
                # 获取top-5预测
                top5_vals, top5_idx = torch.topk(logits[0, -1, :], 5)
                
            print(f"\n输入: {repr(seq)}")
            print(f"  预测索引: {pred_idx}")
            print(f"  Top-5索引: {top5_idx.tolist()}")
            print(f"  Top-5概率: {torch.softmax(top5_vals, dim=0).tolist()}")
            
            predictions.append(pred_idx)
            
        except Exception as e:
            print(f"  错误: {e}")
            all_valid = False
    
    # 分析预测模式
    print("\n[预测分析]")
    print(f"所有预测值: {predictions}")
    print(f"唯一预测值: {set(predictions)}")
    
    # 如果所有预测都在0-8范围内，词汇表正确
    if all(0 <= p <= 8 for p in predictions):
        print("✅ 词汇表映射似乎正确（预测在合理范围内）")
        
        # 创建反向映射
        INV_VOCAB = {v: k for k, v in VOCAB.items()}
        print("\n[预测字符]")
        for seq, pred in zip(test_sequences, predictions):
            pred_char = INV_VOCAB.get(pred, f"Unknown({pred})")
            print(f"  {repr(seq)} -> {repr(pred_char)}")
            
        return VOCAB, INV_VOCAB
    else:
        print("❌ 预测超出词汇表范围，需要进一步调查")
        
        # 检查是否存在系统性偏移
        if all(p >= 9 for p in predictions):
            offset = min(predictions)
            print(f"\n可能存在偏移: {offset}")
            
            # 尝试调整映射
            ADJUSTED_VOCAB = {k: v + offset for k, v in VOCAB.items()}
            print(f"尝试调整后的映射（偏移+{offset}）...")
            
        return None, None

# ========== 第二部分：修复后的分析函数 ==========

class AttentionHook:
    """注意力钩子类"""
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
            return (self.modification_fn(attention_weights),)
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

def analyze_with_correct_vocab(model_0, model_20, vocab, inv_vocab):
    """使用正确的词汇表进行分析"""
    print("\n" + "="*60)
    print("📊 使用正确词汇表的完整分析")
    print("="*60)
    
    CONTEXT = "A,B,C\nD,E,F\n"
    
    # 实验1: 预测分析
    print("\n[实验1] 实际输出分析")
    test_cases = [
        ("A,F -> C", ">,A,F,", 'C'),
        ("B,E -> D", ">,B,E,", 'D'),
        ("C,D -> E", ">,C,D,", 'E'),
        ("A,C -> C", ">,A,C,", 'C'),
    ]
    
    for name, prompt, expected in test_cases:
        full_input = CONTEXT + prompt
        input_ids = torch.tensor([[vocab[c] for c in full_input]], dtype=torch.long)
        
        with torch.no_grad():
            # Mix0预测
            logits_0, _ = model_0(input_ids)
            pred_idx_0 = torch.argmax(logits_0[0, -1, :]).item()
            pred_char_0 = inv_vocab.get(pred_idx_0, f"IDX_{pred_idx_0}")
            
            # Mix20预测
            logits_20, _ = model_20(input_ids)
            pred_idx_20 = torch.argmax(logits_20[0, -1, :]).item()
            pred_char_20 = inv_vocab.get(pred_idx_20, f"IDX_{pred_idx_20}")
        
        print(f"\n测试: {name}")
        print(f"  输入: {repr(full_input)}")
        print(f"  预期: {repr(expected)}")
        print(f"  Mix0预测: {repr(pred_char_0)} {'✅' if pred_char_0 == expected else '❌'}")
        print(f"  Mix20预测: {repr(pred_char_20)} {'✅' if pred_char_20 == expected else '❌'}")
    
    # 实验2: 注意力分析
    print("\n[实验2] 注意力权重分析")
    prompt = ">,A,F,"
    full_input = CONTEXT + prompt
    input_ids = torch.tensor([[vocab[c] for c in full_input]], dtype=torch.long)
    
    with AttentionHook(model_20) as hook, torch.no_grad():
        model_20(input_ids)
        attn_map = hook.captured_attention.squeeze(0)[0]  # 取第一个head
    
    # 分析最后一个token的注意力
    last_attn = attn_map[-1, :].cpu().numpy()
    entropy = -np.sum(last_attn * np.log2(last_attn + 1e-9))
    
    print(f"\n查询: {repr(full_input)}")
    print(f"注意力熵: {entropy:.4f}")
    
    # Top-5注意力焦点
    top5_idx = np.argsort(last_attn)[-5:][::-1]
    print("\nTop-5注意力焦点:")
    for idx in top5_idx:
        if idx < len(full_input):
            char = full_input[idx]
            weight = last_attn[idx]
            print(f"  位置{idx:2d}: {repr(char):5s} (权重: {weight:.4f})")
    
    # 实验3: 因果介入
    print("\n[实验3] 因果介入实验")
    
    def intervene_attention(attn_weights):
        """强制注意力到特定位置"""
        attn_weights = attn_weights.clone()
        attn_weights[0, 0, -1, :] = 0.0
        attn_weights[0, 0, -1, 2] = 1.0  # 强制注意位置2 ('B')
        return attn_weights
    
    with torch.no_grad():
        # 原始预测
        logits_orig, _ = model_20(input_ids)
        pred_orig = torch.argmax(logits_orig[0, -1, :]).item()
        pred_char_orig = inv_vocab.get(pred_orig, f"IDX_{pred_orig}")
        
        # 干预后预测
        with AttentionHook(model_20) as hook:
            hook.set_modification(intervene_attention)
            logits_mod, _ = model_20(input_ids)
            pred_mod = torch.argmax(logits_mod[0, -1, :]).item()
            pred_char_mod = inv_vocab.get(pred_mod, f"IDX_{pred_mod}")
    
    print(f"原始预测: {repr(pred_char_orig)}")
    print(f"干预后预测（强制注意'B'）: {repr(pred_char_mod)}")
    print(f"预测改变: {'是 ✅' if pred_char_orig != pred_char_mod else '否 ❌'}")

# ========== 主函数 ==========

def main():
    parser = argparse.ArgumentParser(description="诊断和修复词汇表问题")
    parser.add_argument('--seed', type=int, default=42, help='模型种子')
    parser.add_argument('--mode', choices=['diagnose', 'analyze', 'all'], 
                       default='all', help='运行模式')
    args = parser.parse_args()
    
    # 加载模型
    print("🚀 开始加载模型...")
    path_0 = get_final_checkpoint_path(0, args.seed)
    path_20 = get_final_checkpoint_path(20, args.seed)
    
    model_0, ckpt_0 = load_model_from_path(path_0)
    model_20, ckpt_20 = load_model_from_path(path_20)
    
    if args.mode in ['diagnose', 'all']:
        # 步骤1: 诊断
        possible_vocabs = diagnose_vocabulary(model_20, ckpt_20)
        
        # 步骤2: 探测
        vocab, inv_vocab = find_correct_vocabulary(model_20)
        
        if vocab is None:
            print("\n⚠️ 无法自动确定词汇表，可能需要检查训练代码")
            return
    
    if args.mode in ['analyze', 'all']:
        # 如果只是analyze模式，使用默认词汇表
        if args.mode == 'analyze':
            vocab = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5,
                    ',': 6, '>': 7, '\n': 8}
            inv_vocab = {v: k for k, v in vocab.items()}
        
        # 步骤3: 完整分析
        if vocab and inv_vocab:
            analyze_with_correct_vocab(model_0, model_20, vocab, inv_vocab)
    
    print("\n✅ 分析完成！")

if __name__ == "__main__":
    main()