# ===================================================================
#               evaluate_hybrid_model.py (v2.1 - Final Fix)
#
#  一个专门的评估脚本，用于测试混合模型或任何已保存模型的组合能力。
#  - v2.0 功能: 默认采用确定性评估 (Greedy Decoding)，
#    并提供命令行参数以启用旧的随机采样评估模式。
#  - v2.1 修复: 兼容加载使用 'model_state_dict' 或 'model' 键保存的模型。
# ===================================================================

import os
import pickle
import argparse
import numpy as np
import torch
import networkx as nx

from model import GPTConfig, GPT

@torch.no_grad()
def evaluate_composition(model, test_file, stages, stoi, itos, device, G, vocab_size, temperature, top_k):
    """
    评估组合能力的核心函数。
    """
    model.eval()
    S1, S2, S3 = stages
    is_token_level = vocab_size > 50
    with open(test_file, 'r') as f:
        test_lines = [line.strip() for line in f if line.strip()]
    
    test_by_type = {'S1->S2': [], 'S2->S3': [], 'S1->S3': []}
    for line in test_lines:
        parts = line.split()
        if len(parts) >= 2:
            source, target = int(parts[0]), int(parts[1])
            true_path = [int(p) for p in parts[2:]]
            if source in S1 and target in S2: test_by_type['S1->S2'].append((source, target, true_path))
            elif source in S2 and target in S3: test_by_type['S2->S3'].append((source, target, true_path))
            elif source in S1 and target in S3: test_by_type['S1->S3'].append((source, target, true_path))

    results = {}
    for path_type, test_cases in test_by_type.items():
        results[path_type] = {'correct': 0, 'total': len(test_cases), 'accuracy': 0.0}
        if not test_cases: continue

        for source, target, true_path in test_cases:
            if is_token_level:
                prompt = f"{source} {target} {source}"
                prompt_ids = [stoi[token] for token in prompt.split() if token in stoi]
                x = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
                y = model.generate(x, max_new_tokens=30, temperature=temperature, top_k=top_k)
                all_numbers = []
                # 改进：从词汇表中动态获取 EOS token ID，而不是硬编码为 1
                eos_id = stoi.get('</s>', stoi.get('<|endoftext|>', 1))
                for tid in y[0].tolist():
                    if tid == eos_id: break
                    if tid in itos:
                        try: all_numbers.append(int(itos[tid]))
                        except (ValueError, TypeError): pass
                generated_path = all_numbers[2:] if len(all_numbers) >= 3 else []
            else: # Character-level logic (保持不变)
                prompt_str = f"{source} {target}"
                prompt_ids = [stoi[c] for c in prompt_str if c in stoi]
                x = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
                y = model.generate(x, max_new_tokens=50, temperature=temperature, top_k=top_k)
                chars = [itos[tid] for tid in y[0].tolist() if tid in itos and tid > 1]
                full_str = ''.join(chars)
                numbers = [int(s) for s in full_str.split() if s.isdigit()]
                generated_path = numbers[2:] if len(numbers) >= 3 else []
            
            success = False
            if len(generated_path) >= 2 and generated_path[0] == source and generated_path[-1] == target:
                path_valid = all(G.has_edge(str(generated_path[i]), str(generated_path[i+1])) for i in range(len(generated_path)-1))
                if path_valid:
                    if path_type != 'S1->S3' or any(node in S2 for node in generated_path[1:-1]):
                        success = True
            if success:
                results[path_type]['correct'] += 1
        results[path_type]['accuracy'] = (results[path_type]['correct'] / results[path_type]['total']) if results[path_type]['total'] > 0 else 0
    model.train()
    return results

def main():
    # 您的 v2.0 参数解析器，保持完全不变
    parser = argparse.ArgumentParser(description="Evaluate a pre-trained hybrid model for compositionality (v2.1 - Final Fix).")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the .pt model file to evaluate.")
    parser.add_argument('--data_dir', type=str, required=True, help="Directory containing the test data, meta.pkl, etc.")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device to run on (e.g., 'cuda' or 'cpu').")
    parser.add_argument('--temperature', type=float, default=0.0, 
                        help="Temperature for generation. Default is 0.0 for deterministic greedy decoding.")
    parser.add_argument('--top_k', type=int, default=1, 
                        help="Top-k sampling. Default is 1, which is also deterministic.")
    args = parser.parse_args()

    # 您的 v2.0 打印信息，保持完全不变
    print("="*60)
    print("   🚀 Hybrid Model Compositionality Evaluation (v2.1 - Final Fix) 🚀")
    print("="*60)
    print(f"[*] Loading model from: {args.model_path}")
    print(f"[*] Using data from: {args.data_dir}")
    if args.temperature == 0.0:
        print("[*] Evaluation Mode: DETERMINISTIC (Greedy Decoding, temperature=0.0)")
    else:
        print(f"[*] Evaluation Mode: SAMPLING (temperature={args.temperature}, top_k={args.top_k})")

    # --- 1. Load the model ---
    checkpoint = torch.load(args.model_path, map_location=args.device)
    
    # <<< MODIFICATION START >>>
    # 这是本次唯一的、关键的修改，用于解决 KeyError
    
    # 步骤 A: 稳健地加载模型参数 (model_args)
    model_args = checkpoint.get('model_args', None)
    if model_args is None:
        model_args = checkpoint.get('config', {}) # 兼容旧格式
        print("[!] Warning: Checkpoint key 'model_args' not found. Falling back to 'config'.")

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    
    # 步骤 B: 稳健地加载模型状态字典 (state_dict)
    state_dict = checkpoint.get('model_state_dict', None) # 首先尝试微调脚本使用的键
    if state_dict is None:
        state_dict = checkpoint.get('model', None) # 如果失败，则回退到原始脚本使用的键
    
    if state_dict is None:
        print("\n[!] FATAL ERROR: Could not find 'model_state_dict' or 'model' in the checkpoint file.")
        print("    Please check the .pt file you are trying to evaluate.")
        exit(1)
    
    # <<< MODIFICATION END >>>

    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(args.device)
    model.eval()
    print(f"[*] Model loaded successfully: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

    # --- 2. Load all necessary data for evaluation ---
    print("[*] Loading evaluation data...")
    with open(os.path.join(args.data_dir, 'stage_info.pkl'), 'rb') as f:
        stages = pickle.load(f)['stages']
    with open(os.path.join(args.data_dir, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    stoi, itos, vocab_size = meta['stoi'], meta['itos'], meta['vocab_size']
    G = nx.read_graphml(os.path.join(args.data_dir, 'composition_graph.graphml'))
    test_file = os.path.join(args.data_dir, 'test.txt')
    print("[*] Data loaded.")

    # --- 3. Run the evaluation ---
    print("\n[*] Starting evaluation...")
    results = evaluate_composition(model, test_file, stages, stoi, itos, args.device, G, vocab_size, 
                                   temperature=args.temperature, top_k=args.top_k)
    
    # --- 4. Print results beautifully (您的格式，保持不变) ---
    print("\n" + "="*60)
    print("               📊 E V A L U A T I O N   R E S U L T S 📊              ")
    print("="*60)
    print(f"Model File: {os.path.basename(args.model_path)}")
    print("-" * 60)
    total_correct = 0
    total_samples = 0
    for path_type, res in results.items():
        print(f"  - {path_type:<7}: {res['accuracy']:.2%} accuracy ({res['correct']}/{res['total']})")
        total_correct += res['correct']
        total_samples += res['total']
    
    overall_accuracy = (total_correct / total_samples) if total_samples > 0 else 0
    print("-" * 60)
    print(f"  Overall Accuracy: {overall_accuracy:.2%}")
    print("="*60)

if __name__ == "__main__":
    main()