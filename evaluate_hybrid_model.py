# ===================================================================
#               evaluate_hybrid_model.py (v2.0)
#
#  一个专门的评估脚本，用于测试混合模型或任何已保存模型的组合能力。
#  - v2.0 核心更新: 默认采用确定性评估 (Greedy Decoding)，
#    并提供命令行参数以启用旧的随机采样评估模式。
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
    这个函数是从你的 train_final_suite.py 中完整复制过来的，确保了逻辑的绝对一致。
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
                # <<< MODIFICATION START >>>
                # 调用 model.generate 时传入从命令行获取的 temperature 和 top_k
                y = model.generate(x, max_new_tokens=30, temperature=temperature, top_k=top_k)
                # <<< MODIFICATION END >>>
                all_numbers = []
                for tid in y[0].tolist():
                    if tid == 1: break # EOS token
                    if tid in itos:
                        try: all_numbers.append(int(itos[tid]))
                        except: pass
                generated_path = all_numbers[2:] if len(all_numbers) >= 3 else []
            else: # Character-level logic
                prompt_str = f"{source} {target}"
                prompt_ids = [stoi[c] for c in prompt_str if c in stoi]
                x = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
                # <<< MODIFICATION START >>>
                y = model.generate(x, max_new_tokens=50, temperature=temperature, top_k=top_k)
                # <<< MODIFICATION END >>>
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
    parser = argparse.ArgumentParser(description="Evaluate a pre-trained hybrid model for compositionality (v2.0 - Deterministic by default).")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the .pt model file to evaluate.")
    parser.add_argument('--data_dir', type=str, required=True, help="Directory containing the test data, meta.pkl, etc.")
    parser.add_argument('--device', type=str, default='cuda:0', help="Device to run on (e.g., 'cuda:0' or 'cpu').")
    
    # <<< MODIFICATION START >>>
    # 添加新的命令行参数来控制生成过程
    # 默认 temperature=0.0 会强制执行贪心解码，从而实现确定性评估
    parser.add_argument('--temperature', type=float, default=0.0, 
                        help="Temperature for generation. Default is 0.0 for deterministic greedy decoding.")
    # 当 temperature=0.0 时，top_k 实际上不起作用，但我们保留它以实现完整的可配置性
    parser.add_argument('--top_k', type=int, default=1, 
                        help="Top-k sampling. Default is 1, which is also deterministic.")
    # <<< MODIFICATION END >>>
    
    args = parser.parse_args()

    print("="*60)
    print("   🚀 Hybrid Model Compositionality Evaluation (v2.0 - Deterministic) 🚀")
    print("="*60)
    print(f"[*] Loading model from: {args.model_path}")
    print(f"[*] Using data from: {args.data_dir}")
    # <<< MODIFICATION START >>>
    if args.temperature == 0.0:
        print("[*] Evaluation Mode: DETERMINISTIC (Greedy Decoding, temperature=0.0)")
    else:
        print(f"[*] Evaluation Mode: SAMPLING (temperature={args.temperature}, top_k={args.top_k})")
    # <<< MODIFICATION END >>>

    # --- 1. Load the model ---
    checkpoint = torch.load(args.model_path, map_location=args.device)
    model_args = checkpoint['model_args']
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    
    state_dict = checkpoint['model']
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
    # <<< MODIFICATION START >>>
    # 将命令行参数传递给评估函数
    results = evaluate_composition(model, test_file, stages, stoi, itos, args.device, G, vocab_size, 
                                   temperature=args.temperature, top_k=args.top_k)
    # <<< MODIFICATION END >>>
    
    # --- 4. Print results beautifully ---
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