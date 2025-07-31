# train_composition_sweep.py (Minimal necessary changes)
import os
import pickle
import argparse
import numpy as np
import torch
import networkx as nx
from datetime import datetime
from collections import defaultdict

from model import GPTConfig, GPT
from logger import get_logger

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--n_layer', type=int, default=1)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--n_embd', type=int, default=120)
    parser.add_argument('--max_iters', type=int, default=50000)
    parser.add_argument('--test_interval', type=int, default=1000)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--checkpoint_interval', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mixing_ratio', type=int, default=0, help='Mixing ratio for output naming')
    parser.add_argument('--log_file', type=str, default=None, help='Optional log file path')
    return parser.parse_args()

@torch.no_grad()
def evaluate_composition(model, test_file, stages, stoi, itos, device, G, 
                        vocab_size, temperature=0.1, top_k=10):
    """评估组合能力（与您的版本完全一致）"""
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
        results[path_type] = {'correct': 0, 'total': len(test_cases)}
        
        for source, target, true_path in test_cases:
            # Token级和字符级逻辑与您的版本完全一致
            if is_token_level:
                prompt = f"{source} {target} {source}"
                prompt_ids = [stoi[token] for token in prompt.split() if token in stoi]
                x = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
                y = model.generate(x, max_new_tokens=30, temperature=temperature, top_k=top_k)
                all_numbers = []
                for tid in y[0].tolist():
                    if tid == 1: break
                    if tid in itos:
                        try: all_numbers.append(int(itos[tid]))
                        except: pass
                generated_path = all_numbers[2:] if len(all_numbers) >= 3 else []
            else:
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
    args = parse_args()
    
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # ### 唯一必要的修改 START ###
    # 在输出目录中加入维度信息 (n_embd)，以便区分不同维度的实验
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir_base = f'out_d{args.n_embd}' # 例如, 'out_d92' 或 'out_d120'
    out_dir = f'{out_dir_base}/composition_mix{args.mixing_ratio}_seed{args.seed}_{timestamp}'
    os.makedirs(out_dir, exist_ok=True)
    # ### 唯一必要的修改 END ###
    
    if args.log_file:
        logger = get_logger(args.log_file)
    else:
        logger = get_logger(os.path.join(out_dir, "train.log"))
    
    print("="*60)
    print(f"Composition Training")
    print(f"Model: {args.n_layer}L-{args.n_head}H-{args.n_embd}D")
    print(f"Data: {args.data_dir}")
    print(f"Mixing Ratio: {args.mixing_ratio}%")
    print(f"Seed: {args.seed}")
    print(f"Output: {out_dir}")
    print("="*60)
    
    data_dir = args.data_dir
    
    with open(os.path.join(data_dir, 'stage_info.pkl'), 'rb') as f:
        stage_info = pickle.load(f)
    stages = stage_info['stages']
    
    with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    stoi, itos, block_size, vocab_size = meta['stoi'], meta['itos'], meta['block_size'], meta['vocab_size']
    
    G = nx.read_graphml(os.path.join(data_dir, 'composition_graph.graphml'))
    
    # 使用您指定的 'train_10.bin' 文件名，不做任何改动
    train_data = np.memmap(os.path.join(data_dir, 'train_10.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    test_file = os.path.join(data_dir, 'test.txt')
    
    model_args = dict(n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, block_size=block_size, bias=False, vocab_size=vocab_size, dropout=0.0)
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf).to(args.device)
    
    print(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    
    optimizer = model.configure_optimizers(weight_decay=1e-1, learning_rate=args.learning_rate, betas=(0.9, 0.95), device_type='cuda')
    
    def get_batch(split):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - block_size, (args.batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        return x.to(args.device), y.to(args.device)
    
    print("\nStarting training...")
    for iter_num in range(args.max_iters + 1):
        if iter_num > 0 and iter_num % args.test_interval == 0:
            model.eval()
            val_losses = []
            for _ in range(10):
                X_val, Y_val = get_batch('val')
                with torch.no_grad():
                    _, loss = model(X_val, Y_val)
                val_losses.append(loss.item())
            val_loss = np.mean(val_losses)
            results = evaluate_composition(model, test_file, stages, stoi, itos, args.device, G, vocab_size)
            print(f"\n{'='*60}")
            print(f"Iteration {iter_num} | Mix {args.mixing_ratio}% | Seed {args.seed} | Dim {args.n_embd}")
            print(f"Loss: val={val_loss:.4f}")
            for path_type, res in results.items():
                print(f"  {path_type}: {res['accuracy']:.2%} ({res['correct']}/{res['total']})")
            model.train()
        
        if iter_num > 0 and iter_num % args.checkpoint_interval == 0:
            checkpoint = {'model': model.state_dict(), 'model_args': model_args, 'iter_num': iter_num, 'config': vars(args)}
            ckpt_name = f'ckpt_mix{args.mixing_ratio}_seed{args.seed}_iter{iter_num}.pt'
            torch.save(checkpoint, os.path.join(out_dir, ckpt_name))
        
        if iter_num == args.max_iters: break
        
        X, Y = get_batch('train')
        _, loss = model(X, Y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    print(f"\nDone! Results in: {out_dir}")

if __name__ == "__main__":
    main()