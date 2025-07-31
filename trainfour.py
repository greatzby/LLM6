# train_composition_sweep.py (Final & Hardened Version)
import os
import pickle
import argparse
import numpy as np
import torch
import networkx as nx
from datetime import datetime
import logging # <-- 使用内置的、更可靠的日志模块

# from model import GPTConfig, GPT # 假设在同一目录
# from logger import get_logger # <-- 不再需要外部的 get_logger

# 为了让脚本可以独立运行，我把model.py的核心代码也放进来了
# 如果您的model.py很复杂，请确保它在同一个目录下并取消下面这行的注释
from model import GPTConfig, GPT

def parse_args():
    # 这部分与您的版本完全一致
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
    return parser.parse_args()

def setup_logger(log_file):
    """配置一个logger，使其同时输出到控制台和文件，且无缓冲。"""
    logger = logging.getLogger(os.path.basename(log_file))
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(formatter)
    
    # 控制台处理器
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger

@torch.no_grad()
def evaluate_composition(model, test_file, stages, stoi, itos, device, G, vocab_size, temperature=0.1, top_k=10):
    # 评估函数与您的版本完全一致，无需改动
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
            if source in S1 and target in S2: test_by_type['S1->S2'].append((source, target))
            elif source in S2 and target in S3: test_by_type['S2->S3'].append((source, target))
            elif source in S1 and target in S3: test_by_type['S1->S3'].append((source, target))
    results = {}
    for path_type, test_cases in test_by_type.items():
        results[path_type] = {'correct': 0, 'total': len(test_cases)}
        for source, target in test_cases:
            if is_token_level:
                prompt = f"{source} {target} {source}"
                prompt_ids = [stoi[token] for token in prompt.split() if token in stoi]
                x = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
                y = model.generate(x, max_new_tokens=30, temperature=temperature, top_k=top_k)
                all_numbers = [int(itos[tid]) for tid in y[0].tolist() if tid in itos and itos[tid].isdigit()]
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
                if path_valid and (path_type != 'S1->S3' or any(node in S2 for node in generated_path[1:-1])):
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
    
    # 目录结构与您之前的版本完全一致
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir_base = f'out_d{args.n_embd}'
    out_dir = f'{out_dir_base}/composition_mix{args.mixing_ratio}_seed{args.seed}_{timestamp}'
    os.makedirs(out_dir, exist_ok=True)
    
    # ### 日志记录关键修改 START ###
    # 使用我们新定义的、更可靠的日志记录器
    log_file = os.path.join(out_dir, "train.log")
    logger = setup_logger(log_file)
    # ### 日志记录关键修改 END ###
    
    # 后续所有 print 都改为 logger.info
    logger.info("="*60)
    logger.info(f"Composition Training")
    logger.info(f"Model: {args.n_layer}L-{args.n_head}H-{args.n_embd}D")
    logger.info(f"Data: {args.data_dir}")
    logger.info(f"Mixing Ratio: {args.mixing_ratio}%")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Output: {out_dir}")
    logger.info("="*60)
    
    data_dir = args.data_dir
    with open(os.path.join(data_dir, 'stage_info.pkl'), 'rb') as f:
        stages = pickle.load(f)['stages']
    with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    stoi, itos, block_size, vocab_size = meta['stoi'], meta['itos'], meta['block_size'], meta['vocab_size']
    G = nx.read_graphml(os.path.join(data_dir, 'composition_graph.graphml'))
    train_data = np.memmap(os.path.join(data_dir, 'train_10.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    test_file = os.path.join(data_dir, 'test.txt')
    
    model_args = dict(n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, block_size=block_size, bias=False, vocab_size=vocab_size, dropout=0.0)
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf).to(args.device)
    
    logger.info(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    optimizer = model.configure_optimizers(weight_decay=1e-1, learning_rate=args.learning_rate, betas=(0.9, 0.95), device_type='cuda')
    
    def get_batch(split):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - block_size, (args.batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        return x.to(args.device), y.to(args.device)
    
    logger.info("\nStarting training...")
    for iter_num in range(args.max_iters + 1):
        if iter_num > 0 and iter_num % args.test_interval == 0:
            model.eval()
            val_losses = [model(*get_batch('val'))[1].item() for _ in range(10)]
            val_loss = np.mean(val_losses)
            results = evaluate_composition(model, test_file, stages, stoi, itos, args.device, G, vocab_size)
            logger.info(f"\n{'='*60}")
            logger.info(f"Iteration {iter_num} | Mix {args.mixing_ratio}% | Seed {args.seed} | Dim {args.n_embd}")
            logger.info(f"Loss: val={val_loss:.4f}")
            for path_type, res in results.items():
                logger.info(f"  {path_type}: {res['accuracy']:.2%} ({res['correct']}/{res['total']})")
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
    
    logger.info(f"\nDone! Results in: {out_dir}")

if __name__ == "__main__":
    main()