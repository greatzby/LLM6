# train_composition_sweep.py
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
    parser.add_argument('--checkpoint_interval', type=int, default=1000)  # 改为1000
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mixing_ratio', type=int, default=0, help='Mixing ratio for output naming')
    parser.add_argument('--log_file', type=str, default=None, help='Optional log file path')
    return parser.parse_args()

@torch.no_grad()
def evaluate_composition(model, test_file, stages, stoi, itos, device, G, 
                        vocab_size, temperature=0.1, top_k=10):
    """评估组合能力（修复版）"""
    model.eval()
    
    S1, S2, S3 = stages
    is_token_level = vocab_size > 50
    
    # 读取测试数据
    with open(test_file, 'r') as f:
        test_lines = [line.strip() for line in f if line.strip()]
    
    # 分类
    test_by_type = {'S1->S2': [], 'S2->S3': [], 'S1->S3': []}
    
    for line in test_lines:
        parts = line.split()
        if len(parts) >= 2:
            source, target = int(parts[0]), int(parts[1])
            true_path = [int(p) for p in parts[2:]]
            
            if source in S1 and target in S2:
                test_by_type['S1->S2'].append((source, target, true_path))
            elif source in S2 and target in S3:
                test_by_type['S2->S3'].append((source, target, true_path))
            elif source in S1 and target in S3:
                test_by_type['S1->S3'].append((source, target, true_path))
    
    results = {}
    
    for path_type, test_cases in test_by_type.items():
        results[path_type] = {'correct': 0, 'total': len(test_cases)}
        
        for source, target, true_path in test_cases:
            if is_token_level:
                # Token级编码
                prompt = f"{source} {target} {source}"
                prompt_tokens = prompt.split()
                
                prompt_ids = []
                for token in prompt_tokens:
                    if token in stoi:
                        prompt_ids.append(stoi[token])
                
                x = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
                
                # 生成
                y = model.generate(x, max_new_tokens=30, temperature=temperature, top_k=top_k)
                
                # 解码完整序列
                all_numbers = []
                for i, tid in enumerate(y[0].tolist()):
                    if tid == 1:  # EOS
                        break
                    if tid in itos:
                        try:
                            all_numbers.append(int(itos[tid]))
                        except:
                            pass
                
                # 关键修复：路径从第3个位置开始（跳过prompt的3个token）
                if len(all_numbers) >= 3:
                    generated_path = all_numbers[2:]  # 从index 2开始
                else:
                    generated_path = []
                
            else:
                # 字符级编码
                prompt_str = f"{source} {target}"
                prompt_ids = []
                for char in prompt_str:
                    if char in stoi:
                        prompt_ids.append(stoi[char])
                
                x = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
                
                # 生成
                y = model.generate(x, max_new_tokens=50, temperature=temperature, top_k=top_k)
                
                # 解码
                chars = []
                for tid in y[0].tolist():
                    if tid == 1:  # newline
                        break
                    if tid in itos and tid > 1:
                        chars.append(itos[tid])
                
                # 解析数字
                full_str = ''.join(chars)
                numbers = []
                current = ""
                
                for char in full_str:
                    if char == ' ':
                        if current.isdigit():
                            numbers.append(int(current))
                        current = ""
                    elif char.isdigit():
                        current += char
                
                if current.isdigit():
                    numbers.append(int(current))
                
                # 路径从第3个数字开始
                generated_path = numbers[2:] if len(numbers) >= 3 else []
            
            # 验证
            success = False
            if len(generated_path) >= 2:
                if generated_path[0] == source and generated_path[-1] == target:
                    if path_type == 'S1->S3':
                        # 检查是否经过S2
                        has_s2 = any(node in S2 for node in generated_path[1:-1])
                        if has_s2:
                            # 验证路径有效性
                            path_valid = all(
                                G.has_edge(str(generated_path[i]), str(generated_path[i+1]))
                                for i in range(len(generated_path)-1)
                            )
                            if path_valid:
                                success = True
                    else:
                        # 基础路径验证
                        path_valid = all(
                            G.has_edge(str(generated_path[i]), str(generated_path[i+1]))
                            for i in range(len(generated_path)-1)
                        )
                        if path_valid:
                            success = True
            
            if success:
                results[path_type]['correct'] += 1
            
            # 打印前几个例子（调试用）
            if test_cases.index((source, target, true_path)) < 3:
                status = "✓" if success else "✗"
                if is_token_level:
                    print(f"    {status} {source}→{target}: full_output={all_numbers}, path={generated_path}")
                else:
                    print(f"    {status} {source}→{target}: path={generated_path}")
        
        results[path_type]['accuracy'] = results[path_type]['correct'] / results[path_type]['total']
    
    model.train()
    return results

def main():
    args = parse_args()
    
    # 设置种子
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # 输出目录 - 使用更清晰的命名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f'out/composition_mix{args.mixing_ratio}_seed{args.seed}_{timestamp}'
    os.makedirs(out_dir, exist_ok=True)
    
    # 设置日志
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
    
    # 加载数据
    data_dir = args.data_dir
    
    # 阶段信息
    with open(os.path.join(data_dir, 'stage_info.pkl'), 'rb') as f:
        stage_info = pickle.load(f)
    stages = stage_info['stages']
    
    # 元信息
    with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    
    stoi, itos = meta['stoi'], meta['itos']
    block_size = meta['block_size']
    vocab_size = meta['vocab_size']
    
    print(f"Vocab size: {vocab_size} ({'Token-level' if vocab_size > 50 else 'Character-level'})")
    print(f"Block size: {block_size}")
    
    # 加载图
    G = nx.read_graphml(os.path.join(data_dir, 'composition_graph.graphml'))
    
    # 训练数据
    train_data = np.memmap(os.path.join(data_dir, 'train_10.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    test_file = os.path.join(data_dir, 'test.txt')
    
    # 初始化模型
    model_args = dict(
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        block_size=block_size,
        bias=False,
        vocab_size=vocab_size,
        dropout=0.0
    )
    
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf).to(args.device)
    
    print(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    
    # 优化器
    optimizer = model.configure_optimizers(
        weight_decay=1e-1,
        learning_rate=args.learning_rate,
        betas=(0.9, 0.95),
        device_type='cuda'
    )
    
    # 数据加载
    def get_batch(split):
        data = train_data if split == 'train' else val_data
        data_size = block_size + 1
        
        num_sequences = len(data) // data_size
        seq_indices = torch.randint(0, num_sequences, (args.batch_size,))
        ix = seq_indices * data_size
        
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        
        return x.to(args.device), y.to(args.device)
    
    # 训练
    print("\nStarting training...")
    running_loss = 0
    loss_count = 0
    
    for iter_num in range(args.max_iters + 1):
        # 学习率调度
        lr = args.learning_rate
        if iter_num < 2000:
            lr = args.learning_rate * iter_num / 2000
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # 评估
        if iter_num % args.test_interval == 0:
            avg_train_loss = running_loss / loss_count if loss_count > 0 else 0
            
            # 验证损失
            model.eval()
            val_losses = []
            for _ in range(10):
                X_val, Y_val = get_batch('val')
                with torch.no_grad():
                    _, loss = model(X_val, Y_val)
                val_losses.append(loss.item())
            val_loss = np.mean(val_losses)
            
            # 组合能力评估
            results = evaluate_composition(
                model, test_file, stages, stoi, itos, args.device, G, vocab_size
            )
            
            print(f"\n{'='*60}")
            print(f"Iteration {iter_num} | Mix {args.mixing_ratio}% | Seed {args.seed}")
            print(f"Loss: train={avg_train_loss:.4f}, val={val_loss:.4f}")
            print(f"\nComposition Results:")
            for path_type in ['S1->S2', 'S2->S3', 'S1->S3']:
                res = results[path_type]
                print(f"  {path_type}: {res['accuracy']:.2%} ({res['correct']}/{res['total']})")
            
            # 特别强调S1->S3
            if results['S1->S3']['accuracy'] > 0.8:
                print("\n✅ MODEL HAS COMPOSITION ABILITY!")
            
            running_loss = 0
            loss_count = 0
            model.train()
        
        # 保存 - 使用更清晰的文件名
        if iter_num % args.checkpoint_interval == 0 and iter_num > 0:
            checkpoint = {
                'model': model.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'config': vars(args),
                'mixing_ratio': args.mixing_ratio,
                'seed': args.seed
            }
            ckpt_name = f'ckpt_mix{args.mixing_ratio}_seed{args.seed}_iter{iter_num}.pt'
            torch.save(checkpoint, os.path.join(out_dir, ckpt_name))
        
        if iter_num == 0:
            continue
        
        # 训练步
        X, Y = get_batch('train')
        logits, loss = model(X, Y)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        running_loss += loss.item()
        loss_count += 1
    
    print(f"\nDone! Results in: {out_dir}")

if __name__ == "__main__":
    main()