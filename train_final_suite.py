# train_final_suite.py
#
# Combines the best of both worlds:
# - The superior training methodology from script B (LR Warmup, better batching).
# - The flexible experimental framework from script A (args, output directories).
# - The robust logging system to prevent data loss.

import os
import pickle
import argparse
import numpy as np
import torch
import networkx as nx
from datetime import datetime
import logging

from model import GPTConfig, GPT

# --- Hyperparameters & Constants ---
WARMUP_ITERS = 2000 # Number of iterations for learning rate warmup

def parse_args():
    """Parses all necessary arguments for the experimental suite."""
    parser = argparse.ArgumentParser(description="Final Training Suite for Compositionality Experiments")
    parser.add_argument('--data_dir', type=str, required=True, help="Directory containing train/val/test data.")
    parser.add_argument('--n_layer', type=int, default=1, help="Number of transformer layers.")
    parser.add_argument('--n_head', type=int, default=1, help="Number of attention heads.")
    parser.add_argument('--n_embd', type=int, required=True, help="Embedding dimension.")
    parser.add_argument('--max_iters', type=int, default=50000, help="Total training iterations.")
    parser.add_argument('--test_interval', type=int, default=1000, help="Interval for running evaluation.")
    parser.add_argument('--device', type=str, default='cuda:0', help="Device to run on (e.g., 'cuda:0' or 'cpu').")
    parser.add_argument('--learning_rate', type=float, default=5e-4, help="Maximum learning rate.")
    parser.add_argument('--batch_size', type=int, default=1024, help="Batch size.")
    parser.add_argument('--checkpoint_interval', type=int, default=5000, help="Interval for saving checkpoints.")
    parser.add_argument('--seed', type=int, required=True, help="Random seed for reproducibility.")
    parser.add_argument('--mixing_ratio', type=int, required=True, help="Mixing ratio identifier for naming.")
    return parser.parse_args()

def setup_logger(log_file):
    """Sets up a robust logger to output to both console and a file."""
    logger = logging.getLogger(os.path.basename(log_file))
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger

@torch.no_grad()
def evaluate_composition(model, test_file, stages, stoi, itos, device, G, vocab_size, temperature=0.1, top_k=10):
    """Evaluates compositionality. The logic is identical to your verified correct versions."""
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
            if is_token_level:
                prompt = f"{source} {target} {source}"
                prompt_ids = [stoi[token] for token in prompt.split() if token in stoi]
                x = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
                y = model.generate(x, max_new_tokens=30, temperature=temperature, top_k=top_k)
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

    # --- Setup ---
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir_base = f'out_d{args.n_embd}'
    out_dir = f'{out_dir_base}/composition_mix{args.mixing_ratio}_seed{args.seed}_{timestamp}'
    os.makedirs(out_dir, exist_ok=True)
    
    logger = setup_logger(os.path.join(out_dir, "train.log"))
    
    logger.info("="*60)
    logger.info(f"Starting Final Training Suite")
    logger.info(f"  - Model: {args.n_layer}L-{args.n_head}H-{args.n_embd}D")
    logger.info(f"  - Data: {args.data_dir}")
    logger.info(f"  - Mixing Ratio: {args.mixing_ratio}%")
    logger.info(f"  - Seed: {args.seed}")
    logger.info(f"  - Output Directory: {out_dir}")
    logger.info("="*60)

    # --- Data Loading ---
    with open(os.path.join(args.data_dir, 'stage_info.pkl'), 'rb') as f:
        stages = pickle.load(f)['stages']
    with open(os.path.join(args.data_dir, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    stoi, itos, block_size, vocab_size = meta['stoi'], meta['itos'], meta['block_size'], meta['vocab_size']
    
    G = nx.read_graphml(os.path.join(args.data_dir, 'composition_graph.graphml'))
    train_data = np.memmap(os.path.join(args.data_dir, 'train_10.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(args.data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    test_file = os.path.join(args.data_dir, 'test.txt')

    # --- Model & Optimizer ---
    model_args = dict(n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, block_size=block_size, bias=False, vocab_size=vocab_size, dropout=0.0)
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf).to(args.device)
    optimizer = model.configure_optimizers(weight_decay=1e-1, learning_rate=args.learning_rate, betas=(0.9, 0.95), device_type='cuda')
    logger.info(f"Model initialized: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

    # --- BATCH SAMPLING LOGIC FROM SCRIPT B ---
    # This version respects sequence boundaries for cleaner training signals.
    def get_batch(split):
        data = train_data if split == 'train' else val_data
        # Assumes data is composed of sequences of length (block_size + 1)
        data_len_per_seq = block_size + 1
        num_sequences = len(data) // data_len_per_seq
        
        # Randomly select indices of sequences
        seq_indices = torch.randint(0, num_sequences, (args.batch_size,))
        start_indices = seq_indices * data_len_per_seq
        
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in start_indices])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in start_indices])
        return x.to(args.device), y.to(args.device)

    # --- Training Loop ---
    logger.info("\nStarting training...")
    for iter_num in range(args.max_iters + 1):
        
        # --- LEARNING RATE WARMUP FROM SCRIPT B ---
        if iter_num < WARMUP_ITERS:
            lr = args.learning_rate * iter_num / WARMUP_ITERS
        else:
            lr = args.learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Evaluation and Checkpointing
        if iter_num > 0 and iter_num % args.test_interval == 0:
            model.eval()
            val_losses = [model(*get_batch('val'))[1].item() for _ in range(10)]
            val_loss = np.mean(val_losses)
            results = evaluate_composition(model, test_file, stages, stoi, itos, args.device, G, vocab_size)
            logger.info(f"\n{'='*60}")
            logger.info(f"Iteration {iter_num} | Mix {args.mixing_ratio}% | Seed {args.seed} | Dim {args.n_embd}")
            logger.info(f"Loss: val={val_loss:.4f} | LR: {lr:.2e}")
            for path_type, res in results.items():
                logger.info(f"  {path_type}: {res['accuracy']:.2%} ({res['correct']}/{res['total']})")
            model.train()
        
        if iter_num > 0 and iter_num % args.checkpoint_interval == 0:
            checkpoint = {'model': model.state_dict(), 'model_args': model_args, 'iter_num': iter_num, 'config': vars(args)}
            ckpt_name = f'ckpt_mix{args.mixing_ratio}_seed{args.seed}_iter{iter_num}.pt'
            torch.save(checkpoint, os.path.join(out_dir, ckpt_name))
        
        if iter_num == args.max_iters: break
        
        # Training Step
        X, Y = get_batch('train')
        _, loss = model(X, Y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"🎉 Training finished! Results are in: {out_dir}")
    logger.info("="*60)

if __name__ == "__main__":
    main()