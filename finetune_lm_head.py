# ----------- 文件: finetune_lm_head.py (最终修正版 v3) -----------
import os
import torch
import numpy as np
import argparse
import time
from model import GPT, GPTConfig # 使用您原始的、未经修改的 model.py

# --- 配置参数 ---
MAX_ITERS = 600
LEARNING_RATE = 3e-4
BATCH_SIZE = 32
EVAL_INTERVAL = 50
SEED = 42
# ==================================================================
# 关键修正 1: 更新为用户提供的正确数据路径
DATA_DIR = 'data/simple_graph/composition_90_mixed_20'
# ==================================================================
OUTPUT_DIR = 'finetuned_models'

# --- 设备配置 ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type=device, dtype=ptdtype)

# --- 数据加载函数 ---
def get_batch(split, block_size, data_dir):
    # ==================================================================
    # 关键修正 2: 处理非标准的训练文件名 (train_10.bin)
    if split == 'train':
        filename = 'train_10.bin'
    else: # split == 'val'
        filename = 'val.bin'
    # ==================================================================
    
    data_path = os.path.join(data_dir, filename)
    try:
        data = np.memmap(data_path, dtype=np.uint16, mode='r')
    except FileNotFoundError:
        print(f"\n!!!!!!!!!! 致命错误 !!!!!!!!!!")
        print(f"数据文件未找到于: '{data_path}'")
        print(f"请再次确认您的 --data_dir 参数设置正确，且该目录下包含 '{filename}' 文件。")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
        exit(1)
        
    ix = torch.randint(len(data) - block_size, (BATCH_SIZE,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device == 'cuda':
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, block_size, data_dir):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(100) # 评估100个批次
        for k in range(100):
            X, Y = get_batch(split, block_size, data_dir)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# --- 主函数 ---
def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"======== 🚀 开始批量微调任务 🚀 ========")
    print(f"微调数据源: {args.data_dir} (训练集: train_10.bin, 验证集: val.bin)")
    print(f"将在以下 {len(args.model_paths)} 个模型上进行 lm_head 微调:")
    for path in args.model_paths:
        print(f"  - {path}")
    print("========================================\n")

    for i, model_path in enumerate(args.model_paths):
        start_time = time.time()
        print(f"--- 处理模型 {i+1}/{len(args.model_paths)}: {os.path.basename(model_path)} ---")

        if not os.path.exists(model_path):
            print(f"[!] 警告: 文件未找到，跳过: {model_path}\n")
            continue

        # ... (模型加载、冻结、优化器设置等逻辑与之前版本完全相同，无需修改) ...
        try:
            ckpt = torch.load(model_path, map_location=device)
            model_args = ckpt['model_args']
            model_args['tie_weights'] = False
            gptconf = GPTConfig(**model_args)
            model = GPT(gptconf)
            state_dict = ckpt.get('model', ckpt.get('model_state_dict'))
            unwanted_prefix = '_orig_mod.'
            for k,v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            model.load_state_dict(state_dict, strict=True)
            model.to(device)
            print(f"[*] 模型加载成功。参数: {model.get_num_params()/1e6:.2f}M")
        except Exception as e:
            print(f"[!] 错误: 加载模型失败，跳过。错误信息: {e}\n")
            continue

        trainable_params = 0
        for name, param in model.named_parameters():
            if 'lm_head.weight' in name:
                param.requires_grad = True
                trainable_params += param.numel()
                print(f"  > 启用梯度进行微调: {name}")
            else:
                param.requires_grad = False
        print(f"[*] 可训练参数 (lm_head.weight): {trainable_params}")

        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)

        print("[*] 开始微调...")
        model.train()
        for iter_num in range(args.max_iters):
            if iter_num % args.eval_interval == 0 or iter_num == args.max_iters - 1:
                losses = estimate_loss(model, gptconf.block_size, args.data_dir)
                print(f"  迭代 {iter_num:4d}/{args.max_iters}: 训练集损失 {losses['train']:.4f}, 验证集损失 {losses['val']:.4f}")

            X, Y = get_batch('train', gptconf.block_size, args.data_dir)
            with ctx:
                logits, loss = model(X, Y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        original_basename = os.path.basename(model_path)
        new_filename = f"{os.path.splitext(original_basename)[0]}_finetuned.pt"
        output_path = os.path.join(args.output_dir, new_filename)
        final_ckpt = {'model_state_dict': model.state_dict(), 'model_args': gptconf.__dict__}
        torch.save(final_ckpt, output_path)
        end_time = time.time()
        print(f"[*] 微调完成！耗时: {end_time - start_time:.2f} 秒")
        print(f"[*] 修复后的模型已保存至: {output_path}\n")

    print("======== ✅ 所有任务已完成 ✅ ========")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Finetune the lm_head of selected hybrid models.")
    parser.add_argument('--model_paths', type=str, nargs='+', required=True, help='A list of model checkpoint paths to finetune.')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help='Directory to save the finetuned models.')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR, help='Directory containing the training data.')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE, help='Learning rate for the optimizer.')
    parser.add_argument('--max_iters', type=int, default=MAX_ITERS, help='Total number of training iterations.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size.')
    parser.add_argument('--eval_interval', type=int, default=EVAL_INTERVAL, help='Interval for evaluation.')
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed.')
    args = parser.parse_args()
    main(args)