# ===================================================================
#           finetune_specialized_v2.py (最终修复版)
#
#  一个高精度的微调脚本，它能正确地：
#  1. 从模型文件 (.pt) 中动态加载模型配置 (GPTConfig)。
#  2. 基于该配置重建完全匹配的模型架构。
#  3. 加载权重，冻结除 lm_head 外的所有参数。
#  4. 在纯净的数据集上进行专项训练。
# ===================================================================
import os
import torch
import numpy as np
import argparse
from tqdm import tqdm
import pickle

# 导入您自己的模型定义，确保我们使用的是同一个 GPT 类
from model import GPT, GPTConfig

def get_batch(data, block_size, batch_size, device):
    """从数据中获取一个批次"""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

def main(args):
    print(f"\n--- 开始处理模型: {os.path.basename(args.source_model_path)} ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 使用设备: {device}")

    # --- 1. 动态加载模型 (核心修复) ---
    try:
        checkpoint = torch.load(args.source_model_path, map_location=device)
        
        # 步骤 A: 稳健地加载模型参数 (model_args)
        # 兼容您的 evaluation 脚本，它也检查 'config'
        model_args_dict = checkpoint.get('model_args', None)
        if model_args_dict is None:
            model_args_dict = checkpoint.get('config', {})
            print("[*] Info: Checkpoint key 'model_args' not found. Falling back to 'config'.")
        
        # 步骤 B: 创建完全匹配的配置和模型
        config = GPTConfig(**model_args_dict)
        model = GPT(config).to(device)
        
        # 步骤 C: 稳健地加载模型状态字典 (state_dict)
        state_dict = checkpoint.get('model', None) # 您的混合脚本使用 'model'
        if state_dict is None:
            state_dict = checkpoint.get('model_state_dict', None) # 备用键

        if state_dict is None:
            raise KeyError("在模型文件中找不到 'model' 或 'model_state_dict' 键。")

        # 处理 torch.compile 可能添加的前缀
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        
        model.load_state_dict(state_dict)
        print(f"[*] 模型加载成功。总参数: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    except Exception as e:
        print(f"[!] 致命错误: 加载模型失败: {e}")
        print("[!] 请确保您的 'model.py' 与被加载的模型架构匹配。")
        return

    # --- 2. 冻结除 lm_head 外的所有参数 ---
    trainable_params = 0
    print("[*] 正在冻结参数...")
    for name, param in model.named_parameters():
        if 'lm_head' in name:
            param.requires_grad = True
            print(f"  > 启用梯度进行微调: {name}")
            trainable_params += param.numel()
        else:
            param.requires_grad = False
    print(f"[*] 可训练参数 (lm_head): {trainable_params}")

    # --- 3. 准备纯净数据 ---
    try:
        train_data = np.memmap(os.path.join(args.dataset_dir, 'train.bin'), dtype=np.uint16, mode='r')
    except FileNotFoundError:
        print(f"[!] 致命错误: 在 '{args.dataset_dir}' 中找不到 train.bin。请先准备数据。")
        return

    # --- 4. 设置优化器 ---
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)

    # --- 5. 训练循环 ---
    print("[*] 开始微调...")
    model.train()
    for i in tqdm(range(args.max_iters), desc="微调进度"):
        xb, yb = get_batch(train_data, config.block_size, args.batch_size, device)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # --- 6. 保存微调后的模型 ---
    # 保存时也包含 model_args，以便评估脚本可以加载
    final_checkpoint = {
        'model_args': model_args_dict,
        'model': model.state_dict(),
    }
    os.makedirs(os.path.dirname(args.output_model_path), exist_ok=True)
    torch.save(final_checkpoint, args.output_model_path)
    print(f"[*] 微调完成！")
    print(f"[*] 修复后的模型已保存至: {args.output_model_path}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="专项微调GPT模型的lm_head (v2 - 兼容版)")
    parser.add_argument('--source_model_path', type=str, required=True)
    parser.add_argument('--output_model_path', type=str, required=True)
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--max_iters', type=int, default=600)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    args = parser.parse_args()
    main(args)