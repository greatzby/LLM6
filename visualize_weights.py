#!/usr/bin/env python3
"""
visualize_weights.py

一个用于可视化单个checkpoint的W_M_prime权重矩阵的脚本。
将矩阵保存为带有区域划分的热力图。
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import argparse
import networkx as nx

# 假设 model.py 在同一目录或Python路径下
try:
    from model import GPTConfig, GPT
except ImportError:
    print("❌ Error: 'model.py' not found. Please ensure it's in the same directory.")
    exit()

def visualize_W_M_prime(checkpoint_path, data_dir, output_file):
    """
    加载checkpoint，计算W_M_prime，并将其可视化为热力图。
    """
    print("="*60)
    print("🚀 Starting Weight Matrix Visualization")
    print(f"  • Checkpoint: {checkpoint_path}")
    print(f"  • Data Dir:   {data_dir}")
    print(f"  • Output:     {output_file}")
    print("="*60)

    # --- 1. 加载配置和元信息 ---
    device = 'cpu' # 可视化在CPU上进行即可

    # 加载 vocab_size 和 S1/S2/S3 分组信息
    try:
        with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
        vocab_size = meta['vocab_size']
        print(f"✅ Loaded meta.pkl: vocab_size = {vocab_size}")

        with open(os.path.join(data_dir, 'stage_info.pkl'), 'rb') as f:
            stage_info = pickle.load(f)
        S1, S2, S3 = stage_info['stages']
        print(f"✅ Loaded stage_info.pkl: S1={len(S1)}, S2={len(S2)}, S3={len(S3)}")
    except FileNotFoundError as e:
        print(f"❌ Error: Required file not found in {data_dir}. Details: {e}")
        return

    # --- 2. 加载模型 ---
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except FileNotFoundError:
        print(f"❌ Error: Checkpoint file not found at {checkpoint_path}")
        return

    model_args = checkpoint['model_args']
    # 确保vocab_size与meta.pkl一致
    model_args['vocab_size'] = vocab_size

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    
    # 清理模型状态字典中的 `_orig_mod.` 前缀
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
    model.load_state_dict(state_dict)
    model.eval()
    print("✅ Model loaded successfully.")

    # --- 3. 计算 W_M_prime 矩阵 ---
    print("⏳ Calculating W_M_prime matrix... (This may take a moment)")
    W_M_prime = []
    with torch.no_grad():
        # 我们只关心节点token，所以可以只计算到 vocab_size
        for i in range(vocab_size):
            token_emb = model.transformer.wte(torch.tensor([i], device=device))
            # 假设是单层transformer
            ffn_out = model.transformer.h[0].mlp(token_emb)
            combined = token_emb + ffn_out
            logits = model.lm_head(combined)
            W_M_prime.append(logits.squeeze().cpu().numpy()[:vocab_size])
    W_M_prime = np.array(W_M_prime)
    print("✅ W_M_prime matrix calculated.")

    # --- 4. 准备可视化 ---
    # 定义S1, S2, S3的token范围
    node_to_token = {node: node + 2 for node in range(len(S1) + len(S2) + len(S3))}
    S1_tokens = sorted([node_to_token[n] for n in S1])
    S2_tokens = sorted([node_to_token[n] for n in S2])
    S3_tokens = sorted([node_to_token[n] for n in S3])

    # --- 5. 绘制热力图 ---
    print("🎨 Generating heatmap...")
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # 使用 'viridis' 或 'plasma' 色彩映射，它们对色盲友好且感知均匀
    # vmin, vmax 用于控制颜色范围，可以根据实际权重分布调整
    vmax = np.percentile(W_M_prime, 99.5) # 忽略极端值，让颜色分布更好
    vmin = np.percentile(W_M_prime, 0.5)
    cax = ax.imshow(W_M_prime, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
    
    # 添加颜色条
    cbar = fig.colorbar(cax, label='Weight Value')
    
    # 设置标题和标签
    ax.set_title(f'W_M_prime Heatmap\n(Checkpoint: {os.path.basename(checkpoint_path)})', fontsize=16, fontweight='bold')
    ax.set_xlabel('Target Token ID', fontsize=12)
    ax.set_ylabel('Source Token ID', fontsize=12)

    # 绘制S1, S2, S3区域的分割线，这是可视化的关键！
    boundaries = []
    if S1_tokens: boundaries.extend([min(S1_tokens) - 0.5, max(S1_tokens) + 0.5])
    if S2_tokens: boundaries.extend([min(S2_tokens) - 0.5, max(S2_tokens) + 0.5])
    if S3_tokens: boundaries.extend([min(S3_tokens) - 0.5, max(S3_tokens) + 0.5])
    
    for b in sorted(list(set(boundaries))):
        ax.axhline(b, color='red', linestyle='--', linewidth=1.5)
        ax.axvline(b, color='red', linestyle='--', linewidth=1.5)

    # 添加区域标签，让图更容易读懂
    def add_region_label(tokens, name):
        if not tokens: return
        center = (min(tokens) + max(tokens)) / 2
        # 在图的顶部和左侧添加标签
        ax.text(center, -0.05 * vocab_size, name, ha='center', va='bottom', color='red', fontsize=14, fontweight='bold')
        ax.text(-0.05 * vocab_size, center, name, ha='right', va='center', color='red', fontsize=14, fontweight='bold', rotation=90)

    add_region_label(S1_tokens, 'S1')
    add_region_label(S2_tokens, 'S2')
    add_region_label(S3_tokens, 'S3')

    # 调整布局以防止标签被裁剪
    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
    
    # 保存图像
    plt.savefig(output_file, dpi=200)
    print(f"✅ Heatmap saved to: {output_file}")
    plt.close(fig) # 关闭图形，释放内存

def main():
    parser = argparse.ArgumentParser(description="Visualize the W_M_prime weight matrix from a single checkpoint.")
    
    parser.add_argument('--checkpoint', type=str, required=True, 
                        help="Path to the specific checkpoint file (e.g., out/my_exp/ckpt_50000.pt).")
    
    parser.add_argument('--data_dir', type=str, required=True, 
                        help="Path to the data directory used for training (e.g., data/simple_graph/my_data).")
    
    parser.add_argument('--output', type=str, default=None,
                        help="Path to save the output heatmap image. If not provided, a default name will be generated.")
    
    args = parser.parse_args()
    
    # 如果没有提供输出文件名，自动生成一个
    if args.output is None:
        ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0] # e.g., ckpt_50000
        exp_name = os.path.basename(os.path.dirname(args.checkpoint)) # e.g., composition_...
        output_file = f'heatmap_{exp_name}_{ckpt_name}.png'
    else:
        output_file = args.output
        
    visualize_W_M_prime(args.checkpoint, args.data_dir, output_file)

if __name__ == "__main__":
    main()