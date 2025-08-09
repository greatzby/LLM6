# 文件名: calibrate_head_adapter_sweep.py
# 描述: 路线图 C (扫描版) - 系统性扫描不同秩对 lm_head 对齐效果的影响。

import torch
import os
import argparse
import glob
from tqdm import tqdm
import numpy as np
import pickle

# --- 从 model.py 导入 ---
from model import GPT, GPTConfig

def get_hybrid_model_path(bundle_type, seed):
    path = f"hybrid_models/hybrid_bundle_{bundle_type}_calibrated_g1.0_seed{seed}.pt"
    if not os.path.exists(path):
        raise FileNotFoundError(f"错误：未找到预期的待修复模型: {path}。请先运行 v3 脚本。")
    print(f"  > 定位到待修复的混合模型: {path}")
    return path

def get_host_model_path(seed):
    pattern = f"out_d92/composition_mix0_seed{seed}_*"
    dirs = glob.glob(pattern)
    if not dirs: raise FileNotFoundError(f"错误：未找到匹配的目录: {pattern}")
    latest_dir = sorted(dirs)[-1]
    path = os.path.join(latest_dir, f"ckpt_mix0_seed{seed}_iter50000.pt")
    if not os.path.exists(path): raise FileNotFoundError(f"错误：在目录 {latest_dir} 中未找到预期的基线模型")
    print(f"  > 定位到目标 (Host) 模型: {path}")
    return path

def load_vocab_indices(data_dir, chars):
    meta_path = os.path.join(data_dir, 'meta.pkl')
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"错误: 在 {data_dir} 中未找到 meta.pkl。")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    
    stoi, itos = meta['stoi'], meta['itos']
    indices = [stoi[c] for c in chars if c in stoi]
    print(f"  > 成功识别 {len(indices)}/{len(chars)} 个任务相关词汇。")
    return indices

# --- 主执行逻辑 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="路线图 C (扫描版): 系统性扫描不同秩对 lm_head 对齐效果的影响")
    parser.add_argument('--seed', type=int, default=42, help='实验用的随机种子')
    parser.add_argument('--bundle_type', type=str, required=True, choices=['attn', 'mlp'], help='要修复的混合模型类型')
    parser.add_argument('--data_dir', type=str, default='data/simple_graph/composition_90', help='数据目录，用于加载 meta.pkl')
    parser.add_argument('--min_rank', type=int, default=0, help='扫描的最小秩 (包含)')
    parser.add_argument('--max_rank', type=int, default=10, help='扫描的最大秩 (包含)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # --- 1. 加载一次共享的模型和数据 ---
    print("\n步骤 1: 加载共享的模型和词汇表...")
    host_model_path = get_host_model_path(args.seed)
    ckpt_host = torch.load(host_model_path, map_location='cpu')
    W_host = ckpt_host['model']['lm_head.weight']
    
    hybrid_model_path = get_hybrid_model_path(args.bundle_type, args.seed)
    ckpt_hybrid = torch.load(hybrid_model_path, map_location='cpu')
    W_hybrid = ckpt_hybrid['model']['lm_head.weight']
    gpt_config_args = ckpt_hybrid['model_args']
    
    task_chars = [str(i) for i in range(10)]
    task_vocab_indices = load_vocab_indices(args.data_dir, task_chars)
    
    W_host_task = W_host[task_vocab_indices, :].to(device)
    W_hybrid_task = W_hybrid[task_vocab_indices, :].to(device)
    d = W_host.shape[1]

    # --- 2. 计算一次完整的对齐矩阵 ---
    print("\n步骤 2: 计算一次完整的对齐矩阵...")
    M = torch.linalg.lstsq(W_hybrid_task, W_host_task).solution
    delta_M = M - torch.eye(d, device=device)
    U, S, Vh = torch.linalg.svd(delta_M, full_matrices=False)
    print("  > 完整的 SVD 分解已完成。")

    generated_files = []
    output_dir = "hybrid_models"
    os.makedirs(output_dir, exist_ok=True)

    # --- 3. 循环生成不同秩的模型 ---
    print(f"\n步骤 3: 循环生成从 rank={args.min_rank} 到 rank={args.max_rank} 的模型...")
    
    rank_range = range(args.min_rank, args.max_rank + 1)
    for rank in tqdm(rank_range, desc="生成不同秩的模型"):
        
        if rank == 0:
            # rank=0 是特殊情况，M_final 就是单位矩阵
            M_final = torch.eye(d, device=device)
        else:
            # 重建低秩的 delta
            r = min(rank, U.shape[1])
            U_r, S_r, Vh_r = U[:, :r], torch.diag(S[:r]), Vh[:r, :]
            delta_M_rank_r = U_r @ S_r @ Vh_r
            M_final = torch.eye(d, device=device) + delta_M_rank_r
        
        # "烘焙"变换到 lm_head 权重中
        W_hybrid_calibrated = W_hybrid.to(device) @ M_final.T
        
        calibrated_state_dict = ckpt_hybrid['model'].copy()
        calibrated_state_dict['lm_head.weight'] = W_hybrid_calibrated.cpu()
        if gpt_config_args.get('tie_weights', True):
            calibrated_state_dict['transformer.wte.weight'] = W_hybrid_calibrated.cpu()

        # 保存新的模型文件
        output_filename = f"hybrid_bundle_{args.bundle_type}_head_calibrated_rank{rank}_seed{args.seed}.pt"
        output_path = os.path.join(output_dir, output_filename)
        
        new_ckpt = ckpt_hybrid.copy()
        new_ckpt['model'] = calibrated_state_dict
        torch.save(new_ckpt, output_path)
        generated_files.append(output_path)

    print("  > 所有秩的模型文件均已生成。")

    # --- 4. 生成总括性的评估脚本 ---
    print("\n步骤 4: 生成总括性评估脚本...")
    script_content = "#!/bin/bash\n"
    script_content += f"echo '🚀 开始批量评估 {args.bundle_type} Bundle 的头部校准模型 (rank {args.min_rank}-{args.max_rank})... 🚀'\n"
    
    for model_path in generated_files:
        rank_val = model_path.split('rank')[1].split('_')[0]
        script_content += f"\necho '\n--- 正在评估 Rank={rank_val}: {os.path.basename(model_path)} ---'\n"
        script_content += f"python evaluate_hybrid_model.py --model_path {model_path} --data_dir {args.data_dir} --temperature 0\n"
        
    script_filename = f"run_{args.bundle_type}_head_sweep_eval.sh"
    with open(script_filename, 'w') as f:
        f.write(script_content)
    
    print("-" * 60)
    print(f"🎉 头部校准扫描完成！")
    print(f"   所有模型已保存。")
    print(f"   总括评估脚本 '{script_filename}' 也已创建。")