#!/usr/bin/env python3
"""
corrected_similarity.py
修正的相似度计算，考虑有效秩和主成分
"""

import os
import glob
import torch
import numpy as np
from scipy.linalg import svd
import matplotlib.pyplot as plt

def get_checkpoint_path(ratio, seed, iteration):
    pattern = f"out/composition_mix{ratio}_seed{seed}_*"
    dirs = glob.glob(pattern)
    if not dirs:
        raise FileNotFoundError(f"No directory found matching: {pattern}")
    selected_dir = sorted(dirs)[-1]
    return f"{selected_dir}/ckpt_mix{ratio}_seed{seed}_iter{iteration}.pt"

def load_weight_matrix(ratio, seed, iteration):
    path = get_checkpoint_path(ratio, seed, iteration)
    checkpoint = torch.load(path, map_location='cpu')
    
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    W = state_dict['lm_head.weight'].float().numpy()
    del checkpoint, state_dict
    
    return W

def effective_rank(S, threshold=0.01):
    """计算有效秩（基于熵）"""
    S = S[S > threshold * S[0]]  # 只保留相对重要的奇异值
    if len(S) == 0:
        return 0
    S_normalized = S / S.sum()
    entropy = -np.sum(S_normalized * np.log(S_normalized + 1e-12))
    return np.exp(entropy)

def compute_subspace_similarity_corrected(W1, W2, method='grassmann'):
    """
    修正的子空间相似度计算
    method: 'grassmann', 'principal', 'weighted'
    """
    # SVD分解
    U1, S1, Vt1 = svd(W1, full_matrices=False)
    U2, S2, Vt2 = svd(W2, full_matrices=False)
    
    # 计算有效秩
    er1 = effective_rank(S1)
    er2 = effective_rank(S2)
    k = int(min(er1, er2))  # 使用有效秩作为截断维度
    
    print(f"\nEffective ranks: {er1:.2f} vs {er2:.2f}, using k={k}")
    
    # 只使用前k个主要成分
    V1_k = Vt1[:k, :].T  # (120, k)
    V2_k = Vt2[:k, :].T  # (120, k)
    
    # 计算主角度
    overlap = V1_k.T @ V2_k  # (k, k)
    cos_angles = svd(overlap, compute_uv=False)
    cos_angles = np.clip(cos_angles, 0, 1)
    angles = np.arccos(cos_angles)
    
    results = {
        'er1': er1,
        'er2': er2,
        'k': k,
        'cos_angles': cos_angles,
        'angles_deg': np.degrees(angles),
    }
    
    if method == 'grassmann':
        # Grassmann距离
        results['similarity'] = 1 - np.sqrt(np.sum(angles**2)) / np.sqrt(k * (np.pi/2)**2)
        results['distance'] = np.sqrt(np.sum(angles**2))
    
    elif method == 'principal':
        # 基于主角度的相似度（平均余弦）
        results['similarity'] = np.mean(cos_angles)
        results['distance'] = np.mean(angles)
    
    elif method == 'weighted':
        # 加权相似度（按奇异值加权）
        weights1 = S1[:k] / S1[:k].sum()
        weights2 = S2[:k] / S2[:k].sum()
        weights = (weights1 + weights2) / 2
        results['similarity'] = np.sum(weights * cos_angles)
        results['distance'] = np.sum(weights * angles)
    
    # 计算正交补空间的维度
    # V2中不能被V1表示的部分
    V2_proj_V1 = V1_k @ (V1_k.T @ V2_k)
    residual = V2_k - V2_proj_V1
    residual_rank = np.linalg.matrix_rank(residual, tol=0.1)
    results['new_dims'] = residual_rank
    
    # Coverage（V1能解释V2的比例）
    coverage = np.linalg.norm(V2_proj_V1, 'fro')**2 / np.linalg.norm(V2_k, 'fro')**2
    results['coverage'] = coverage
    
    return results

def analyze_evolution():
    """分析训练过程中的演化"""
    print("="*80)
    print("Corrected Subspace Similarity Analysis")
    print("="*80)
    
    # 测试配置
    configs = [
        ("Initial", 0, 42, 3000, 20, 42, 3000),
        ("Middle", 0, 42, 20000, 20, 42, 20000),
        ("Final", 0, 42, 50000, 20, 42, 50000),
    ]
    
    results_all = []
    
    for stage, r1, s1, i1, r2, s2, i2 in configs:
        print(f"\n### {stage}: 0% vs 20% mix")
        
        W1 = load_weight_matrix(r1, s1, i1)
        W2 = load_weight_matrix(r2, s2, i2)
        
        # 尝试不同的方法
        for method in ['grassmann', 'principal', 'weighted']:
            print(f"\n{method.capitalize()} method:")
            results = compute_subspace_similarity_corrected(W1, W2, method)
            
            print(f"  Similarity: {results['similarity']:.4f}")
            print(f"  Distance: {results['distance']:.4f}")
            print(f"  Coverage: {results['coverage']:.4f}")
            print(f"  New dimensions: {results['new_dims']}")
            print(f"  Top 5 angles (deg): {results['angles_deg'][:5]}")
            
            results['stage'] = stage
            results['method'] = method
            results_all.append(results)
    
    # 可视化
    plot_results(results_all)
    
    return results_all

def plot_results(results_all):
    """绘制结果"""
    plt.figure(figsize=(15, 10))
    
    # 1. 相似度演化
    plt.subplot(2, 2, 1)
    stages = ['Initial', 'Middle', 'Final']
    for method in ['grassmann', 'principal', 'weighted']:
        sims = [r['similarity'] for r in results_all if r['method'] == method]
        plt.plot(stages, sims, 'o-', label=method, linewidth=2, markersize=8)
    plt.ylabel('Similarity')
    plt.title('Subspace Similarity Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. 主角度分布
    plt.subplot(2, 2, 2)
    for i, stage in enumerate(stages):
        r = [r for r in results_all if r['stage'] == stage and r['method'] == 'principal'][0]
        angles = r['angles_deg'][:10]
        plt.plot(range(1, len(angles)+1), angles, 'o-', label=stage, linewidth=2)
    plt.xlabel('Principal Angle Index')
    plt.ylabel('Angle (degrees)')
    plt.title('Principal Angles Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Coverage演化
    plt.subplot(2, 2, 3)
    coverages = [r['coverage'] for r in results_all if r['method'] == 'principal']
    plt.plot(stages, coverages, 'o-', color='green', linewidth=3, markersize=10)
    plt.ylabel('Coverage')
    plt.title('How much of 0% is contained in 20%')
    plt.ylim([0.7, 1.0])
    plt.grid(True, alpha=0.3)
    
    # 4. 有效秩比较
    plt.subplot(2, 2, 4)
    er1s = [r['er1'] for r in results_all if r['method'] == 'principal']
    er2s = [r['er2'] for r in results_all if r['method'] == 'principal']
    x = np.arange(len(stages))
    width = 0.35
    plt.bar(x - width/2, er1s, width, label='0% mix', color='blue', alpha=0.7)
    plt.bar(x + width/2, er2s, width, label='20% mix', color='orange', alpha=0.7)
    plt.xlabel('Training Stage')
    plt.ylabel('Effective Rank')
    plt.title('Effective Rank Comparison')
    plt.xticks(x, stages)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('corrected_similarity_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nPlot saved as 'corrected_similarity_analysis.png'")

def main():
    results = analyze_evolution()
    
    # 打印总结
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    principal_results = [r for r in results if r['method'] == 'principal']
    
    print("\nUsing Principal Angle method:")
    for r in principal_results:
        print(f"\n{r['stage']}:")
        print(f"  Similarity: {r['similarity']:.4f}")
        print(f"  Coverage: {r['coverage']:.4f}")
        print(f"  ER: {r['er1']:.1f} → {r['er2']:.1f} (diff: {r['er2']-r['er1']:.1f})")
        print(f"  First angle: {r['angles_deg'][0]:.1f}°")

if __name__ == "__main__":
    main()