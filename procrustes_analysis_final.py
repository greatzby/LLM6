#!/usr/bin/env python3
"""
procrustes_analysis_final.py

本脚本是Procrustes分析的最终、最完善的版本，采纳了用户的深刻见解和严谨设计。

核心特性:
1.  **方法论严谨**: 独立验证实验先行，确保工具的准确性。
2.  **度量全面**: 同时使用三种互补的度量，提供完整视图。
3.  **代码稳健**: 清晰的函数分离和严格的断言检查。
4.  **结果智能**: 自动化的结果解释和结论总结。
"""

import os
import glob
import torch
import numpy as np
from scipy.linalg import svd, orthogonal_procrustes

# 全局配置
ITERATION_TO_ANALYZE = 50000
K_VALUE = 60

# ============= 数据加载函数 =============
def get_checkpoint_path(d_model, ratio, seed, iteration):
    pattern = f"out_d{d_model}/composition_mix{ratio}_seed{seed}_*"
    dirs = sorted(glob.glob(pattern))
    if not dirs: return None
    latest_dir = dirs[-1]
    path = os.path.join(latest_dir, f'ckpt_mix{ratio}_seed{seed}_iter{iteration}.pt')
    return path if os.path.exists(path) else None

def load_weight_matrix(d_model, ratio, seed, iteration):
    path = get_checkpoint_path(d_model, ratio, seed, iteration)
    if path is None: return None
    try:
        checkpoint = torch.load(path, map_location='cpu')
        state_dict = checkpoint.get('model', checkpoint)
        return state_dict['lm_head.weight'].float().numpy()
    except Exception as e:
        print(f"  [错误] 加载失败: {e}")
        return None

# ============= 核心度量函数 =============
def calculate_subspace_similarity(V1, V2):
    assert V1.shape == V2.shape and V1.shape[0] >= V1.shape[1]
    overlap = V1.T @ V2
    cos_theta = np.clip(svd(overlap, compute_uv=False), 0, 1)
    return np.mean(cos_theta)

def calculate_basis_alignment(V1, V2):
    assert V1.shape == V2.shape and V1.shape[0] >= V1.shape[1]
    cos_sim_matrix = np.abs(V1.T @ V2)
    max_cosines = np.max(cos_sim_matrix, axis=1)
    return np.mean(max_cosines)

def calculate_frobenius_alignment(V1, V2):
    assert V1.shape == V2.shape and V1.shape[0] >= V1.shape[1]
    diff = V1 - V2
    distance = np.linalg.norm(diff, 'fro')
    max_distance = np.sqrt(2 * V1.shape[1])
    alignment = 1 - (distance / max_distance)
    return max(0, alignment)

# ============= 验证实验 =============
def run_verification_experiment():
    print("\n" + "="*80)
    print("步骤 0: 运行验证实验，确保方法论正确")
    print("="*80)
    np.random.seed(42)
    d, k = 92, 60
    Q1, _ = np.linalg.qr(np.random.randn(d, k))
    R_true, _ = np.linalg.qr(np.random.randn(k, k))
    Q2 = Q1 @ R_true
    
    print("已创建 Q1 (随机正交基) 和 Q2 (Q1的精确旋转版本)。")
    print("理论预期: 子空间相似度≈1.0, 基向量对齐度在对齐后应恢复到≈1.0。")
    
    subspace_sim_before = calculate_subspace_similarity(Q1, Q2)
    basis_align_before = calculate_basis_alignment(Q1, Q2)
    
    R_found, _ = orthogonal_procrustes(Q2, Q1)
    Q2_aligned = Q2 @ R_found
    
    subspace_sim_after = calculate_subspace_similarity(Q1, Q2_aligned)
    basis_align_after = calculate_basis_alignment(Q1, Q2_aligned)
    
    # 【微小修正】使用更直接的误差计算
    R_error = np.linalg.norm(R_found - R_true.T, 'fro')
    
    print("\n实验结果：")
    print(f"  - 子空间相似度: {subspace_sim_before:.6f} → {subspace_sim_after:.6f} (变化: {subspace_sim_after - subspace_sim_before:+.6f})")
    print(f"  - 基向量对齐度: {basis_align_before:.6f} → {basis_align_after:.6f} (变化: {basis_align_after - basis_align_before:+.6f})")
    print(f"  - 旋转矩阵恢复误差: {R_error:.6f} (应接近0)")
    
    if basis_align_after > 0.999 and abs(subspace_sim_after - subspace_sim_before) < 0.001:
        print("\n✅ 验证成功！方法论和实现均正确。")
        return True
    else:
        print("\n⚠️ 验证失败！请检查实现。")
        return False

# ============= 主分析函数 =============
def analyze_model_pair(config):
    W1 = load_weight_matrix(config['d_model'], config['ratio1'], config['seed1'], config['iter'])
    W2 = load_weight_matrix(config['d_model'], config['ratio2'], config['seed2'], config['iter'])
    if W1 is None or W2 is None: return None
    
    _, _, Vt1 = svd(W1, full_matrices=False)
    _, _, Vt2 = svd(W2, full_matrices=False)
    k = config['k']
    V1_k, V2_k = Vt1[:k, :].T, Vt2[:k, :].T
    
    metrics_before = {
        'subspace': calculate_subspace_similarity(V1_k, V2_k),
        'basis': calculate_basis_alignment(V1_k, V2_k),
        'frobenius': calculate_frobenius_alignment(V1_k, V2_k)
    }
    
    R, _ = orthogonal_procrustes(V2_k, V1_k)
    V2_k_aligned = V2_k @ R
    
    metrics_after = {
        'subspace': calculate_subspace_similarity(V1_k, V2_k_aligned),
        'basis': calculate_basis_alignment(V1_k, V2_k_aligned),
        'frobenius': calculate_frobenius_alignment(V1_k, V2_k_aligned)
    }
    
    print(f"\n模型对比: mix={config['ratio1']}%/s={config['seed1']} vs mix={config['ratio2']}%/s={config['seed2']}")
    print("="*70)
    print(f"{'度量':<20} | {'对齐前':>12} | {'对齐后':>12} | {'变化':>12}")
    print("-"*70)
    for name, disp in [('subspace', '子空间相似度'), ('basis', '基向量对齐度'), ('frobenius', 'Frobenius对齐度')]:
        b, a = metrics_before[name], metrics_after[name]
        print(f"{disp:<20} | {b:>12.6f} | {a:>12.6f} | {a-b:>+12.6f}")
    print("-"*70)
    return metrics_before, metrics_after

# ============= 主程序 =============
def main():
    if not run_verification_experiment(): return
    
    configs = [
        {'description': '控制组: 0% mix, 不同种子', 'd_model': 92, 'k': K_VALUE, 'iter': ITERATION_TO_ANALYZE, 'ratio1': 0, 'seed1': 42, 'ratio2': 0, 'seed2': 123},
        {'description': '控制组: 20% mix, 不同种子', 'd_model': 92, 'k': K_VALUE, 'iter': ITERATION_TO_ANALYZE, 'ratio1': 20, 'seed1': 42, 'ratio2': 20, 'seed2': 123},
        {'description': '实验组: 0% vs 20%, 种子42', 'd_model': 92, 'k': K_VALUE, 'iter': ITERATION_TO_ANALYZE, 'ratio1': 0, 'seed1': 42, 'ratio2': 20, 'seed2': 42},
        {'description': '实验组: 0% vs 20%, 种子123', 'd_model': 92, 'k': K_VALUE, 'iter': ITERATION_TO_ANALYZE, 'ratio1': 0, 'seed1': 123, 'ratio2': 20, 'seed2': 123}
    ]
    
    print("\n" + "="*80)
    print("步骤 1: 开始分析真实模型")
    print("="*80)
    
    results = {cfg['description']: analyze_model_pair(cfg) for cfg in configs}
    
    print("\n" + "#"*80)
    print("分析总结")
    print("#"*80)
    
    control_alignments = [r[1]['basis'] for d, r in results.items() if '控制组' in d and r is not None]
    exp_alignments = [r[1]['basis'] for d, r in results.items() if '实验组' in d and r is not None]
    
    if control_alignments:
        avg_control_align = np.mean(control_alignments)
        print(f"\n[控制组] 平均对齐后基向量对齐度: {avg_control_align:.4f}")
        if avg_control_align > 0.99:
            print("  ✅ 结论: 不同种子间的差异几乎完全是旋转，可被成功消除。")
        else:
            print("  ⚠️ 警告: 控制组对齐度未达预期，可能存在其他非旋转差异。")

    if exp_alignments:
        avg_exp_align = np.mean(exp_alignments)
        structural_diff = 1 - avg_exp_align
        print(f"\n[实验组] 平均对齐后基向量对齐度: {avg_exp_align:.4f}")
        print(f"  📊 核心发现: 即使在消除旋转后，0%和20%模型间仍存在 {structural_diff:.2%} 的不可消除的结构性差异。")
        print("     这，就是mix_ratio带来的真实改变。")

    print("\n" + "#"*80)
    print("🎉 分析完成！")
    print("#"*80)

if __name__ == "__main__":
    main()