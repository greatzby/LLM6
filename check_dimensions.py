#!/usr/bin/env python3
"""
check_dimensions.py
检查权重矩阵维度和相似度计算问题
"""

import os
import glob
import torch
import numpy as np
from scipy.linalg import svd

def get_checkpoint_path(ratio, seed, iteration):
    """构建checkpoint路径"""
    pattern = f"out/composition_mix{ratio}_seed{seed}_*"
    dirs = glob.glob(pattern)
    
    if not dirs:
        raise FileNotFoundError(f"No directory found matching: {pattern}")
    
    selected_dir = sorted(dirs)[-1]
    checkpoint_path = f"{selected_dir}/ckpt_mix{ratio}_seed{seed}_iter{iteration}.pt"
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    return checkpoint_path

def load_weight_matrix(ratio, seed, iteration):
    """加载权重矩阵"""
    path = get_checkpoint_path(ratio, seed, iteration)
    print(f"\nLoading: {path}")
    
    checkpoint = torch.load(path, map_location='cpu')
    
    # 获取state dict
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # 打印所有键
    print(f"Available keys in checkpoint:")
    for i, key in enumerate(state_dict.keys()):
        print(f"  {i}: {key} - shape: {state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'N/A'}")
        if i > 20:  # 只打印前20个
            print(f"  ... and {len(state_dict) - 20} more keys")
            break
    
    # 尝试找到权重矩阵
    possible_keys = [
        'lm_head.weight',
        'model.lm_head.weight', 
        'decoder.weight',
        'output_projection.weight',
        'output.weight',
        'fc_out.weight',
        'final_layer.weight'
    ]
    
    W = None
    used_key = None
    for key in possible_keys:
        if key in state_dict:
            W = state_dict[key].float().numpy()
            used_key = key
            break
    
    # 如果还是没找到，尝试找包含'weight'的最后一层
    if W is None:
        for key in reversed(list(state_dict.keys())):
            if 'weight' in key and 'embed' not in key:
                W = state_dict[key].float().numpy()
                used_key = key
                print(f"Using key: {key}")
                break
    
    if W is None:
        raise KeyError(f"Cannot find weight matrix")
    
    print(f"Found weight matrix with key: {used_key}")
    print(f"Weight matrix shape: {W.shape}")
    
    del checkpoint
    del state_dict
    
    return W

def compute_column_similarity_debug(W1, W2):
    """调试版本的列空间相似度计算"""
    print("\n=== Column Space Similarity Debug ===")
    
    # SVD分解
    U1, S1, Vt1 = svd(W1, full_matrices=False)
    U2, S2, Vt2 = svd(W2, full_matrices=False)
    
    print(f"W1 shape: {W1.shape}")
    print(f"W2 shape: {W2.shape}")
    print(f"U1 shape: {U1.shape}, S1 length: {len(S1)}, Vt1 shape: {Vt1.shape}")
    print(f"U2 shape: {U2.shape}, S2 length: {len(S2)}, Vt2 shape: {Vt2.shape}")
    
    # V矩阵
    V1 = Vt1.T
    V2 = Vt2.T
    print(f"V1 shape: {V1.shape}")
    print(f"V2 shape: {V2.shape}")
    
    # 计算重叠矩阵
    overlap = V1.T @ V2
    print(f"Overlap matrix shape: {overlap.shape}")
    
    # 计算奇异值
    singular_values = svd(overlap, compute_uv=False)
    print(f"Singular values shape: {singular_values.shape}")
    print(f"Top 10 singular values: {singular_values[:10]}")
    
    # 检查是否都接近1
    num_ones = np.sum(singular_values > 0.9999)
    print(f"Number of singular values > 0.9999: {num_ones}")
    
    # 计算有效秩
    def effective_rank(S):
        S = S[S > 1e-10]
        if len(S) == 0:
            return 0
        S_normalized = S / S.sum()
        entropy = -np.sum(S_normalized * np.log(S_normalized + 1e-12))
        return np.exp(entropy)
    
    er1 = effective_rank(S1)
    er2 = effective_rank(S2)
    print(f"\nEffective ranks: {er1:.2f} vs {er2:.2f}")
    
    # 检查秩
    rank1 = np.linalg.matrix_rank(W1)
    rank2 = np.linalg.matrix_rank(W2)
    print(f"Matrix ranks: {rank1} vs {rank2}")
    
    # 计算正确的相似度
    if min(V1.shape[1], V2.shape[1]) == 0:
        similarity = 0
    else:
        # 只取有效的维度
        k = min(V1.shape[1], V2.shape[1], rank1, rank2)
        V1_truncated = V1[:, :k]
        V2_truncated = V2[:, :k]
        
        overlap_truncated = V1_truncated.T @ V2_truncated
        singular_values_truncated = svd(overlap_truncated, compute_uv=False)
        similarity = singular_values_truncated[0]
        
        print(f"\nUsing top {k} dimensions")
        print(f"Truncated overlap shape: {overlap_truncated.shape}")
        print(f"Top singular value (similarity): {similarity:.6f}")
    
    return similarity, singular_values

def main():
    """主函数"""
    print("="*80)
    print("Weight Matrix Dimension Check")
    print("="*80)
    
    # 测试配置
    test_configs = [
        (0, 42, 3000),   # Initial
        (20, 42, 3000),  # Initial
        (0, 42, 50000),  # Final
        (20, 42, 50000), # Final
    ]
    
    matrices = {}
    for ratio, seed, iter in test_configs:
        try:
            W = load_weight_matrix(ratio, seed, iter)
            matrices[(ratio, seed, iter)] = W
        except Exception as e:
            print(f"\nError loading mix{ratio}_seed{seed}_iter{iter}: {e}")
    
    # 比较分析
    print("\n" + "="*80)
    print("COMPARISON ANALYSIS")
    print("="*80)
    
    # 1. Initial: 0% vs 20%
    if (0, 42, 3000) in matrices and (20, 42, 3000) in matrices:
        print("\n### Initial (3k iterations): 0% vs 20%")
        W1 = matrices[(0, 42, 3000)]
        W2 = matrices[(20, 42, 3000)]
        similarity, sv = compute_column_similarity_debug(W1, W2)
    
    # 2. Final: 0% vs 20%
    if (0, 42, 50000) in matrices and (20, 42, 50000) in matrices:
        print("\n### Final (50k iterations): 0% vs 20%")
        W1 = matrices[(0, 42, 50000)]
        W2 = matrices[(20, 42, 50000)]
        similarity, sv = compute_column_similarity_debug(W1, W2)
        
        # 额外检查：为什么相似度是1？
        print("\n### Additional checks:")
        
        # 检查矩阵是否相同
        are_equal = np.allclose(W1, W2)
        print(f"Matrices are equal: {are_equal}")
        
        # 检查差异
        diff = np.linalg.norm(W1 - W2, 'fro')
        print(f"Frobenius norm of difference: {diff:.6f}")
        
        # 检查前几个奇异向量
        U1, S1, Vt1 = svd(W1, full_matrices=False)
        U2, S2, Vt2 = svd(W2, full_matrices=False)
        
        print(f"\nFirst singular vector similarity:")
        for i in range(min(5, Vt1.shape[0])):
            sim = np.abs(np.dot(Vt1[i], Vt2[i]))
            print(f"  v{i}: {sim:.6f}")

if __name__ == "__main__":
    main()