"""
analyze_matched_dimensions.py (FIXED)

通过寻找V-space的最佳匹配，深入分析0%和20%模型在92维空间下的真实结构差异。
该脚本解决了SVD排序带来的“维度身份不匹配”问题，并为精确的消融实验提供指导。
修复了 'TypeError: keys must be ... not int64' 的JSON序列化问题。
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import json
import glob
import os
from scipy.optimize import linear_sum_assignment

# --- 配置 ---
CHECKPOINT_DIR = "out_d92"
OUTPUT_DIR = "matched_dims_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 核心功能函数 ---

def get_checkpoint_path(ratio, seed, iteration):
    """构建checkpoint路径 - 自动选择最新的timestamp"""
    pattern = f"{CHECKPOINT_DIR}/composition_mix{ratio}_seed{seed}_*"
    dirs = glob.glob(pattern)
    if not dirs:
        raise FileNotFoundError(f"No directory found matching: {pattern}")
    selected_dir = sorted(dirs)[-1]
    return f"{selected_dir}/ckpt_mix{ratio}_seed{seed}_iter{iteration}.pt"

def load_lm_head_weight(mix_ratio, seed, iteration):
    """加载lm_head.weight"""
    path = get_checkpoint_path(mix_ratio, seed, iteration)
    checkpoint = torch.load(path, map_location='cpu')
    state_dict = checkpoint.get('model', checkpoint)
    W = state_dict['lm_head.weight'].float().numpy()
    print(f"  Loaded lm_head.weight from {path.split('/')[-2]}: shape {W.shape}")
    return W

def find_best_matches_and_analyze(W0, W20):
    """
    通过寻找V-space的最佳匹配来分析差异，解决SVD排序陷阱问题。
    使用匈牙利算法 (linear_sum_assignment) 确保一一对应的最佳匹配。
    """
    # 1. SVD分解
    U0, S0, V0_T = np.linalg.svd(W0, full_matrices=False)
    V0 = V0_T.T
    U20, S20, V20_T = np.linalg.svd(W20, full_matrices=False)
    V20 = V20_T.T
    
    # 2. 计算相似度矩阵 (成本矩阵)
    similarity_matrix = np.abs(V0.T @ V20) # (92, 92)
    cost_matrix = 1 - similarity_matrix
    
    # 3. 使用匈牙利算法找到全局最优匹配
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # 4. 基于匹配结果进行分析
    matched_analysis = []
    for i, j_match in zip(row_ind, col_ind):
        cos_angle = similarity_matrix[i, j_match]
        angle_deg = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi
        energy_change = S20[j_match] - S0[i]
        relative_change = energy_change / (S0[i] + 1e-10)
        rank_swap_distance = abs(i - j_match)
        
        matched_analysis.append({
            'dim_0_index': i,
            'dim_20_match_index': j_match,
            'rank_swap_distance': rank_swap_distance,
            'true_angle_deg': angle_deg,
            'true_relative_energy_change': relative_change,
            's0': float(S0[i]), # 转换为原生float
            's20_matched': float(S20[j_match]), # 转换为原生float
            'similarity': float(cos_angle) # 转换为原生float
        })
        
    return matched_analysis, (U0, S0, V0), (U20, S20, V20)

def create_matched_visualization(analysis, output_prefix):
    """为匹配分析结果创建更深入的可视化"""
    if not analysis:
        return
        
    angles = [d['true_angle_deg'] for d in analysis]
    rel_energy_changes = [d['true_relative_energy_change'] for d in analysis]
    rank_swaps = [d['rank_swap_distance'] for d in analysis]
    dim0_indices = [d['dim_0_index'] for d in analysis]

    fig = plt.figure(figsize=(20, 18))
    
    # 1. 真实角度分布
    ax1 = plt.subplot(3, 2, 1)
    sns.histplot(angles, bins=30, ax=ax1, color='green', kde=True)
    ax1.set_title('Distribution of TRUE Angles (after matching)')
    ax1.set_xlabel('Angle (degrees)')
    ax1.axvline(np.mean(angles), color='red', linestyle='--', label=f'Mean: {np.mean(angles):.1f}°')
    ax1.legend()

    # 2. 真实相对能量变化分布
    ax2 = plt.subplot(3, 2, 2)
    sns.histplot(rel_energy_changes, bins=30, ax=ax2, color='purple', kde=True)
    ax2.set_title('Distribution of TRUE Relative Energy Changes')
    ax2.set_xlabel('Relative Energy Change')
    ax2.axvline(0, color='black', linestyle='-')
    ax2.axvline(np.mean(rel_energy_changes), color='red', linestyle='--', label=f'Mean: {np.mean(rel_energy_changes):.2%}')
    ax2.legend()

    # 3. 核心图：真实角度 vs 真实能量变化
    ax3 = plt.subplot(3, 2, 3)
    scatter = ax3.scatter(angles, rel_energy_changes, c=dim0_indices, cmap='viridis', s=50, alpha=0.7)
    plt.colorbar(scatter, ax=ax3, label='Original Dimension Index (in 0% model)')
    ax3.set_title('TRUE Angle vs. TRUE Energy Change')
    ax3.set_xlabel('True Angle (degrees)')
    ax3.set_ylabel('True Relative Energy Change')
    ax3.grid(True, linestyle='--', alpha=0.5)

    # 4. 维度互换图 (Rank Swap Plot)
    ax4 = plt.subplot(3, 2, 4)
    sorted_analysis = sorted(analysis, key=lambda x: x['dim_0_index'])
    dim0 = [d['dim_0_index'] for d in sorted_analysis]
    dim20_match = [d['dim_20_match_index'] for d in sorted_analysis]
    ax4.plot([0, 91], [0, 91], 'r--', label='No Swap')
    ax4.scatter(dim0, dim20_match, c=rank_swaps, cmap='coolwarm', s=25, alpha=0.8)
    ax4.set_xlabel('Dimension Rank in 0% Model')
    ax4.set_ylabel('Matched Dimension Rank in 20% Model')
    ax4.set_title('Dimension Rank Swapping')
    ax4.grid(True, alpha=0.5)
    ax4.set_aspect('equal')

    # 5. 排序变化距离分布
    ax5 = plt.subplot(3, 2, 5)
    sns.histplot(rank_swaps, bins=30, ax=ax5, color='orange', kde=True)
    ax5.set_title('Distribution of Rank Swap Distances')
    ax5.set_xlabel('abs(rank_0% - rank_20%)')
    ax5.axvline(np.mean(rank_swaps), color='red', linestyle='--', label=f'Mean: {np.mean(rank_swaps):.1f}')
    ax5.legend()
    
    # 6. 文本总结
    ax6 = plt.subplot(3, 2, 6)
    ax6.axis('off')
    
    most_rotated_dims = sorted(analysis, key=lambda x: x['true_angle_deg'], reverse=True)[:5]
    most_stable_dims = sorted(analysis, key=lambda x: x['true_angle_deg'])[:5]
    largest_rank_swaps = sorted(analysis, key=lambda x: x['rank_swap_distance'], reverse=True)[:5]
    
    info_text = f"MATCHED ANALYSIS SUMMARY (Seed {output_prefix.split('_')[-1]})\n"
    info_text += "="*40 + "\n"
    info_text += f"Mean True Angle: {np.mean(angles):.1f}° (vs. naive ~81°)\n"
    info_text += f"Mean Rank Swap Distance: {np.mean(rank_swaps):.1f}\n"
    info_text += f"Dims with >45° True Angle: {sum(a > 45 for a in angles)}\n"
    info_text += f"Dims with >10 rank swap: {sum(r > 10 for r in rank_swaps)}\n\n"
    
    info_text += "MOST RESTRUCTURED DIMS (Largest True Angle):\n"
    for d in most_rotated_dims:
        info_text += f"  - Dim {d['dim_0_index']:<2} -> {d['dim_20_match_index']:<2} | Angle: {d['true_angle_deg']:.1f}°\n"
        
    info_text += "\nMOST STABLE CONCEPTS (Smallest True Angle):\n"
    for d in most_stable_dims:
        info_text += f"  - Dim {d['dim_0_index']:<2} -> {d['dim_20_match_index']:<2} | Angle: {d['true_angle_deg']:.1f}°\n"
        
    info_text += "\nBIGGEST RANK SHIFTS:\n"
    for d in largest_rank_swaps:
        info_text += f"  - Dim {d['dim_0_index']:<2} -> {d['dim_20_match_index']:<2} | Dist: {d['rank_swap_distance']}\n"

    ax6.text(0.05, 0.95, info_text, transform=ax6.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/{output_prefix}_visualization.png', dpi=150)
    plt.close()
    print(f"  Visualization saved to {OUTPUT_DIR}/{output_prefix}_visualization.png")

def generate_precise_transplant_code(analysis, output_prefix):
    """根据匹配分析结果，生成更精确的消融实验代码和候选维度"""
    
    restructured_dims = sorted(analysis, key=lambda x: x['true_angle_deg'], reverse=True)
    energy_boosted_dims = sorted(analysis, key=lambda x: x['true_relative_energy_change'], reverse=True)
    stable_dims = sorted(analysis, key=lambda x: x['true_angle_deg'])

    # ==================== FIX START ====================
    # 将Numpy的整数键转换为Python的原生int
    transplant_map_restructured_top20 = {int(d['dim_0_index']): int(d['dim_20_match_index']) for d in restructured_dims[:20]}
    transplant_map_energy_top20 = {int(d['dim_0_index']): int(d['dim_20_match_index']) for d in energy_boosted_dims[:20]}
    transplant_map_stable_bottom20 = {int(d['dim_0_index']): int(d['dim_20_match_index']) for d in stable_dims[:20]}
    # ===================== FIX END =====================
    
    report = {
        'transplant_candidates': {
            'by_angle_top20': transplant_map_restructured_top20,
            'by_energy_top20': transplant_map_energy_top20,
            'control_stable_bottom20': transplant_map_stable_bottom20
        }
    }
    
    report_path = f'{OUTPUT_DIR}/{output_prefix}_transplant_candidates.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  Transplant candidates saved to {report_path}")

    # 生成代码
    code = f'''
# ===================================================================
# PRECISE ABLATION STUDY CODE (Generated from matched analysis)
# ===================================================================
import torch
import numpy as np
import json

def transplant_matched_dimensions(path_0, path_20, transplant_map):
    """
    将20%模型的匹配维度精确嫁接到0%模型。
    
    Args:
        path_0: 0%模型的checkpoint路径
        path_20: 20%模型的checkpoint路径
        transplant_map: 一个字典 {{{{dim_0_idx: dim_20_idx, ...}}}}
    """
    # 加载权重
    ckpt_0 = torch.load(path_0, map_location='cpu')
    state_0 = ckpt_0.get('model', ckpt_0)
    W0 = state_0['lm_head.weight']
    
    ckpt_20 = torch.load(path_20, map_location='cpu')
    state_20 = ckpt_20.get('model', ckpt_20)
    W20 = state_20['lm_head.weight']
    
    # SVD分解
    U0, S0, V0t = torch.linalg.svd(W0, full_matrices=False)
    U20, S20, V20t = torch.linalg.svd(W20, full_matrices=False)
    
    # 创建混合版本
    U_hybrid = U0.clone()
    S_hybrid = S0.clone()
    V_hybrid_t = V0t.clone()
    
    print(f"Transplanting {{{{len(transplant_map)}}}} dimensions...")
    for dim0_idx, dim20_idx in transplant_map.items():
        print(f"  - Mapping 0%[dim {{{{dim0_idx}}}}] -> 20%[dim {{{{dim20_idx}}}}]")
        U_hybrid[:, dim0_idx] = U20[:, dim20_idx]
        S_hybrid[dim0_idx] = S20[dim20_idx]
        V_hybrid_t[dim0_idx, :] = V20t[dim20_idx, :]
    
    # 重构权重
    W_hybrid = U_hybrid @ torch.diag(S_hybrid) @ V_hybrid_t
    
    # 创建新的checkpoint
    state_hybrid = state_0.copy()
    state_hybrid['lm_head.weight'] = W_hybrid
    
    ckpt_hybrid = ckpt_0.copy()
    if 'model' in ckpt_hybrid:
        ckpt_hybrid['model'] = state_hybrid
    else:
        ckpt_hybrid.update(state_hybrid)
    
    return ckpt_hybrid

# --- 使用示例 ---
if __name__ == "__main__":
    # 1. 定义模型路径 (请根据实际情况修改)
    seed = 42
    iteration = 50000
    path_0 = "{get_checkpoint_path(0, seed, iteration)}"
    path_20 = "{get_checkpoint_path(20, seed, iteration)}"
    
    # 2. 加载候选维度列表
    candidates_path = "{report_path}"
    with open(candidates_path, 'r') as f:
        candidates = json.load(f)
    
    # 3. 选择一个策略进行移植 (例如，移植角度变化最大的Top 20)
    transplant_map = candidates['transplant_candidates']['by_angle_top20']
    # transplant_map = candidates['transplant_candidates']['by_energy_top20']
    # transplant_map = candidates['transplant_candidates']['control_stable_bottom20']
    
    # 将JSON中的字符串key转为int
    transplant_map = {{{{int(k): v for k, v in transplant_map.items()}}}}

    # 4. 执行嫁接
    hybrid_ckpt = transplant_matched_dimensions(path_0, path_20, transplant_map)
    
    # 5. 保存混合模型
    output_filename = f"hybrid_model_transplant_angle_top20.pt"
    torch.save(hybrid_ckpt, output_filename)
    print(f"\\nHybrid model saved to {{{{output_filename}}}}!")
'''
    print("\n" + "="*60)
    print("PRECISE TRANSPLANT CODE GENERATED")
    print("="*60)
    print(code)


# --- 主函数 ---
def main():
    """主函数"""
    print("="*80)
    print("Analyzing Matched Dimensions (Solving SVD Sort-Order Pitfall)")
    print("="*80)
    
    seeds = [42, 123, 456]
    iteration = 50000
    
    for seed in seeds:
        print(f"\n\n--- Analyzing Seed {seed} ---")
        try:
            W0 = load_lm_head_weight(0, seed, iteration)
            W20 = load_lm_head_weight(20, seed, iteration)
            
            # 核心分析
            analysis, svd0, svd20 = find_best_matches_and_analyze(W0, W20)
            
            # 可视化
            output_prefix = f"seed_{seed}"
            create_matched_visualization(analysis, output_prefix)
            
            # 生成消融实验代码和候选列表
            generate_precise_transplant_code(analysis, output_prefix)

        except Exception as e:
            print(f"ERROR analyzing seed {seed}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()