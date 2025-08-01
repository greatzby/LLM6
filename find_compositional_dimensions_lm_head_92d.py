"""
find_compositional_dimensions_lm_head_92d.py
专门分析92维模型的lm_head.weight，找出决定合成功能的关键维度
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import json
import glob
import os

# 配置
CHECKPOINT_DIR = "out_d92"
OUTPUT_DIR = "compositional_dims_lm_head_92d"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_checkpoint_path(ratio, seed, iteration):
    """构建checkpoint路径 - 自动选择最新的timestamp"""
    # 查找所有匹配的目录
    pattern = f"{CHECKPOINT_DIR}/composition_mix{ratio}_seed{seed}_*"
    dirs = glob.glob(pattern)
    
    if not dirs:
        raise FileNotFoundError(f"No directory found matching: {pattern}")
    
    # 选择最新的目录（按名称排序，timestamp在最后）
    selected_dir = sorted(dirs)[-1]
    print(f"  Selected directory: {selected_dir}")
    
    checkpoint_path = f"{selected_dir}/ckpt_mix{ratio}_seed{seed}_iter{iteration}.pt"
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    return checkpoint_path

def load_lm_head_weight(mix_ratio, seed, iteration):
    """只加载lm_head.weight"""
    path = get_checkpoint_path(mix_ratio, seed, iteration)
    checkpoint = torch.load(path, map_location='cpu')
    
    # 获取state dict
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # 提取lm_head.weight
    if 'lm_head.weight' not in state_dict:
        raise KeyError(f"lm_head.weight not found. Available keys: {list(state_dict.keys())[:10]}...")
    
    W = state_dict['lm_head.weight'].float().numpy()
    print(f"  Loaded lm_head.weight: shape {W.shape}")
    
    return W

def analyze_lm_head_differences(W0, W20):
    """分析0%和20%模型的lm_head差异，找出关键维度"""
    # SVD分解
    U0, S0, V0 = np.linalg.svd(W0, full_matrices=False)
    U20, S20, V20 = np.linalg.svd(W20, full_matrices=False)
    
    print(f"\n  SVD shapes: U={U0.shape}, S={S0.shape}, V={V0.T.shape}")
    
    # 1. 计算V空间（思维空间）的主角度
    angles_deg = []
    for i in range(min(V0.shape[0], V20.shape[0])):
        # 计算对应维度之间的角度
        cos_angle = np.abs(np.dot(V0[i, :], V20[i, :]))
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.arccos(cos_angle) * 180 / np.pi
        angles_deg.append(angle)
    
    angles_deg = np.array(angles_deg)
    
    # 2. 计算能量（奇异值）变化
    min_len = min(len(S0), len(S20))
    S0_truncated = S0[:min_len]
    S20_truncated = S20[:min_len]
    
    energy_change = S20_truncated - S0_truncated
    relative_energy_change = energy_change / (S0_truncated + 1e-10)
    
    # 3. 识别关键维度
    # 标准1：角度变化大于45度
    high_angle_dims = np.where(angles_deg > 45)[0]
    
    # 标准2：能量显著增加（相对变化>10%）
    energy_increased_dims = np.where(relative_energy_change > 0.1)[0]
    
    # 标准3：综合判断 - 既有大角度变化又有能量增加
    key_dims = sorted(list(set(high_angle_dims.tolist()) & set(energy_increased_dims.tolist())))
    
    # 额外找出最可疑的维度（角度最大的前10个）
    top_angle_dims = np.argsort(angles_deg)[-10:][::-1]
    
    results = {
        'angles_deg': angles_deg,
        'energy_change': energy_change,
        'relative_energy_change': relative_energy_change,
        'high_angle_dims': high_angle_dims.tolist(),
        'energy_increased_dims': energy_increased_dims.tolist(),
        'key_dims': key_dims,
        'top_angle_dims': top_angle_dims.tolist(),
        'S0': S0,
        'S20': S20,
        'V0': V0,
        'V20': V20
    }
    
    # 打印分析结果
    print(f"\n  Analysis Results:")
    print(f"    Dimensions with angle > 45°: {len(high_angle_dims)} dims = {high_angle_dims.tolist()[:20]}...")
    print(f"    Dimensions with energy increase > 10%: {len(energy_increased_dims)} dims")
    print(f"    Key dimensions (both criteria): {len(key_dims)} dims = {key_dims}")
    print(f"    Top 10 highest angle dims: {top_angle_dims.tolist()}")
    print(f"    Max angle: {np.max(angles_deg):.1f}°")
    print(f"    Mean angle: {np.mean(angles_deg):.1f}°")
    
    return results

def create_visualization(all_results):
    """创建可视化"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    seeds = [42, 123, 456]
    
    for idx, seed in enumerate(seeds):
        if seed not in all_results:
            continue
            
        results = all_results[seed]
        
        # 1. 角度分布
        ax = axes[0, idx]
        ax.hist(results['angles_deg'], bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(45, color='red', linestyle='--', label='45° threshold')
        ax.axvline(np.mean(results['angles_deg']), color='green', linestyle=':', 
                  label=f'Mean: {np.mean(results["angles_deg"]):.1f}°')
        ax.set_xlabel('Principal Angle (degrees)')
        ax.set_ylabel('Count')
        ax.set_title(f'Seed {seed}: Angle Distribution')
        ax.legend()
        
        # 2. 能量变化 vs 角度
        ax = axes[1, idx]
        scatter = ax.scatter(results['angles_deg'], 
                           results['relative_energy_change'][:len(results['angles_deg'])], 
                           c=range(len(results['angles_deg'])), cmap='viridis', alpha=0.6)
        
        # 标记关键维度
        for dim in results['key_dims'][:5]:  # 标记前5个
            ax.annotate(f'{dim}', 
                       (results['angles_deg'][dim], 
                        results['relative_energy_change'][dim]),
                       fontsize=10, color='red')
        
        ax.axhline(0.1, color='orange', linestyle='--', alpha=0.5, label='10% increase')
        ax.axvline(45, color='red', linestyle='--', alpha=0.5, label='45°')
        ax.set_xlabel('Principal Angle (degrees)')
        ax.set_ylabel('Relative Energy Change')
        ax.set_title(f'Seed {seed}: Energy Change vs Angle')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/lm_head_analysis_all_seeds.png', dpi=150)
    plt.close()
    print(f"\nVisualization saved to {OUTPUT_DIR}/lm_head_analysis_all_seeds.png")

def generate_transplant_code(common_dims):
    """生成嫁接实验代码"""
    print("\n" + "="*60)
    print("TRANSPLANT EXPERIMENT CODE")
    print("="*60)
    
    code = f'''
import torch
import numpy as np

def transplant_lm_head_dimensions(path_0, path_20, key_dims={common_dims}):
    """
    将20%模型的关键lm_head维度嫁接到0%模型
    
    Args:
        path_0: 0%模型的checkpoint路径
        path_20: 20%模型的checkpoint路径
        key_dims: 要嫁接的维度列表
    """
    # 加载checkpoints
    ckpt_0 = torch.load(path_0, map_location='cpu')
    ckpt_20 = torch.load(path_20, map_location='cpu')
    
    # 获取state dict
    state_0 = ckpt_0['model'] if 'model' in ckpt_0 else ckpt_0
    state_20 = ckpt_20['model'] if 'model' in ckpt_20 else ckpt_20
    
    # 获取lm_head权重
    W0 = state_0['lm_head.weight']
    W20 = state_20['lm_head.weight']
    
    # SVD分解
    U0, S0, V0 = torch.svd(W0)
    U20, S20, V20 = torch.svd(W20)
    
    # 创建混合版本
    V_hybrid = V0.clone()
    S_hybrid = S0.clone()
    
    # 嫁接关键维度
    print(f"Transplanting dimensions: {{key_dims}}")
    for dim in key_dims:
        if dim < V_hybrid.shape[0]:
            V_hybrid[dim, :] = V20[dim, :]  # 嫁接V空间的基向量
            S_hybrid[dim] = S20[dim]        # 嫁接对应的奇异值
    
    # 重构权重
    W_hybrid = torch.mm(torch.mm(U0, torch.diag(S_hybrid)), V_hybrid.t())
    
    # 创建新的state dict
    state_hybrid = state_0.copy()
    state_hybrid['lm_head.weight'] = W_hybrid
    
    # 创建新的checkpoint
    ckpt_hybrid = ckpt_0.copy()
    if 'model' in ckpt_hybrid:
        ckpt_hybrid['model'] = state_hybrid
    else:
        ckpt_hybrid.update(state_hybrid)
    
    return ckpt_hybrid

# 使用示例
if __name__ == "__main__":
    # 路径需要根据实际情况调整
    path_0 = "out_d92/composition_mix0_seed42_*/ckpt_mix0_seed42_iter50000.pt"
    path_20 = "out_d92/composition_mix20_seed42_*/ckpt_mix20_seed42_iter50000.pt"
    
    # 执行嫁接
    hybrid_ckpt = transplant_lm_head_dimensions(path_0, path_20)
    
    # 保存混合模型
    torch.save(hybrid_ckpt, "hybrid_model_lm_head_transplant.pt")
    print("Hybrid model saved!")
'''
    
    print(code)

def main():
    """主函数"""
    print("="*80)
    print("Finding Compositional Dimensions in 92D LM Head")
    print("="*80)
    
    seeds = [42, 123, 456]
    iteration = 50000  # 分析最终状态
    
    all_results = {}
    all_key_dims = []
    
    # 分析每个种子
    for seed in seeds:
        print(f"\n\nAnalyzing Seed {seed}:")
        print("-"*40)
        
        try:
            # 加载0%和20%的lm_head权重
            print(f"\nLoading 0% model...")
            W0 = load_lm_head_weight(0, seed, iteration)
            
            print(f"\nLoading 20% model...")
            W20 = load_lm_head_weight(20, seed, iteration)
            
            # 分析差异
            results = analyze_lm_head_differences(W0, W20)
            all_results[seed] = results
            all_key_dims.append(set(results['key_dims']))
            
        except Exception as e:
            print(f"Error analyzing seed {seed}: {e}")
            continue
    
    # 找出跨种子一致的维度
    print("\n\n" + "="*60)
    print("CROSS-SEED ANALYSIS")
    print("="*60)
    
    if len(all_key_dims) > 0:
        # 找交集
        common_dims = all_key_dims[0]
        for dims in all_key_dims[1:]:
            common_dims = common_dims.intersection(dims)
        
        common_dims = sorted(list(common_dims))
        
        print(f"\nDimensions critical in ALL seeds: {common_dims}")
        print(f"Total: {len(common_dims)} dimensions")
        
        # 如果交集太小，找出在至少2个种子中出现的维度
        if len(common_dims) < 5:
            dim_counts = {}
            for dims in all_key_dims:
                for d in dims:
                    dim_counts[d] = dim_counts.get(d, 0) + 1
            
            frequent_dims = [d for d, count in dim_counts.items() if count >= 2]
            frequent_dims = sorted(frequent_dims)
            
            print(f"\nDimensions critical in ≥2 seeds: {frequent_dims}")
            print(f"Total: {len(frequent_dims)} dimensions")
            
            # 使用这些作为候选
            if len(frequent_dims) > len(common_dims):
                common_dims = frequent_dims
    
    # 保存结果
    save_results = {
        'analysis_params': {
            'checkpoint_dir': CHECKPOINT_DIR,
            'iteration': iteration,
            'angle_threshold': 45,
            'energy_threshold': 0.1
        },
        'seed_results': {}
    }
    
    for seed, results in all_results.items():
        save_results['seed_results'][seed] = {
            'key_dims': results['key_dims'],
            'num_high_angle_dims': len(results['high_angle_dims']),
            'num_energy_increased_dims': len(results['energy_increased_dims']),
            'max_angle': float(np.max(results['angles_deg'])),
            'mean_angle': float(np.mean(results['angles_deg']))
        }
    
    save_results['common_key_dims'] = common_dims
    
    with open(f'{OUTPUT_DIR}/lm_head_key_dimensions.json', 'w') as f:
        json.dump(save_results, f, indent=2)
    
    print(f"\nResults saved to {OUTPUT_DIR}/lm_head_key_dimensions.json")
    
    # 创建可视化
    if all_results:
        create_visualization(all_results)
    
    # 生成嫁接代码
    if common_dims:
        generate_transplant_code(common_dims)
    
    print("\n\nAnalysis complete!")
    print(f"Found {len(common_dims)} critical dimensions for compositional ability")

if __name__ == "__main__":
    main()