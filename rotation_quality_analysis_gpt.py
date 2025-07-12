import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import json
import glob
import os

# 配置
CHECKPOINT_DIR = "out"

def get_checkpoint_path(ratio, seed, iteration):
    """构建checkpoint路径 - 处理带时间戳的目录"""
    pattern = f"{CHECKPOINT_DIR}/composition_mix{ratio}_seed{seed}_*"
    dirs = glob.glob(pattern)
    
    if not dirs:
        raise FileNotFoundError(f"No directory found matching: {pattern}")
    
    # 选择最新的目录
    selected_dir = sorted(dirs)[-1]
    checkpoint_path = f"{selected_dir}/ckpt_mix{ratio}_seed{seed}_iter{iteration}.pt"
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    return checkpoint_path

class RotationQualityAnalyzer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_model(self, mix_ratio: int, seed: int, iteration: int):
        """加载模型权重"""
        path = get_checkpoint_path(mix_ratio, seed, iteration)
        print(f"Loading: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        return checkpoint
    
    def extract_weights_for_analysis(self, checkpoint):
        """提取用于分析的权重矩阵"""
        # 检查checkpoint结构
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            # 假设checkpoint直接就是state_dict
            state_dict = checkpoint
        
        # 打印可用的键来调试
        all_keys = list(state_dict.keys())
        print(f"  Found {len(all_keys)} keys in state_dict")
        
        # 选择要分析的权重矩阵
        # 优先选择lm_head权重，因为它直接影响输出
        W = None
        weight_name = None
        
        # 按优先级尝试不同的权重
        weight_candidates = [
            'lm_head.weight',                    # 输出投影层
            'transformer.h.11.mlp.c_proj.weight', # 最后一层的MLP投影
            'transformer.h.11.attn.c_proj.weight',# 最后一层的注意力投影
            'transformer.h.0.mlp.c_proj.weight',  # 第一层的MLP投影
            'transformer.wte.weight',             # 词嵌入层
        ]
        
        for candidate in weight_candidates:
            if candidate in state_dict:
                W = state_dict[candidate].cpu().numpy()
                weight_name = candidate
                print(f"  Using weight matrix: {candidate}, shape: {W.shape}")
                break
        
        # 如果还没找到，尝试找任何包含'weight'的2D张量
        if W is None:
            for key in all_keys:
                if 'weight' in key and len(state_dict[key].shape) == 2:
                    W = state_dict[key].cpu().numpy()
                    weight_name = key
                    print(f"  Using weight matrix: {key}, shape: {W.shape}")
                    break
        
        if W is None:
            raise KeyError(f"Cannot find suitable weight matrix. Available keys: {all_keys[:20]}...")
        
        # 确保矩阵不是太大（如果是embedding矩阵，可能需要转置或截取）
        if W.shape[0] > 10000:  # 可能是词汇表大小
            print(f"  Note: Weight matrix has large first dimension ({W.shape[0]}), likely vocabulary size")
            # 对于embedding矩阵，我们可能更关心embedding维度的结构
            if W.shape[1] < W.shape[0]:  # 确认这是 [vocab_size, embed_dim] 格式
                W = W.T  # 转置成 [embed_dim, vocab_size]
                print(f"  Transposed to shape: {W.shape}")
        
        return W, weight_name
    
    def compute_svd(self, W):
        """计算SVD分解"""
        U, S, Vt = np.linalg.svd(W, full_matrices=False)
        return U, S, Vt.T
    
    def compute_effective_rank(self, S, threshold=0.99):
        """计算有效秩"""
        if len(S) == 0 or S[0] == 0:
            return 0
        S_squared = S**2
        S_squared_norm = S_squared / np.sum(S_squared)
        S_cumsum = np.cumsum(S_squared_norm)
        return np.sum(S_cumsum < threshold) + 1
    
    def analyze_rotation_quality(self, mix_ratio: int, seed: int):
        """主要分析函数：分析旋转质量"""
        print(f"\n{'='*60}")
        print(f"Analyzing rotation quality for {mix_ratio}% mix, seed {seed}")
        print('='*60)
        
        # 加载初始和最终模型
        try:
            init_ckpt = self.load_model(mix_ratio, seed, 3000)
            final_ckpt = self.load_model(mix_ratio, seed, 50000)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return None
        
        W_init, weight_name_init = self.extract_weights_for_analysis(init_ckpt)
        W_final, weight_name_final = self.extract_weights_for_analysis(final_ckpt)
        
        # 确保使用相同的权重矩阵
        assert weight_name_init == weight_name_final, f"Weight names don't match: {weight_name_init} vs {weight_name_final}"
        
        print(f"  Analyzing weight: {weight_name_init}")
        print(f"  Weight matrix shape: {W_init.shape}")
        
        # SVD分解
        U_init, S_init, V_init = self.compute_svd(W_init)
        U_final, S_final, V_final = self.compute_svd(W_final)
        
        # 1. 分析奇异值分布变化
        print("\n1. Singular Value Distribution:")
        print(f"   Initial top 10 S: {S_init[:10]}")
        print(f"   Final top 10 S: {S_final[:10]}")
        if len(S_init) >= 10 and len(S_final) >= 10:
            print(f"   S ratio (final/init) top 10: {S_final[:10] / (S_init[:10] + 1e-10)}")
        
        # 2. 计算子空间差异
        print("\n2. Subspace Difference Analysis:")
        
        # 分析列空间（更重要，因为它决定了输出空间的结构）
        # 计算V_final中不能被V_init表示的部分
        V_init_trunc = V_init[:, :min(V_init.shape[1], V_final.shape[1])]
        V_final_trunc = V_final[:, :min(V_init.shape[1], V_final.shape[1])]
        
        # 投影并计算差异
        V_final_proj = V_init_trunc @ (V_init_trunc.T @ V_final_trunc)
        V_diff = V_final_trunc - V_final_proj
        
        # 对差异进行SVD
        _, S_diff, _ = np.linalg.svd(V_diff, full_matrices=False)
        
        # 使用阈值0.1来确定显著旋转的维度
        rotated_dims = np.where(S_diff > 0.1)[0]
        print(f"   Rotated dimensions (threshold=0.1): {len(rotated_dims)} dims")
        if len(rotated_dims) > 0:
            print(f"   S_diff values for rotated dims: {S_diff[rotated_dims[:5]]}...")
        
        # 3. 能量分析
        print("\n3. Energy Analysis:")
        
        # 计算总能量
        total_energy_init = np.sum(S_init**2)
        total_energy_final = np.sum(S_final**2)
        
        print(f"   Total energy: init={total_energy_init:.2f}, final={total_energy_final:.2f}")
        
        # 计算前k个奇异值的能量占比
        k_values = [10, 20, 30, 50]
        print("   Energy concentration (cumulative %):")
        for k in k_values:
            if k <= len(S_init) and k <= len(S_final):
                energy_init_k = np.sum(S_init[:k]**2) / total_energy_init * 100
                energy_final_k = np.sum(S_final[:k]**2) / total_energy_final * 100
                print(f"     Top {k:2d}: init={energy_init_k:5.1f}%, final={energy_final_k:5.1f}%, change={energy_final_k-energy_init_k:+5.1f}%")
        
        # 4. 维度活跃度分析
        print("\n4. Dimension Activity Analysis:")
        
        # 计算有效维度数
        threshold_ratios = [0.1, 0.05, 0.01]
        for thr in threshold_ratios:
            if S_init[0] > 0 and S_final[0] > 0:
                active_dims_init = np.sum(S_init > S_init[0] * thr)
                active_dims_final = np.sum(S_final > S_final[0] * thr)
                print(f"   Active dims (>{thr*100:.0f}% of max): init={active_dims_init}, final={active_dims_final}, change={active_dims_final - active_dims_init}")
        
        # 5. 子空间对齐度
        print("\n5. Subspace Alignment:")
        
        # 计算主要奇异向量的对齐度
        alignment_scores = []
        k_align = min(20, V_init_trunc.shape[1], V_final_trunc.shape[1])
        
        for i in range(k_align):
            alignment = np.abs(np.dot(V_init_trunc[:, i], V_final_trunc[:, i]))
            alignment_scores.append(alignment)
        
        if alignment_scores:
            avg_alignment_10 = np.mean(alignment_scores[:min(10, len(alignment_scores))])
            avg_alignment_all = np.mean(alignment_scores)
            print(f"   Average alignment (top 10 modes): {avg_alignment_10:.4f}")
            print(f"   Average alignment (all {k_align} modes): {avg_alignment_all:.4f}")
        else:
            avg_alignment_10 = 0
            avg_alignment_all = 0
        
        # 6. 权重变化统计
        print("\n6. Weight Change Statistics:")
        W_diff = W_final - W_init
        relative_change = np.linalg.norm(W_diff, 'fro') / (np.linalg.norm(W_init, 'fro') + 1e-10)
        print(f"   Relative Frobenius norm change: {relative_change:.4f}")
        print(f"   Max absolute change: {np.max(np.abs(W_diff)):.4f}")
        print(f"   Mean absolute change: {np.mean(np.abs(W_diff)):.4f}")
        
        # 计算有效秩
        er_init = self.compute_effective_rank(S_init)
        er_final = self.compute_effective_rank(S_final)
        print(f"   Effective Rank: init={er_init:.1f}, final={er_final:.1f}, change={er_final-er_init:.1f}")
        
        # 返回结果
        return {
            'mix_ratio': mix_ratio,
            'seed': seed,
            'weight_name': weight_name_init,
            'weight_shape': W_init.shape,
            'rotated_dims': len(rotated_dims),
            'energy_concentration_20': np.sum(S_final[:20]**2) / total_energy_final if len(S_final) >= 20 else 1.0,
            'active_dims_01': np.sum(S_final > S_final[0] * 0.01) if S_final[0] > 0 else 0,
            'active_dims_change': (np.sum(S_final > S_final[0] * 0.01) - np.sum(S_init > S_init[0] * 0.01)) if S_init[0] > 0 and S_final[0] > 0 else 0,
            'avg_alignment_10': avg_alignment_10,
            'relative_change': relative_change,
            'er_init': er_init,
            'er_final': er_final,
            'er_change': er_final - er_init,
            'S_init': S_init,
            'S_final': S_final,
            'S_diff': S_diff[:30] if len(S_diff) >= 30 else S_diff
        }
    
    def gradient_alignment_test(self, mix_ratio: int, seed: int):
        """测试梯度对齐度（使用权重变化作为代理）"""
        print(f"\n{'='*60}")
        print(f"Gradient Alignment Test for {mix_ratio}% mix, seed {seed}")
        print('='*60)
        
        try:
            # 加载三个时间点的权重
            ckpt_3k = self.load_model(mix_ratio, seed, 3000)
            ckpt_10k = self.load_model(mix_ratio, seed, 10000)
            ckpt_50k = self.load_model(mix_ratio, seed, 50000)
            
            W_3k, _ = self.extract_weights_for_analysis(ckpt_3k)
            W_10k, _ = self.extract_weights_for_analysis(ckpt_10k)
            W_50k, _ = self.extract_weights_for_analysis(ckpt_50k)
        except Exception as e:
            print(f"Error: {e}")
            return None
        
        # 计算权重变化方向
        delta_early = W_10k - W_3k
        delta_late = W_50k - W_10k
        
        # 归一化
        norm_early = np.linalg.norm(delta_early)
        norm_late = np.linalg.norm(delta_late)
        
        if norm_early > 1e-10 and norm_late > 1e-10:
            delta_early_norm = delta_early / norm_early
            delta_late_norm = delta_late / norm_late
            
            # 计算对齐度（余弦相似度）
            alignment = np.sum(delta_early_norm * delta_late_norm)
            
            print(f"   Early-Late gradient alignment: {alignment:.4f}")
            print(f"   Interpretation: {'Consistent' if alignment > 0.5 else 'Divergent'} learning direction")
            
            # 额外分析：变化速度
            change_rate_early = norm_early / 7000  # 3k到10k
            change_rate_late = norm_late / 40000   # 10k到50k
            
            print(f"   Change magnitude early: {norm_early:.4f}")
            print(f"   Change magnitude late: {norm_late:.4f}")
            print(f"   Change rate early: {change_rate_early:.6f}")
            print(f"   Change rate late: {change_rate_late:.6f}")
            if change_rate_late > 1e-10:
                print(f"   Deceleration factor: {change_rate_early/change_rate_late:.2f}x")
        else:
            alignment = 0
            print(f"   Warning: Changes too small to compute alignment")
        
        return alignment
    
    def comparative_analysis(self):
        """对比分析0% mix和20% mix的所有种子"""
        results = []
        
        # 分析所有配置
        for mix_ratio in [0, 20]:
            for seed in [42, 123, 456]:
                result = self.analyze_rotation_quality(mix_ratio, seed)
                if result is not None:
                    gradient_align = self.gradient_alignment_test(mix_ratio, seed)
                    if gradient_align is not None:
                        result['gradient_alignment'] = gradient_align
                    results.append(result)
        
        # 汇总结果
        print("\n" + "="*80)
        print("SUMMARY: Rotation Quality Comparison")
        print("="*80)
        
        # 按mix_ratio分组
        mix0_results = [r for r in results if r['mix_ratio'] == 0]
        mix20_results = [r for r in results if r['mix_ratio'] == 20]
        
        if mix0_results:
            print("\n0% Mix Average:")
            print(f"  Analyzed weight: {mix0_results[0]['weight_name']}")
            print(f"  Rotated dimensions: {np.mean([r['rotated_dims'] for r in mix0_results]):.1f}")
            print(f"  Energy in top 20 dims: {np.mean([r['energy_concentration_20'] for r in mix0_results])*100:.1f}%")
            print(f"  Active dims change: {np.mean([r['active_dims_change'] for r in mix0_results]):.1f}")
            print(f"  Top-10 mode alignment: {np.mean([r['avg_alignment_10'] for r in mix0_results]):.4f}")
            if 'gradient_alignment' in mix0_results[0]:
                print(f"  Gradient alignment: {np.mean([r.get('gradient_alignment', 0) for r in mix0_results]):.4f}")
            print(f"  Relative weight change: {np.mean([r['relative_change'] for r in mix0_results]):.4f}")
            print(f"  ER change: {np.mean([r['er_change'] for r in mix0_results]):.1f}")
        
        if mix20_results:
            print("\n20% Mix Average:")
            print(f"  Analyzed weight: {mix20_results[0]['weight_name']}")
            print(f"  Rotated dimensions: {np.mean([r['rotated_dims'] for r in mix20_results]):.1f}")
            print(f"  Energy in top 20 dims: {np.mean([r['energy_concentration_20'] for r in mix20_results])*100:.1f}%")
            print(f"  Active dims change: {np.mean([r['active_dims_change'] for r in mix20_results]):.1f}")
            print(f"  Top-10 mode alignment: {np.mean([r['avg_alignment_10'] for r in mix20_results]):.4f}")
            if 'gradient_alignment' in mix20_results[0]:
                print(f"  Gradient alignment: {np.mean([r.get('gradient_alignment', 0) for r in mix20_results]):.4f}")
            print(f"  Relative weight change: {np.mean([r['relative_change'] for r in mix20_results]):.4f}")
            print(f"  ER change: {np.mean([r['er_change'] for r in mix20_results]):.1f}")
        
        # 关键对比
        if mix0_results and mix20_results:
            print("\n" + "="*60)
            print("KEY DIFFERENCES (0% vs 20% mix):")
            print("="*60)
            
            # ER变化对比
            er_change_0 = np.mean([r['er_change'] for r in mix0_results])
            er_change_20 = np.mean([r['er_change'] for r in mix20_results])
            print(f"\nER change: 0% mix = {er_change_0:.1f}, 20% mix = {er_change_20:.1f}")
            if er_change_0 < 0:
                protection = (1 - abs(er_change_20)/abs(er_change_0)) * 100
                print(f"Protection effect: {protection:.1f}% less ER reduction with 20% mix")
            
            # 对齐度对比
            align_0 = np.mean([r['avg_alignment_10'] for r in mix0_results])
            align_20 = np.mean([r['avg_alignment_10'] for r in mix20_results])
            print(f"\nMode alignment (top-10): 0% = {align_0:.4f}, 20% = {align_20:.4f}")
            if align_0 > 0:
                print(f"Alignment change: {(align_20/align_0 - 1)*100:+.1f}% with 20% mix")
            
            # 旋转维度对比
            rot_0 = np.mean([r['rotated_dims'] for r in mix0_results])
            rot_20 = np.mean([r['rotated_dims'] for r in mix20_results])
            print(f"\nRotated dimensions: 0% = {rot_0:.1f}, 20% = {rot_20:.1f}")
            
            # 梯度一致性
            if 'gradient_alignment' in mix0_results[0]:
                grad_0 = np.mean([r.get('gradient_alignment', 0) for r in mix0_results])
                grad_20 = np.mean([r.get('gradient_alignment', 0) for r in mix20_results])
                print(f"\nGradient consistency: 0% = {grad_0:.4f}, 20% = {grad_20:.4f}")
                print(f"Learning stability: {'20% mix' if grad_20 > grad_0 else '0% mix'} is more stable")
        
        # 可视化
        self.visualize_results(results)
        
        return results
    
    def visualize_results(self, results):
        """可视化分析结果"""
        if not results:
            print("No results to visualize")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 准备数据
        mix0_data = [r for r in results if r['mix_ratio'] == 0]
        mix20_data = [r for r in results if r['mix_ratio'] == 20]
        
        # 1. ER变化对比
        ax = axes[0, 0]
        if mix0_data and mix20_data:
            seeds = [42, 123, 456]
            er_changes_0 = []
            er_changes_20 = []
            
            for s in seeds:
                er_0 = next((r['er_change'] for r in mix0_data if r['seed'] == s), None)
                er_20 = next((r['er_change'] for r in mix20_data if r['seed'] == s), None)
                if er_0 is not None:
                    er_changes_0.append(er_0)
                if er_20 is not None:
                    er_changes_20.append(er_20)
            
            x = np.arange(len(er_changes_0))
            width = 0.35
            
            ax.bar(x - width/2, er_changes_0, width, label='0% mix', color='#ff7f0e')
            ax.bar(x + width/2, er_changes_20[:len(er_changes_0)], width, label='20% mix', color='#2ca02c')
            
            ax.set_xlabel('Seed')
            ax.set_ylabel('ER Change (final - initial)')
            ax.set_title('Effective Rank Change During Training')
            ax.set_xticks(x)
            ax.set_xticklabels(seeds[:len(er_changes_0)])
            ax.legend()
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # 2. 模式对齐度
        ax = axes[0, 1]
        if mix0_data and mix20_data:
            align_0 = [r['avg_alignment_10'] for r in mix0_data]
            align_20 = [r['avg_alignment_10'] for r in mix20_data]
            
            if align_0 and align_20:
                ax.boxplot([align_0, align_20], labels=['0% mix', '20% mix'])
                ax.set_ylabel('Average Alignment')
                ax.set_title('Top-10 Mode Alignment')
                ax.set_ylim([0, 1])
        
        # 3. 奇异值衰减曲线（第一个种子）
        ax = axes[1, 0]
        if mix0_data and mix20_data:
            # 找到第一个完整的数据
            data_0 = next((r for r in mix0_data if 'S_init' in r and 'S_final' in r), None)
            data_20 = next((r for r in mix20_data if 'S_init' in r and 'S_final' in r), None)
            
            if data_0 and data_20:
                n_show = min(50, len(data_0['S_init']), len(data_0['S_final']),
                            len(data_20['S_init']), len(data_20['S_final']))
                
                ax.semilogy(data_0['S_init'][:n_show], 'b-', label='0% mix initial', linewidth=2, alpha=0.7)
                ax.semilogy(data_0['S_final'][:n_show], 'b--', label='0% mix final', linewidth=2)
                ax.semilogy(data_20['S_init'][:n_show], 'g-', label='20% mix initial', linewidth=2, alpha=0.7)
                ax.semilogy(data_20['S_final'][:n_show], 'g--', label='20% mix final', linewidth=2)
                
                ax.set_xlabel('Dimension')
                ax.set_ylabel('Singular Value')
                ax.set_title(f'Singular Value Decay (seed {data_0["seed"]})')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # 4. 旋转维度数
        ax = axes[1, 1]
        if mix0_data and mix20_data:
            seeds = [42, 123, 456]
            rot_dims_0 = []
            rot_dims_20 = []
            
            for s in seeds:
                rot_0 = next((r['rotated_dims'] for r in mix0_data if r['seed'] == s), None)
                rot_20 = next((r['rotated_dims'] for r in mix20_data if r['seed'] == s), None)
                if rot_0 is not None:
                    rot_dims_0.append(rot_0)
                if rot_20 is not None:
                    rot_dims_20.append(rot_20)
            
            if rot_dims_0 and rot_dims_20:
                x = np.arange(len(rot_dims_0))
                width = 0.35
                
                ax.bar(x - width/2, rot_dims_0, width, label='0% mix', color='#ff7f0e')
                ax.bar(x + width/2, rot_dims_20[:len(rot_dims_0)], width, label='20% mix', color='#2ca02c')
                
                ax.set_xlabel('Seed')
                ax.set_ylabel('Number of Rotated Dimensions')
                ax.set_title('Subspace Rotation (threshold=0.1)')
                ax.set_xticks(x)
                ax.set_xticklabels(seeds[:len(rot_dims_0)])
                ax.legend()
        
        plt.tight_layout()
        plt.savefig('rotation_quality_analysis_gpt.png', dpi=300, bbox_inches='tight')
        print("\nVisualization saved as 'rotation_quality_analysis_gpt.png'")
        plt.close()

## 在main函数中，修改JSON保存部分：
def main():
    analyzer = RotationQualityAnalyzer()
    
    print("Starting Rotation Quality Analysis for GPT Model...")
    print("This will analyze how weight matrices change between 0% and 20% mix training")
    
    # 运行完整的对比分析
    results = analyzer.comparative_analysis()
    
    # 保存结果
    if results:
        with open('rotation_analysis_results_gpt.json', 'w') as f:
            # 转换numpy数组为列表以便JSON序列化
            json_results = []
            for r in results:
                json_r = {}
                for k, v in r.items():
                    if k not in ['S_init', 'S_final', 'S_diff']:
                        if isinstance(v, np.ndarray):
                            json_r[k] = v.tolist()
                        elif isinstance(v, (np.float32, np.float64)):
                            json_r[k] = float(v)
                        elif isinstance(v, (np.int32, np.int64)):
                            json_r[k] = int(v)
                        else:
                            json_r[k] = v
                json_results.append(json_r)
            json.dump(json_results, f, indent=2)
        
        print("\nAnalysis complete! Results saved to 'rotation_analysis_results_gpt.json'")
    else:
        print("\nNo results generated. Please check your checkpoints.")

if __name__ == "__main__":
    main()