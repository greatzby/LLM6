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
    
    def extract_hidden_weights(self, checkpoint):
        """提取隐藏层权重"""
        W_ih = checkpoint['model_state_dict']['rnn.weight_ih_l0'].cpu().numpy()
        W_hh = checkpoint['model_state_dict']['rnn.weight_hh_l0'].cpu().numpy()
        # 使用hidden-to-hidden权重作为主要分析对象
        return W_hh
    
    def compute_svd(self, W):
        """计算SVD分解"""
        U, S, Vt = np.linalg.svd(W, full_matrices=False)
        return U, S, Vt.T
    
    def compute_effective_rank(self, S, threshold=0.01):
        """计算有效秩"""
        S_norm = S / S[0]
        S_cum = np.cumsum(S_norm**2) / np.sum(S_norm**2)
        return np.sum(S_cum < 0.99) + 1
    
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
        
        W_init = self.extract_hidden_weights(init_ckpt)
        W_final = self.extract_hidden_weights(final_ckpt)
        
        # SVD分解
        U_init, S_init, V_init = self.compute_svd(W_init)
        U_final, S_final, V_final = self.compute_svd(W_final)
        
        # 1. 分析奇异值分布变化
        print("\n1. Singular Value Distribution:")
        print(f"   Initial top 10 S: {S_init[:10]}")
        print(f"   Final top 10 S: {S_final[:10]}")
        print(f"   S ratio (final/init) top 10: {S_final[:10] / S_init[:10]}")
        
        # 2. 计算旋转的28个维度
        V_diff = V_final - V_init @ (V_init.T @ V_final)
        U_diff, S_diff, _ = np.linalg.svd(V_diff.T @ V_diff)
        rotated_dims = np.where(S_diff > 0.1)[0]
        print(f"\n2. Rotated dimensions: {len(rotated_dims)} dims")
        print(f"   S_diff values for rotated dims: {S_diff[rotated_dims][:5]}...")
        
        # 3. 分析这些维度上的能量变化
        print("\n3. Energy on rotated dimensions:")
        
        # 计算初始和最终权重在旋转维度上的能量
        energy_init = 0
        energy_final = 0
        
        # 对每个旋转维度计算能量
        for i, dim in enumerate(rotated_dims[:28]):  # 只看前28个主要旋转维度
            # 计算第dim个右奇异向量对应的能量
            # 能量 = 对应奇异值的平方
            if dim < len(S_init):
                energy_init += S_init[dim]**2
            if dim < len(S_final):
                energy_final += S_final[dim]**2
        
        # 总能量
        total_energy_init = np.sum(S_init**2)
        total_energy_final = np.sum(S_final**2)
        
        # 旋转维度上的能量占比
        energy_ratio_init = energy_init / total_energy_init
        energy_ratio_final = energy_final / total_energy_final
        
        print(f"   Energy fraction on rotated dims (init): {energy_ratio_init:.4f}")
        print(f"   Energy fraction on rotated dims (final): {energy_ratio_final:.4f}")
        print(f"   Energy ratio change: {energy_ratio_final/energy_ratio_init:.4f}")
        
        # 4. 维度稀疏性分析
        print("\n4. Dimension Sparsity Analysis:")
        
        # 计算激活模式（使用奇异值作为代理）
        active_dims_init = np.sum(S_init > S_init[0] * 0.1)
        active_dims_final = np.sum(S_final > S_final[0] * 0.1)
        
        print(f"   Active dims initial: {active_dims_init}")
        print(f"   Active dims final: {active_dims_final}")
        print(f"   Active dims change: {active_dims_final - active_dims_init}")
        
        # 5. 计算功能保持度指标
        print("\n5. Functional Preservation Score:")
        
        # 基于奇异向量的对齐度
        alignment_scores = []
        for i in range(min(30, len(S_init))):
            # 计算每个主要模式的对齐度
            alignment = np.abs(np.dot(V_init[:, i], V_final[:, i]))
            alignment_scores.append(alignment)
        
        avg_alignment = np.mean(alignment_scores[:20])  # 前20个主要模式
        print(f"   Average alignment (top 20 modes): {avg_alignment:.4f}")
        
        # 6. 权重矩阵的直接比较
        print("\n6. Weight Matrix Direct Comparison:")
        W_diff = W_final - W_init
        relative_change = np.linalg.norm(W_diff, 'fro') / np.linalg.norm(W_init, 'fro')
        print(f"   Relative Frobenius norm change: {relative_change:.4f}")
        
        # 返回结果用于对比
        return {
            'mix_ratio': mix_ratio,
            'seed': seed,
            'energy_ratio_change': energy_ratio_final/energy_ratio_init,
            'energy_fraction_init': energy_ratio_init,
            'energy_fraction_final': energy_ratio_final,
            'active_dims_change': active_dims_final - active_dims_init,
            'avg_alignment': avg_alignment,
            'rotated_dims': len(rotated_dims),
            'relative_change': relative_change,
            'S_init': S_init,
            'S_final': S_final
        }
    
    def gradient_alignment_test(self, mix_ratio: int, seed: int):
        """测试梯度对齐度（需要实际的合成数据）"""
        print(f"\n{'='*60}")
        print(f"Gradient Alignment Test for {mix_ratio}% mix, seed {seed}")
        print('='*60)
        
        try:
            # 这里我们使用权重变化作为梯度的代理
            W_init = self.extract_hidden_weights(self.load_model(mix_ratio, seed, 3000))
            W_mid = self.extract_hidden_weights(self.load_model(mix_ratio, seed, 10000))
            W_final = self.extract_hidden_weights(self.load_model(mix_ratio, seed, 50000))
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return None
        
        # 计算权重变化方向
        delta_early = W_mid - W_init
        delta_late = W_final - W_mid
        
        # 归一化
        delta_early_norm = delta_early / (np.linalg.norm(delta_early) + 1e-8)
        delta_late_norm = delta_late / (np.linalg.norm(delta_late) + 1e-8)
        
        # 计算对齐度
        alignment = np.sum(delta_early_norm * delta_late_norm)
        
        print(f"   Early-Late gradient alignment: {alignment:.4f}")
        print(f"   Interpretation: {'Consistent' if alignment > 0.5 else 'Divergent'} learning direction")
        
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
            print(f"  Energy ratio change: {np.mean([r['energy_ratio_change'] for r in mix0_results]):.4f}")
            print(f"  Energy fraction on rotated dims (init): {np.mean([r['energy_fraction_init'] for r in mix0_results]):.4f}")
            print(f"  Energy fraction on rotated dims (final): {np.mean([r['energy_fraction_final'] for r in mix0_results]):.4f}")
            print(f"  Active dims change: {np.mean([r['active_dims_change'] for r in mix0_results]):.1f}")
            print(f"  Mode alignment: {np.mean([r['avg_alignment'] for r in mix0_results]):.4f}")
            if 'gradient_alignment' in mix0_results[0]:
                print(f"  Gradient alignment: {np.mean([r['gradient_alignment'] for r in mix0_results]):.4f}")
            print(f"  Relative weight change: {np.mean([r['relative_change'] for r in mix0_results]):.4f}")
        
        if mix20_results:
            print("\n20% Mix Average:")
            print(f"  Energy ratio change: {np.mean([r['energy_ratio_change'] for r in mix20_results]):.4f}")
            print(f"  Energy fraction on rotated dims (init): {np.mean([r['energy_fraction_init'] for r in mix20_results]):.4f}")
            print(f"  Energy fraction on rotated dims (final): {np.mean([r['energy_fraction_final'] for r in mix20_results]):.4f}")
            print(f"  Active dims change: {np.mean([r['active_dims_change'] for r in mix20_results]):.1f}")
            print(f"  Mode alignment: {np.mean([r['avg_alignment'] for r in mix20_results]):.4f}")
            if 'gradient_alignment' in mix20_results[0]:
                print(f"  Gradient alignment: {np.mean([r['gradient_alignment'] for r in mix20_results]):.4f}")
            print(f"  Relative weight change: {np.mean([r['relative_change'] for r in mix20_results]):.4f}")
        
        # 可视化
        self.visualize_results(results)
        
        return results
    
    def visualize_results(self, results):
        """可视化分析结果"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 准备数据
        mix0_data = [r for r in results if r['mix_ratio'] == 0]
        mix20_data = [r for r in results if r['mix_ratio'] == 20]
        
        # 1. Energy Ratio Change对比
        ax = axes[0, 0]
        if mix0_data and mix20_data:
            ax.boxplot([
                [r['energy_ratio_change'] for r in mix0_data],
                [r['energy_ratio_change'] for r in mix20_data]
            ], labels=['0% mix', '20% mix'])
            ax.set_ylabel('Energy Ratio Change')
            ax.set_title('Energy Change on Rotated Dimensions')
            ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
        
        # 2. Mode Alignment对比
        ax = axes[0, 1]
        if mix0_data and mix20_data:
            ax.boxplot([
                [r['avg_alignment'] for r in mix0_data],
                [r['avg_alignment'] for r in mix20_data]
            ], labels=['0% mix', '20% mix'])
            ax.set_ylabel('Average Alignment')
            ax.set_title('Principal Mode Alignment')
            ax.axhline(y=0.8, color='r', linestyle='--', alpha=0.5)
        
        # 3. 奇异值衰减曲线
        ax = axes[1, 0]
        # 使用第一个种子的数据作为示例
        if mix0_data:
            S_init = mix0_data[0]['S_init'][:30]
            S_final = mix0_data[0]['S_final'][:30]
            ax.semilogy(S_init, 'b-', label='0% mix initial', linewidth=2)
            ax.semilogy(S_final, 'b--', label='0% mix final', linewidth=2)
        
        if mix20_data:
            S_init = mix20_data[0]['S_init'][:30]
            S_final = mix20_data[0]['S_final'][:30]
            ax.semilogy(S_init, 'r-', label='20% mix initial', linewidth=2)
            ax.semilogy(S_final, 'r--', label='20% mix final', linewidth=2)
        
        ax.set_xlabel('Dimension')
        ax.set_ylabel('Singular Value')
        ax.set_title('Singular Value Decay')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Gradient Alignment
        ax = axes[1, 1]
        if mix0_data and mix20_data and 'gradient_alignment' in mix0_data[0]:
            ax.boxplot([
                [r['gradient_alignment'] for r in mix0_data],
                [r['gradient_alignment'] for r in mix20_data]
            ], labels=['0% mix', '20% mix'])
            ax.set_ylabel('Gradient Alignment')
            ax.set_title('Learning Direction Consistency')
            ax.axhline(y=0.0, color='r', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig('rotation_quality_analysis.png', dpi=300, bbox_inches='tight')
        print("\nVisualization saved as 'rotation_quality_analysis.png'")
        plt.show()

# 运行分析的主函数
def main():
    analyzer = RotationQualityAnalyzer()
    
    print("Starting Rotation Quality Analysis...")
    print("This will analyze how the 28-dimensional rotation differs between 0% and 20% mix")
    
    # 运行完整的对比分析
    results = analyzer.comparative_analysis()
    
    # 保存结果
    with open('rotation_analysis_results.json', 'w') as f:
        # 转换numpy数组为列表以便JSON序列化
        json_results = []
        for r in results:
            json_r = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                     for k, v in r.items() if k not in ['S_init', 'S_final']}
            json_results.append(json_r)
        json.dump(json_results, f, indent=2)
    
    print("\nAnalysis complete! Results saved to 'rotation_analysis_results.json'")

if __name__ == "__main__":
    main()