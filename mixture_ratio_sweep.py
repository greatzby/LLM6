# mixture_ratio_sweep.py
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import subprocess
import json
from datetime import datetime

class MixtureRatioSweepAnalyzer:
    def __init__(self, base_config):
        self.base_config = base_config
        self.results = {}
        
    def create_mixed_dataset(self, mixture_ratio):
        """创建特定混合比例的数据集"""
        print(f"\nCreating dataset with {mixture_ratio*100:.1f}% mixture...")
        
        # 调用你的create_mixed_composition_dataset.py
        cmd = [
            'python', 'create_mixed_composition_dataset.py',
            '--original_dir', 'data/simple_graph/composition_90',
            '--s1_s3_ratio', str(mixture_ratio),
            '--seed', '42'
        ]
        
        subprocess.run(cmd, check=True)
        
        # 准备数据
        data_dir = f'data/simple_graph/composition_90_mixed_{int(mixture_ratio*100)}'
        cmd = [
            'python', 'prepare_mixed_data_simple.py'
        ]
        subprocess.run(cmd, check=True)
        
        return data_dir
    
    def train_model(self, data_dir, mixture_ratio):
        """训练模型并收集结果"""
        print(f"\nTraining model with {mixture_ratio*100:.1f}% mixture...")
        
        # 创建输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = f'out/sweep_{int(mixture_ratio*100)}pct_{timestamp}'
        
        # 训练命令
        cmd = [
            'python', 'train_composition_fixed_final.py',
            '--data_dir', data_dir,
            '--n_layer', '1',
            '--n_head', '1', 
            '--n_embd', '120',
            '--max_iters', '50000',
            '--test_interval', '5000',
            '--checkpoint_interval', '10000',
            '--batch_size', '1024'
        ]
        
        # 运行训练
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # 收集结果
        success_rates = []
        for line in process.stdout:
            if "S1->S3:" in line:
                try:
                    rate = float(line.split(':')[1].split('%')[0]) / 100
                    success_rates.append(rate)
                except:
                    pass
        
        process.wait()
        
        return success_rates, out_dir
    
    def analyze_stability(self, success_rates):
        """分析稳定性指标"""
        if len(success_rates) < 3:
            return {'is_stable': False, 'mean': 0, 'std': 1, 'range': 1}
            
        rates = np.array(success_rates[-5:])  # 使用最后5个测量点
        
        return {
            'mean': np.mean(rates),
            'std': np.std(rates),
            'min': np.min(rates),
            'max': np.max(rates),
            'range': np.max(rates) - np.min(rates),
            'is_stable': np.std(rates) < 0.1  # 标准差<10%认为稳定
        }
    
    def run_sweep(self, ratios=None):
        """运行混合比例扫描"""
        if ratios is None:
            ratios = [0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15]
        
        print("="*60)
        print("Running Mixture Ratio Sweep")
        print("="*60)
        
        for ratio in ratios:
            print(f"\n{'='*40}")
            print(f"Testing mixture ratio: {ratio*100:.1f}%")
            print(f"{'='*40}")
            
            if ratio == 0:
                # 使用原始数据
                data_dir = 'data/simple_graph/composition_90'
            else:
                # 创建混合数据
                data_dir = self.create_mixed_dataset(ratio)
            
            # 训练并评估
            success_rates, out_dir = self.train_model(data_dir, ratio)
            
            # 分析稳定性
            stability = self.analyze_stability(success_rates)
            
            self.results[ratio] = {
                'success_rates': success_rates,
                'stability': stability,
                'out_dir': out_dir
            }
            
            print(f"\nResults for {ratio*100:.1f}% mixture:")
            print(f"  Mean success: {stability['mean']:.2%}")
            print(f"  Stability: {'Stable' if stability['is_stable'] else 'Unstable'}")
        
        # 分析和绘图
        self.analyze_results()
    
    def analyze_results(self):
        """分析结果并绘图"""
        # 准备数据
        ratios = sorted(self.results.keys())
        means = [self.results[r]['stability']['mean'] for r in ratios]
        stds = [self.results[r]['stability']['std'] for r in ratios]
        is_stable = [self.results[r]['stability']['is_stable'] for r in ratios]
        
        # 绘制主图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # 成功率图
        ax1.errorbar(ratios, means, yerr=stds, 
                    marker='o', markersize=10, linewidth=2, capsize=5)
        
        # 标记稳定/不稳定
        for i, (r, m, stable) in enumerate(zip(ratios, means, is_stable)):
            color = 'green' if stable else 'red'
            ax1.scatter(r, m, color=color, s=200, alpha=0.5, zorder=5)
        
        ax1.set_xlabel('Mixture Ratio')
        ax1.set_ylabel('S1→S3 Success Rate')
        ax1.set_title('Performance vs Mixture Ratio')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.1)
        
        # 稳定性概率图（Logistic回归）
        if len(ratios) > 3:
            X = np.array(ratios).reshape(-1, 1)
            y = np.array(is_stable).astype(int)
            
            lr = LogisticRegression()
            lr.fit(X, y)
            
            # 生成平滑曲线
            ratios_fine = np.linspace(0, max(ratios), 100)
            probs = lr.predict_proba(ratios_fine.reshape(-1, 1))[:, 1]
            
            ax2.plot(ratios_fine, probs, 'b-', linewidth=2, label='Fitted curve')
            ax2.scatter(ratios, y, color=['red' if not s else 'green' for s in is_stable], 
                       s=100, alpha=0.7, zorder=5)
            
            # 找临界点
            critical_idx = np.argmin(np.abs(probs - 0.5))
            critical_ratio = ratios_fine[critical_idx]
            
            ax2.axvline(critical_ratio, color='red', linestyle='--', 
                       label=f'Critical ratio: {critical_ratio:.3f}')
            ax2.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
            
            ax2.set_xlabel('Mixture Ratio')
            ax2.set_ylabel('Stability Probability')
            ax2.set_title('Stability Analysis')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(-0.1, 1.1)
        
        plt.tight_layout()
        plt.savefig('mixture_ratio_sweep_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存数值结果
        self.save_results()
    
    def save_results(self):
        """保存详细结果"""
        with open('mixture_ratio_sweep_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        with open('mixture_ratio_sweep_summary.txt', 'w') as f:
            f.write("Mixture Ratio Sweep Results\n")
            f.write("="*60 + "\n\n")
            
            for ratio in sorted(self.results.keys()):
                result = self.results[ratio]
                stability = result['stability']
                
                f.write(f"Mixture Ratio: {ratio*100:.1f}%\n")
                f.write("-"*30 + "\n")
                f.write(f"  Mean Success: {stability['mean']:.2%}\n")
                f.write(f"  Std Dev: {stability['std']:.2%}\n")
                f.write(f"  Range: {stability['range']:.2%}\n")
                f.write(f"  Stable: {'Yes' if stability['is_stable'] else 'No'}\n")
                f.write(f"  Output Dir: {result['out_dir']}\n")
                f.write("\n")

def run_mixture_sweep():
    """运行混合比例扫描"""
    # 基础配置
    base_config = {
        'n_layer': 1,
        'n_head': 1,
        'n_embd': 120,
        'max_iters': 50000,
        'batch_size': 1024
    }
    
    analyzer = MixtureRatioSweepAnalyzer(base_config)
    
    # 可以自定义要测试的比例
    # analyzer.run_sweep([0.0, 0.02, 0.05, 0.07, 0.10])
    
    # 或使用默认比例
    analyzer.run_sweep()
    
    print("\nSweep complete! Results saved to:")
    print("  - mixture_ratio_sweep_results.png")
    print("  - mixture_ratio_sweep_results.json")
    print("  - mixture_ratio_sweep_summary.txt")

if __name__ == "__main__":
    run_mixture_sweep()