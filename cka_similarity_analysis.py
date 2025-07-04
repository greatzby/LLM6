# cka_similarity_analysis.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from model import GPTConfig, GPT
from sklearn.metrics.pairwise import linear_kernel
from tqdm import tqdm

class CKASimilarityAnalyzer:
    def __init__(self, device='cuda:0'):
        self.device = device
        self.representations_cache = {}
        self.results = {}
        
    def compute_cka(self, X, Y):
        """计算CKA (Centered Kernel Alignment)"""
        # 使用线性核
        K_X = linear_kernel(X)
        K_Y = linear_kernel(Y)
        
        # 中心化
        n = K_X.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        K_X_centered = H @ K_X @ H
        K_Y_centered = H @ K_Y @ H
        
        # 计算CKA
        hsic_xy = np.trace(K_X_centered @ K_Y_centered)
        hsic_xx = np.trace(K_X_centered @ K_X_centered)
        hsic_yy = np.trace(K_Y_centered @ K_Y_centered)
        
        cka = hsic_xy / (np.sqrt(hsic_xx) * np.sqrt(hsic_yy) + 1e-8)
        return cka
    
    def get_s2_representations(self, checkpoint_path, data_dir):
        """获取S2节点的表示"""
        # 加载模型
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config = GPTConfig(**checkpoint['model_args'])
        model = GPT(config).to(self.device)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        
        # 加载阶段信息
        with open(os.path.join(data_dir, 'stage_info.pkl'), 'rb') as f:
            stage_info = pickle.load(f)
        S2 = stage_info['stages'][1]
        
        representations = []
        with torch.no_grad():
            for node in S2:
                # 构造输入序列
                node_id = node + 2
                input_ids = torch.tensor([[node_id]], device=self.device)
                
                # 获取深层表示
                tok_emb = model.transformer.wte(input_ids)
                pos = torch.zeros(1, 1, dtype=torch.long, device=self.device)
                pos_emb = model.transformer.wpe(pos)
                x = model.transformer.drop(tok_emb + pos_emb)
                
                # 通过所有transformer层
                for block in model.transformer.h:
                    x = block(x)
                x = model.transformer.ln_f(x)
                
                # 使用最后一层的输出作为表示
                representations.append(x.squeeze().cpu().numpy())
        
        return np.array(representations)
    
    def load_test_results(self, log_file):
        """从训练日志中提取测试结果"""
        results = {}
        
        if not os.path.exists(log_file):
            return results
            
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        current_iter = None
        for line in lines:
            if "Iteration" in line and line.strip().startswith("Iteration"):
                try:
                    current_iter = int(line.split()[1])
                except:
                    continue
            
            if "S1->S3:" in line and current_iter is not None:
                try:
                    # 提取准确率
                    parts = line.split(':')[1].strip()
                    accuracy = float(parts.split('%')[0]) / 100
                    results[current_iter] = accuracy
                except:
                    continue
        
        return results
    
    def analyze_drift_rate(self):
        """分析表示漂移率"""
        model_configs = {
            'original': {
                'checkpoint_dir': 'out/composition_20250702_063926',
                'data_dir': 'data/simple_graph/composition_90',
                'log_file': 'out/composition_20250702_063926/train.log'
            },
            '5% mixed': {
                'checkpoint_dir': 'out/composition_20250703_004537',
                'data_dir': 'data/simple_graph/composition_90_mixed_5',
                'log_file': 'out/composition_20250703_004537/train.log'
            },
            '10% mixed': {
                'checkpoint_dir': 'out/composition_20250703_011304',
                'data_dir': 'data/simple_graph/composition_90_mixed_10',
                'log_file': 'out/composition_20250703_004537/train.log'
            }
        }
        
        iterations = [5000, 15000, 25000, 35000, 50000]
        
        print("Computing CKA Similarity...")
        print("="*60)
        
        for model_name, config in model_configs.items():
            print(f"\nAnalyzing {model_name}...")
            
            # 获取测试结果
            test_results = self.load_test_results(config['log_file'])
            
            # 获取各时间点的表示
            representations = {}
            for iter in iterations:
                checkpoint_path = os.path.join(config['checkpoint_dir'], f'ckpt_{iter}.pt')
                if os.path.exists(checkpoint_path):
                    representations[iter] = self.get_s2_representations(
                        checkpoint_path, config['data_dir']
                    )
            
            # 计算CKA
            cka_scores = []
            iter_pairs = []
            
            for i in range(len(iterations) - 1):
                if iterations[i] in representations and iterations[i+1] in representations:
                    cka = self.compute_cka(
                        representations[iterations[i]], 
                        representations[iterations[i+1]]
                    )
                    cka_scores.append(cka)
                    iter_pairs.append((iterations[i], iterations[i+1]))
            
            # 获取对应的成功率
            success_rates = [test_results.get(iter, 0) for iter in iterations]
            
            self.results[model_name] = {
                'iterations': iterations,
                'cka_scores': cka_scores,
                'iter_pairs': iter_pairs,
                'success_rates': success_rates,
                'drift_rate': 1 - np.mean(cka_scores) if cka_scores else 0
            }
            
            print(f"  Average drift rate: {self.results[model_name]['drift_rate']:.3f}")
    
    def plot_results(self):
        """绘制结果"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        
        colors = {'original': 'blue', '5% mixed': 'orange', '10% mixed': 'green'}
        
        for model_name, data in self.results.items():
            if not data['cka_scores']:
                continue
                
            # CKA scores
            cka_iters = [(pair[0] + pair[1]) / 2 for pair in data['iter_pairs']]
            ax1.plot(cka_iters, data['cka_scores'], 
                    marker='o', label=f'{model_name}', 
                    color=colors[model_name], linewidth=2)
            
            # Success rates
            ax2.plot(data['iterations'], data['success_rates'], 
                    marker='s', label=f'{model_name}', 
                    color=colors[model_name], linewidth=2)
        
        ax1.set_ylabel('CKA Score')
        ax1.set_title('S2 Representation Stability (Higher = More Stable)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.1)
        
        ax2.set_ylabel('S1→S3 Success Rate')
        ax2.set_xlabel('Iteration')
        ax2.set_title('Compositional Performance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.1)
        
        plt.tight_layout()
        plt.savefig('cka_similarity_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 计算相关性
        self.compute_correlations()
    
    def compute_correlations(self):
        """计算CKA与成功率的相关性"""
        with open('cka_correlation_results.txt', 'w') as f:
            f.write("CKA-Performance Correlation Analysis\n")
            f.write("="*60 + "\n\n")
            
            for model_name, data in self.results.items():
                f.write(f"{model_name}:\n")
                f.write("-"*30 + "\n")
                
                if data['cka_scores'] and len(data['success_rates']) > 1:
                    # 对齐CKA和成功率
                    cka_aligned = []
                    success_aligned = []
                    
                    for i, (iter1, iter2) in enumerate(data['iter_pairs']):
                        idx1 = data['iterations'].index(iter1)
                        idx2 = data['iterations'].index(iter2)
                        
                        if idx2 < len(data['success_rates']):
                            cka_aligned.append(data['cka_scores'][i])
                            success_aligned.append(data['success_rates'][idx2])
                    
                    if len(cka_aligned) > 1:
                        correlation = np.corrcoef(cka_aligned, success_aligned)[0, 1]
                        f.write(f"  CKA-Success Correlation: {correlation:.3f}\n")
                    
                    f.write(f"  Average Drift Rate: {data['drift_rate']:.3f}\n")
                    f.write(f"  Final Success Rate: {data['success_rates'][-1]:.3f}\n")
                
                f.write("\n")

def run_cka_analysis():
    analyzer = CKASimilarityAnalyzer()
    analyzer.analyze_drift_rate()
    analyzer.plot_results()
    print("\nAnalysis complete! Results saved to:")
    print("  - cka_similarity_analysis.png")
    print("  - cka_correlation_results.txt")

if __name__ == "__main__":
    run_cka_analysis()