#!/usr/bin/env python3
# performance_verification.py

import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from model import GPTConfig, GPT
from sklearn.metrics.pairwise import linear_kernel
from datetime import datetime
from tqdm import tqdm

class PerformanceVerificationAnalyzer:
    def __init__(self, device='cuda:0', output_dir=None):
        self.device = device
        self.results = {}
        self.output_dir = output_dir
    
    def test_model_performance(self, checkpoint_path, data_dir):
        """直接测试模型性能"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config = GPTConfig(**checkpoint['model_args'])
        model = GPT(config).to(self.device)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        
        # 加载数据
        with open(os.path.join(data_dir, 'stage_info.pkl'), 'rb') as f:
            stage_info = pickle.load(f)
        S1, S2, S3 = stage_info['stages']
        
        with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
        
        # 测试各种路径类型
        test_file = os.path.join(data_dir, 'test.txt')
        results = {'S1->S2': {'success': 0, 'total': 0},
                  'S2->S3': {'success': 0, 'total': 0},
                  'S1->S3': {'success': 0, 'total': 0}}
        
        # 收集路径示例
        path_examples = {'S1->S3': {'successful': [], 'failed': []}}
        
        with open(test_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    source, target = int(parts[0]), int(parts[1])
                    
                    # 确定路径类型
                    if source in S1 and target in S2:
                        path_type = 'S1->S2'
                    elif source in S2 and target in S3:
                        path_type = 'S2->S3'
                    elif source in S1 and target in S3:
                        path_type = 'S1->S3'
                    else:
                        continue
                    
                    results[path_type]['total'] += 1
                    
                    # 生成路径
                    prompt = f"{source} {target} {source}"
                    prompt_ids = [meta['stoi'][t] for t in prompt.split()]
                    x = torch.tensor(prompt_ids, device=self.device).unsqueeze(0)
                    
                    with torch.no_grad():
                        y = model.generate(x, max_new_tokens=30, temperature=0.1, top_k=10)
                    
                    # 解码验证
                    generated = []
                    for tid in y[0].tolist():
                        if tid == 1:  # EOS
                            break
                        if tid in meta['itos']:
                            try:
                                generated.append(int(meta['itos'][tid]))
                            except:
                                pass
                    
                    path = generated[2:] if len(generated) >= 3 else []
                    
                    # 验证路径
                    if len(path) >= 2 and path[0] == source and path[-1] == target:
                        results[path_type]['success'] += 1
                        
                        # 记录S1->S3的成功例子
                        if path_type == 'S1->S3' and len(path_examples['S1->S3']['successful']) < 5:
                            path_examples['S1->S3']['successful'].append({
                                'source': source,
                                'target': target,
                                'path': path
                            })
                    else:
                        # 记录S1->S3的失败例子
                        if path_type == 'S1->S3' and len(path_examples['S1->S3']['failed']) < 5:
                            path_examples['S1->S3']['failed'].append({
                                'source': source,
                                'target': target,
                                'generated': path
                            })
                    
                    # 限制每种类型的测试数量
                    if results[path_type]['total'] >= 50:
                        results[path_type]['total'] = 50
                        continue
        
        # 计算成功率
        performance = {}
        for path_type, stats in results.items():
            if stats['total'] > 0:
                performance[path_type] = stats['success'] / stats['total']
            else:
                performance[path_type] = 0.0
        
        return performance, path_examples
    
    def compute_simple_cka(self, representations1, representations2):
        """计算简单的CKA相似度"""
        # 展平表示
        X = representations1.reshape(len(representations1), -1)
        Y = representations2.reshape(len(representations2), -1)
        
        # 计算核矩阵
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
        """获取S2表示用于CKA计算"""
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
                node_id = node + 2
                x = torch.tensor([[node_id]], device=self.device)
                
                # 通过模型获取表示
                tok_emb = model.transformer.wte(x)
                pos = torch.zeros(1, 1, dtype=torch.long, device=self.device)
                pos_emb = model.transformer.wpe(pos)
                h = model.transformer.drop(tok_emb + pos_emb)
                
                for block in model.transformer.h:
                    h = block(h)
                h = model.transformer.ln_f(h)
                
                representations.append(h.squeeze().cpu().numpy())
        
        return np.array(representations)
    
    def run_comprehensive_verification(self):
        """运行综合验证分析"""
        model_configs = {
            'original': {
                'checkpoint_dir': 'out/composition_20250702_063926',
                'data_dir': 'data/simple_graph/composition_90'
            },
            '5% mixed': {
                'checkpoint_dir': 'out/composition_20250703_004537',
                'data_dir': 'data/simple_graph/composition_90_mixed_5'
            },
            '10% mixed': {
                'checkpoint_dir': 'out/composition_20250703_011304',
                'data_dir': 'data/simple_graph/composition_90_mixed_10'
            }
        }
        
        iterations = [5000, 15000, 25000, 35000, 50000]
        
        print("\nRunning Performance Verification...")
        print("="*60)
        
        for model_name, config in model_configs.items():
            print(f"\n{model_name}:")
            self.results[model_name] = {
                'performance': [],
                'representations': {},
                'path_examples': {}
            }
            
            for iteration in tqdm(iterations):
                checkpoint_path = os.path.join(config['checkpoint_dir'], f'ckpt_{iteration}.pt')
                
                if os.path.exists(checkpoint_path):
                    # 测试性能
                    perf, examples = self.test_model_performance(checkpoint_path, config['data_dir'])
                    
                    # 获取S2表示
                    s2_reps = self.get_s2_representations(checkpoint_path, config['data_dir'])
                    
                    self.results[model_name]['performance'].append({
                        'iteration': iteration,
                        'S1->S2': perf['S1->S2'],
                        'S2->S3': perf['S2->S3'],
                        'S1->S3': perf['S1->S3']
                    })
                    
                    self.results[model_name]['representations'][iteration] = s2_reps
                    self.results[model_name]['path_examples'][iteration] = examples
                    
                    print(f"  Iter {iteration}: S1→S3 = {perf['S1->S3']:.2%}")
        
        # 计算CKA稳定性
        self.compute_representation_stability()
    
    def compute_representation_stability(self):
        """计算表示稳定性（CKA）"""
        print("\nComputing representation stability...")
        
        for model_name, data in self.results.items():
            if 'representations' not in data:
                continue
                
            iterations = sorted(data['representations'].keys())
            cka_scores = []
            
            for i in range(len(iterations) - 1):
                iter1, iter2 = iterations[i], iterations[i+1]
                reps1 = data['representations'][iter1]
                reps2 = data['representations'][iter2]
                
                cka = self.compute_simple_cka(reps1, reps2)
                cka_scores.append({
                    'iter_pair': (iter1, iter2),
                    'cka': cka
                })
            
            data['cka_scores'] = cka_scores
            
            if cka_scores:
                avg_stability = np.mean([s['cka'] for s in cka_scores])
                print(f"  {model_name}: Average stability = {avg_stability:.3f}")
    
    def plot_results(self):
        """绘制结果"""
        # 创建性能对比图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        
        colors = {'original': 'blue', '5% mixed': 'orange', '10% mixed': 'green'}
        
        for model_name, data in self.results.items():
            if 'performance' not in data or not data['performance']:
                continue
                
            iterations = [d['iteration'] for d in data['performance']]
            s1_s3_perf = [d['S1->S3'] for d in data['performance']]
            
            # 性能曲线
            ax1.plot(iterations, s1_s3_perf, 
                    'o-', label=model_name, 
                    color=colors[model_name], 
                    linewidth=3, markersize=10)
            
            # CKA稳定性
            if 'cka_scores' in data:
                cka_iters = [(p['iter_pair'][0] + p['iter_pair'][1]) / 2 
                            for p in data['cka_scores']]
                cka_values = [p['cka'] for p in data['cka_scores']]
                
                ax2.plot(cka_iters, cka_values,
                        'o-', label=model_name,
                        color=colors[model_name],
                        linewidth=2, markersize=8)
        
        ax1.set_ylabel('S1→S3 Success Rate')
        ax1.set_title('Compositional Performance Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.05)
        
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('CKA Score')
        ax2.set_title('S2 Representation Stability')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.05)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'performance_verification.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 创建详细性能表格图
        self.plot_performance_table()
        
        # 保存结果
        self.save_results()
    
    def plot_performance_table(self):
        """创建性能表格可视化"""
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # 准备表格数据
        headers = ['Model', 'Iteration', 'S1→S2', 'S2→S3', 'S1→S3']
        table_data = []
        
        for model_name in ['original', '5% mixed', '10% mixed']:
            if model_name not in self.results:
                continue
                
            data = self.results[model_name]
            if 'performance' in data:
                # 只显示关键迭代
                key_iters = [5000, 25000, 50000]
                for perf in data['performance']:
                    if perf['iteration'] in key_iters:
                        table_data.append([
                            model_name,
                            f"{perf['iteration']}",
                            f"{perf['S1->S2']:.1%}",
                            f"{perf['S2->S3']:.1%}",
                            f"{perf['S1->S3']:.1%}"
                        ])
        
        # 创建表格
        table = ax.table(cellText=table_data,
                        colLabels=headers,
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.2, 0.15, 0.15, 0.15, 0.15])
        
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.5)
        
        # 设置颜色
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # 为S1->S3性能低的单元格着色
        for i in range(1, len(table_data) + 1):
            s1_s3_value = float(table_data[i-1][4].strip('%')) / 100
            if s1_s3_value < 0.5:
                table[(i, 4)].set_facecolor('#ffcccc')
            elif s1_s3_value > 0.8:
                table[(i, 4)].set_facecolor('#ccffcc')
        
        plt.title('Performance Summary Table', fontsize=16, pad=20)
        plt.savefig(os.path.join(self.output_dir, 'performance_table.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self):
        """保存所有结果"""
        # 保存原始数据
        with open(os.path.join(self.output_dir, 'performance_results.pkl'), 'wb') as f:
            pickle.dump(self.results, f)
        
        # 保存文本报告
        output_file = os.path.join(self.output_dir, 'performance_verification_report.txt')
        
        with open(output_file, 'w') as f:
            f.write("Performance Verification Report\n")
            f.write("="*60 + "\n\n")
            
            for model_name, data in self.results.items():
                f.write(f"{model_name}:\n")
                f.write("-"*30 + "\n")
                
                if 'performance' in data and data['performance']:
                    # 初始和最终性能
                    initial = data['performance'][0]
                    final = data['performance'][-1]
                    
                    f.write(f"\nInitial ({initial['iteration']}k):\n")
                    f.write(f"  S1→S2: {initial['S1->S2']:.1%}\n")
                    f.write(f"  S2→S3: {initial['S2->S3']:.1%}\n")
                    f.write(f"  S1→S3: {initial['S1->S3']:.1%}\n")
                    
                    f.write(f"\nFinal ({final['iteration']}k):\n")
                    f.write(f"  S1→S2: {final['S1->S2']:.1%}\n")
                    f.write(f"  S2→S3: {final['S2->S3']:.1%}\n")
                    f.write(f"  S1→S3: {final['S1->S3']:.1%}\n")
                    
                    f.write(f"\nS1→S3 Performance Change: ")
                    change = (final['S1->S3'] - initial['S1->S3']) * 100
                    f.write(f"{change:+.1f}%\n")
                    
                    # CKA稳定性
                    if 'cka_scores' in data and data['cka_scores']:
                        avg_cka = np.mean([s['cka'] for s in data['cka_scores']])
                        f.write(f"\nAverage CKA Stability: {avg_cka:.3f}\n")
                    
                    # 路径示例
                    if 50000 in data['path_examples']:
                        examples = data['path_examples'][50000]['S1->S3']
                        f.write(f"\nExample S1→S3 paths at 50k:\n")
                        
                        if examples['successful']:
                            f.write("  Successful:\n")
                            for ex in examples['successful'][:3]:
                                f.write(f"    {ex['source']}→{ex['target']}: {ex['path']}\n")
                        
                        if examples['failed']:
                            f.write("  Failed:\n")
                            for ex in examples['failed'][:3]:
                                f.write(f"    {ex['source']}→{ex['target']}: {ex['generated']}\n")
                
                f.write("\n")
        
        print(f"Report saved to: {output_file}")

def main(output_dir=None):
    if output_dir is None:
        # 如果没有提供输出目录，创建一个新的
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f'deep_mechanism_analysis_{timestamp}'
        os.makedirs(output_dir, exist_ok=True)
    
    analyzer = PerformanceVerificationAnalyzer(output_dir=output_dir)
    analyzer.run_comprehensive_verification()
    analyzer.plot_results()
    
    print("\nPerformance verification complete!")
    
    return output_dir

if __name__ == "__main__":
    main()