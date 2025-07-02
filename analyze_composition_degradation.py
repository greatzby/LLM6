# analyze_composition_degradation.py
import os
import torch
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from datetime import datetime
import pandas as pd

from model import GPTConfig, GPT

class CompositionAnalyzer:
    def __init__(self, checkpoint_dir, data_dir):
        self.checkpoint_dir = checkpoint_dir
        self.data_dir = data_dir
        
        # 加载必要的数据
        self._load_metadata()
        self._load_graph()
        self._load_test_data()
        
    def _load_metadata(self):
        """加载元数据"""
        with open(os.path.join(self.data_dir, 'meta.pkl'), 'rb') as f:
            self.meta = pickle.load(f)
        self.stoi = self.meta['stoi']
        self.itos = self.meta['itos']
        self.vocab_size = self.meta['vocab_size']
        self.block_size = self.meta['block_size']
        
        with open(os.path.join(self.data_dir, 'stage_info.pkl'), 'rb') as f:
            stage_info = pickle.load(f)
        self.stages = stage_info['stages']
        self.S1, self.S2, self.S3 = self.stages
        
    def _load_graph(self):
        """加载图结构"""
        self.G = nx.read_graphml(os.path.join(self.data_dir, 'composition_graph.graphml'))
        
    def _load_test_data(self):
        """加载测试数据"""
        with open(os.path.join(self.data_dir, 'test.txt'), 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        self.test_data = {
            'S1->S2': [],
            'S2->S3': [],
            'S1->S3': []
        }
        
        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                source, target = int(parts[0]), int(parts[1])
                path = [int(p) for p in parts[2:]]
                
                if source in self.S1 and target in self.S2:
                    self.test_data['S1->S2'].append((source, target, path))
                elif source in self.S2 and target in self.S3:
                    self.test_data['S2->S3'].append((source, target, path))
                elif source in self.S1 and target in self.S3:
                    self.test_data['S1->S3'].append((source, target, path))
    
    def load_checkpoint(self, iteration):
        """加载特定的checkpoint"""
        checkpoint_path = os.path.join(self.checkpoint_dir, f'ckpt_{iteration}.pt')
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 创建模型
        model_args = checkpoint['model_args']
        config = GPTConfig(**model_args)
        model = GPT(config).cuda()
        model.load_state_dict(checkpoint['model'])
        model.eval()
        
        return model, checkpoint
    
    @torch.no_grad()
    def evaluate_model(self, model, max_samples=50):
        """评估模型性能"""
        results = {}
        
        for path_type, test_cases in self.test_data.items():
            correct = 0
            total = min(len(test_cases), max_samples)
            failures = []
            successes = []
            
            for idx in range(total):
                source, target, true_path = test_cases[idx]
                
                # Token级编码
                prompt = f"{source} {target} {source}"
                prompt_tokens = prompt.split()
                prompt_ids = [self.stoi[t] for t in prompt_tokens if t in self.stoi]
                
                x = torch.tensor(prompt_ids, dtype=torch.long).cuda().unsqueeze(0)
                
                # 生成
                y = model.generate(x, max_new_tokens=30, temperature=0.1, top_k=10)
                
                # 解码
                all_numbers = []
                for tid in y[0].tolist():
                    if tid == 1:  # EOS
                        break
                    if tid in self.itos:
                        try:
                            all_numbers.append(int(self.itos[tid]))
                        except:
                            pass
                
                # 提取路径
                generated_path = all_numbers[2:] if len(all_numbers) >= 3 else []
                
                # 验证
                success = self._validate_path(generated_path, source, target, path_type)
                
                if success:
                    correct += 1
                    if len(successes) < 5:  # 保存前5个成功案例
                        successes.append({
                            'source': source,
                            'target': target,
                            'generated': generated_path,
                            'true': true_path
                        })
                else:
                    if len(failures) < 10:  # 保存前10个失败案例
                        failures.append({
                            'source': source,
                            'target': target,
                            'generated': generated_path,
                            'true': true_path,
                            'error': self._analyze_failure(generated_path, source, target, path_type)
                        })
            
            results[path_type] = {
                'accuracy': correct / total if total > 0 else 0,
                'correct': correct,
                'total': total,
                'successes': successes,
                'failures': failures
            }
        
        return results
    
    def _validate_path(self, path, source, target, path_type):
        """验证路径是否有效"""
        if len(path) < 2:
            return False
        
        if path[0] != source or path[-1] != target:
            return False
        
        # 检查路径有效性
        for i in range(len(path) - 1):
            if not self.G.has_edge(str(path[i]), str(path[i+1])):
                return False
        
        # S1->S3需要经过S2
        if path_type == 'S1->S3':
            has_s2 = any(node in self.S2 for node in path[1:-1])
            if not has_s2:
                return False
        
        return True
    
    def _analyze_failure(self, path, source, target, path_type):
        """分析失败原因"""
        if len(path) < 2:
            return 'too_short'
        
        if path[0] != source:
            return 'wrong_source'
        
        if path[-1] != target:
            return 'wrong_target'
        
        # 检查无效边
        for i in range(len(path) - 1):
            if not self.G.has_edge(str(path[i]), str(path[i+1])):
                return f'invalid_edge_{path[i]}->{path[i+1]}'
        
        # S1->S3特殊检查
        if path_type == 'S1->S3':
            has_s2 = any(node in self.S2 for node in path[1:-1])
            if not has_s2:
                return 'no_s2_intermediate'
        
        return 'unknown'
    
    @torch.no_grad()
    def extract_attention_patterns(self, model, test_cases, num_samples=5):
        """提取注意力模式"""
        attention_patterns = []
        
        for idx in range(min(num_samples, len(test_cases))):
            source, target, _ = test_cases[idx]
            
            # 准备输入
            prompt = f"{source} {target} {source}"
            prompt_tokens = prompt.split()
            prompt_ids = [self.stoi[t] for t in prompt_tokens if t in self.stoi]
            
            x = torch.tensor(prompt_ids, dtype=torch.long).cuda().unsqueeze(0)
            
            # 前向传播获取注意力
            model.eval()
            
            # Hook来提取注意力
            attentions = []
            
            def hook_fn(module, input, output):
                # 假设输出是 (attention_weights, values) 或类似结构
                if hasattr(module, 'attn'):
                    attentions.append(output.detach().cpu())
            
            # 注册hook
            hooks = []
            for block in model.transformer.h:
                hook = block.attn.register_forward_hook(hook_fn)
                hooks.append(hook)
            
            # 前向传播
            _ = model(x)
            
            # 移除hooks
            for hook in hooks:
                hook.remove()
            
            if attentions:
                attention_patterns.append({
                    'source': source,
                    'target': target,
                    'prompt': prompt,
                    'attention': attentions
                })
        
        return attention_patterns
    
    def analyze_all_checkpoints(self):
        """分析所有checkpoint"""
        iterations = [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000]
        
        all_results = {}
        
        for iteration in iterations:
            print(f"\nAnalyzing checkpoint at iteration {iteration}...")
            
            try:
                model, checkpoint = self.load_checkpoint(iteration)
                results = self.evaluate_model(model)
                
                all_results[iteration] = results
                
                # 打印结果
                print(f"Results at iteration {iteration}:")
                for path_type in ['S1->S2', 'S2->S3', 'S1->S3']:
                    acc = results[path_type]['accuracy']
                    print(f"  {path_type}: {acc:.2%}")
                
                # 分析S1->S3的失败模式
                if results['S1->S3']['failures']:
                    print(f"\n  S1->S3 failure analysis:")
                    error_counts = defaultdict(int)
                    for failure in results['S1->S3']['failures']:
                        error_type = failure['error']
                        error_counts[error_type] += 1
                    
                    for error_type, count in error_counts.items():
                        print(f"    {error_type}: {count}")
                
                # 清理GPU内存
                del model
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error loading checkpoint {iteration}: {e}")
        
        return all_results
    
    def plot_results(self, all_results):
        """绘制分析结果"""
        # 准备数据
        iterations = sorted(all_results.keys())
        accuracies = {
            'S1->S2': [],
            'S2->S3': [],
            'S1->S3': []
        }
        
        for iter in iterations:
            for path_type in accuracies:
                accuracies[path_type].append(all_results[iter][path_type]['accuracy'])
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 准确率曲线
        ax1 = axes[0, 0]
        ax1.plot(iterations, accuracies['S1->S2'], 'b-o', label='S1->S2', markersize=8)
        ax1.plot(iterations, accuracies['S2->S3'], 'g-s', label='S2->S3', markersize=8)
        ax1.plot(iterations, accuracies['S1->S3'], 'r-^', label='S1->S3', linewidth=3, markersize=10)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy Over Training')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.05)
        
        # 标记最高点
        max_s1s3_idx = np.argmax(accuracies['S1->S3'])
        max_s1s3_iter = iterations[max_s1s3_idx]
        max_s1s3_acc = accuracies['S1->S3'][max_s1s3_idx]
        ax1.annotate(f'Peak: {max_s1s3_acc:.2%}\n@ iter {max_s1s3_iter}',
                    xy=(max_s1s3_iter, max_s1s3_acc),
                    xytext=(max_s1s3_iter + 5000, max_s1s3_acc - 0.1),
                    arrowprops=dict(arrowstyle='->', color='red'))
        
        # 2. 组合能力差距
        ax2 = axes[0, 1]
        basic_avg = [(s12 + s23) / 2 for s12, s23 in zip(accuracies['S1->S2'], accuracies['S2->S3'])]
        comp_gap = [b - s13 for b, s13 in zip(basic_avg, accuracies['S1->S3'])]
        
        ax2.plot(iterations, comp_gap, 'purple', linewidth=2, marker='o', markersize=8)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Composition Gap')
        ax2.set_title('Basic Performance - S1->S3 Performance')
        ax2.grid(True, alpha=0.3)
        
        # 3. 错误类型分布（S1->S3）
        ax3 = axes[1, 0]
        error_data = defaultdict(list)
        
        for iter in iterations:
            error_counts = defaultdict(int)
            failures = all_results[iter]['S1->S3']['failures']
            
            for failure in failures:
                error_type = failure['error'].split('_')[0]  # 简化错误类型
                error_counts[error_type] += 1
            
            # 归一化
            total_failures = sum(error_counts.values())
            for error_type in ['too', 'wrong', 'invalid', 'no', 'unknown']:
                if total_failures > 0:
                    error_data[error_type].append(error_counts.get(error_type, 0) / total_failures)
                else:
                    error_data[error_type].append(0)
        
        # 堆叠条形图
        bottom = np.zeros(len(iterations))
        colors = ['red', 'orange', 'yellow', 'green', 'blue']
        
        for i, (error_type, values) in enumerate(error_data.items()):
            ax3.bar(range(len(iterations)), values, bottom=bottom, 
                   label=error_type, color=colors[i % len(colors)], alpha=0.7)
            bottom += np.array(values)
        
        ax3.set_xticks(range(len(iterations)))
        ax3.set_xticklabels([f'{it//1000}k' for it in iterations])
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Error Type Distribution')
        ax3.set_title('S1->S3 Failure Modes')
        ax3.legend()
        
        # 4. 性能热图
        ax4 = axes[1, 1]
        heatmap_data = []
        for iter in iterations:
            row = []
            for path_type in ['S1->S2', 'S2->S3', 'S1->S3']:
                row.append(all_results[iter][path_type]['accuracy'])
            heatmap_data.append(row)
        
        im = ax4.imshow(np.array(heatmap_data).T, aspect='auto', cmap='RdYlGn')
        ax4.set_yticks([0, 1, 2])
        ax4.set_yticklabels(['S1->S2', 'S2->S3', 'S1->S3'])
        ax4.set_xticks(range(len(iterations)))
        ax4.set_xticklabels([f'{it//1000}k' for it in iterations])
        ax4.set_xlabel('Iteration')
        ax4.set_title('Performance Heatmap')
        
        # 添加数值标注
        for i in range(len(iterations)):
            for j in range(3):
                text = ax4.text(i, j, f'{heatmap_data[i][j]:.2f}',
                               ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im, ax=ax4)
        
        plt.tight_layout()
        plt.savefig('composition_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 额外的图：S1->S3的详细分析
        self.plot_s1s3_detailed(all_results, iterations)
    
    def plot_s1s3_detailed(self, all_results, iterations):
        """S1->S3的详细分析"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 成功案例的路径长度分布
        ax1 = axes[0, 0]
        for i, iter in enumerate(iterations[::2]):  # 每隔一个
            successes = all_results[iter]['S1->S3']['successes']
            if successes:
                path_lengths = [len(s['generated']) for s in successes]
                ax1.hist(path_lengths, alpha=0.5, label=f'Iter {iter//1000}k', bins=range(2, 10))
        
        ax1.set_xlabel('Path Length')
        ax1.set_ylabel('Count')
        ax1.set_title('S1->S3 Path Length Distribution (Successes)')
        ax1.legend()
        
        # 2. 失败案例分析
        ax2 = axes[0, 1]
        iter_labels = []
        failure_rates = []
        
        for iter in iterations:
            total = all_results[iter]['S1->S3']['total']
            failures = len(all_results[iter]['S1->S3']['failures'])
            iter_labels.append(f'{iter//1000}k')
            failure_rates.append(failures / total if total > 0 else 0)
        
        ax2.bar(iter_labels, failure_rates, color='red', alpha=0.7)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Failure Rate')
        ax2.set_title('S1->S3 Failure Rate')
        ax2.set_ylim(0, 1)
        
        # 3. 路径示例
        ax3 = axes[1, 0]
        ax3.axis('off')
        
        # 显示早期（成功）和后期（失败）的例子
        early_iter = iterations[1]  # 10000
        late_iter = iterations[-1]  # 50000
        
        text = "Path Examples:\n\n"
        text += f"Early (Iter {early_iter}, High Accuracy):\n"
        
        for ex in all_results[early_iter]['S1->S3']['successes'][:2]:
            text += f"  {ex['source']}→{ex['target']}: {ex['generated']}\n"
        
        text += f"\nLate (Iter {late_iter}, Low Accuracy):\n"
        
        for ex in all_results[late_iter]['S1->S3']['failures'][:2]:
            text += f"  {ex['source']}→{ex['target']}: {ex['generated']} (Error: {ex['error']})\n"
        
        ax3.text(0.05, 0.95, text, transform=ax3.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        # 4. 组合能力的稳定性
        ax4 = axes[1, 1]
        
        # 计算滑动窗口标准差
        s1s3_acc = [all_results[iter]['S1->S3']['accuracy'] for iter in iterations]
        
        # 简单的变化率
        changes = [abs(s1s3_acc[i] - s1s3_acc[i-1]) for i in range(1, len(s1s3_acc))]
        
        ax4.plot(iterations[1:], changes, 'o-', color='darkred', linewidth=2)
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Absolute Change from Previous')
        ax4.set_title('S1->S3 Performance Volatility')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('s1s3_detailed_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_detailed_report(self, all_results):
        """保存详细报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f'composition_analysis_report_{timestamp}.txt'
        
        with open(report_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("COMPOSITION ABILITY DEGRADATION ANALYSIS\n")
            f.write("="*60 + "\n\n")
            
            # 总结
            f.write("SUMMARY:\n")
            f.write("-"*40 + "\n")
            
            s1s3_accs = {iter: res['S1->S3']['accuracy'] for iter, res in all_results.items()}
            best_iter = max(s1s3_accs, key=s1s3_accs.get)
            worst_iter = min(s1s3_accs, key=s1s3_accs.get)
            
            f.write(f"Best S1->S3 Performance: {s1s3_accs[best_iter]:.2%} at iteration {best_iter}\n")
            f.write(f"Worst S1->S3 Performance: {s1s3_accs[worst_iter]:.2%} at iteration {worst_iter}\n")
            f.write(f"Performance Drop: {s1s3_accs[best_iter] - s1s3_accs[worst_iter]:.2%}\n\n")
            
            # 详细结果
            f.write("DETAILED RESULTS:\n")
            f.write("-"*40 + "\n")
            
            for iter in sorted(all_results.keys()):
                f.write(f"\nIteration {iter}:\n")
                
                for path_type in ['S1->S2', 'S2->S3', 'S1->S3']:
                    res = all_results[iter][path_type]
                    f.write(f"  {path_type}: {res['accuracy']:.2%} ({res['correct']}/{res['total']})\n")
                
                # S1->S3错误分析
                if all_results[iter]['S1->S3']['failures']:
                    f.write("\n  S1->S3 Failure Analysis:\n")
                    error_counts = defaultdict(int)
                    
                    for failure in all_results[iter]['S1->S3']['failures']:
                        error_counts[failure['error']] += 1
                    
                    for error, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
                        f.write(f"    {error}: {count}\n")
                    
                    # 失败示例
                    f.write("\n  Example Failures:\n")
                    for i, failure in enumerate(all_results[iter]['S1->S3']['failures'][:3]):
                        f.write(f"    {i+1}. {failure['source']}→{failure['target']}: ")
                        f.write(f"Generated: {failure['generated']}, ")
                        f.write(f"True: {failure['true'][:5]}..., ")
                        f.write(f"Error: {failure['error']}\n")
        
        print(f"\nDetailed report saved to: {report_file}")

def main():
    """主函数"""
    checkpoint_dir = "out/composition_20250703_004537"
    data_dir = "data/simple_graph/composition_90_mixed_5"
    
    print("Starting Composition Degradation Analysis...")
    print("="*60)
    
    # 创建分析器
    analyzer = CompositionAnalyzer(checkpoint_dir, data_dir)
    
    # 分析所有checkpoint
    all_results = analyzer.analyze_all_checkpoints()
    
    # 绘制结果
    print("\nGenerating visualizations...")
    analyzer.plot_results(all_results)
    
    # 保存详细报告
    analyzer.save_detailed_report(all_results)
    
    print("\nAnalysis complete!")
    print("Outputs:")
    print("  - composition_analysis.png")
    print("  - s1s3_detailed_analysis.png")
    print("  - composition_analysis_report_*.txt")

if __name__ == "__main__":
    main()