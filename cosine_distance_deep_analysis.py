#!/usr/bin/env python3
# cosine_distance_deep_analysis.py

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from model import GPTConfig, GPT
from tqdm import tqdm
from datetime import datetime

class CosineDistanceAnalyzer:
    def __init__(self, device='cuda:0', output_dir=None):
        self.device = device
        self.results = {}
        self.output_dir = output_dir
        
    def load_model_and_data(self, checkpoint_path, data_dir):
        """加载模型和数据"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config = GPTConfig(**checkpoint['model_args'])
        model = GPT(config).to(self.device)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        
        with open(os.path.join(data_dir, 'stage_info.pkl'), 'rb') as f:
            stage_info = pickle.load(f)
        
        return model, stage_info['stages']
    
    def get_deep_node_embeddings(self, model, nodes):
        """获取节点的深层表示（通过整个transformer）"""
        embeddings = []
        
        with torch.no_grad():
            for node in nodes:
                # 构造输入
                node_id = node + 2  # 0是PAD，1是EOS
                x = torch.tensor([[node_id]], device=self.device)
                
                # 通过embedding层
                tok_emb = model.transformer.wte(x)
                pos = torch.zeros(1, 1, dtype=torch.long, device=self.device)
                pos_emb = model.transformer.wpe(pos)
                h = model.transformer.drop(tok_emb + pos_emb)
                
                # 通过所有transformer层
                for block in model.transformer.h:
                    h = block(h)
                h = model.transformer.ln_f(h)
                
                # 确保是1D张量 - 修复在这里！
                embeddings.append(h.squeeze())  # 移除所有大小为1的维度
        
        return torch.stack(embeddings)
    
    def compute_centroid_distances(self, model, stages):
        """计算S2质心与S1/S3质心的余弦距离"""
        S1, S2, S3 = stages
        
        # 获取各阶段的深层嵌入
        s1_embeddings = self.get_deep_node_embeddings(model, S1)
        s2_embeddings = self.get_deep_node_embeddings(model, S2)
        s3_embeddings = self.get_deep_node_embeddings(model, S3)
        
        # 计算质心
        s1_centroid = s1_embeddings.mean(dim=0)
        s2_centroid = s2_embeddings.mean(dim=0)
        s3_centroid = s3_embeddings.mean(dim=0)
        
        # 计算余弦相似度 - 修复在这里！
        cos_s2_s1 = F.cosine_similarity(s2_centroid, s1_centroid, dim=0).item()
        cos_s2_s3 = F.cosine_similarity(s2_centroid, s3_centroid, dim=0).item()
        
        # 计算每个S2节点的偏向
        s2_biases = []
        s2_bias_details = {}
        
        for idx, s2_emb in enumerate(s2_embeddings):
            cos_to_s1 = F.cosine_similarity(s2_emb, s1_centroid, dim=0).item()
            cos_to_s3 = F.cosine_similarity(s2_emb, s3_centroid, dim=0).item()
            bias = (cos_to_s3 - cos_to_s1 + 1) / 2  # 归一化到[0,1]
            s2_biases.append(bias)
            s2_bias_details[S2[idx]] = {
                'cos_to_s1': cos_to_s1,
                'cos_to_s3': cos_to_s3,
                'bias': bias
            }
        
        # 计算偏向度统计
        mean_bias = np.mean(s2_biases)
        s3_biased_count = sum(1 for b in s2_biases if b > 0.6)
        s1_biased_count = sum(1 for b in s2_biases if b < 0.4)
        neutral_count = len(s2_biases) - s3_biased_count - s1_biased_count
        
        return {
            'cos_s2_s1': cos_s2_s1,
            'cos_s2_s3': cos_s2_s3,
            'mean_bias': mean_bias,
            's3_biased_ratio': s3_biased_count / len(s2_biases),
            's1_biased_ratio': s1_biased_count / len(s2_biases),
            'neutral_ratio': neutral_count / len(s2_biases),
            'bias_std': np.std(s2_biases),
            's2_bias_details': s2_bias_details,
            'all_biases': s2_biases
        }
    
    def analyze_all_models(self):
        """分析所有模型"""
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
        
        print("Computing Deep Cosine Distances...")
        print("="*60)
        
        for model_name, config in model_configs.items():
            print(f"\nAnalyzing {model_name}...")
            self.results[model_name] = []
            
            for iteration in tqdm(iterations):
                checkpoint_path = os.path.join(config['checkpoint_dir'], f'ckpt_{iteration}.pt')
                
                if not os.path.exists(checkpoint_path):
                    print(f"  Warning: {checkpoint_path} not found")
                    continue
                
                model, stages = self.load_model_and_data(checkpoint_path, config['data_dir'])
                distances = self.compute_centroid_distances(model, stages)
                distances['iteration'] = iteration
                
                self.results[model_name].append(distances)
                
                print(f"  Iter {iteration}: bias={distances['mean_bias']:.3f}, "
                      f"S3-biased={distances['s3_biased_ratio']:.2%}")
    
    def plot_results(self):
        """绘制结果"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        colors = {'original': 'blue', '5% mixed': 'orange', '10% mixed': 'green'}
        
        for model_name, data in self.results.items():
            if not data:
                continue
                
            iterations = [d['iteration'] for d in data]
            
            # Mean bias evolution
            mean_bias = [d['mean_bias'] for d in data]
            axes[0, 0].plot(iterations, mean_bias, 
                           marker='o', label=model_name, 
                           color=colors[model_name], linewidth=2)
            
            # S3-biased ratio
            s3_ratio = [d['s3_biased_ratio'] for d in data]
            axes[0, 1].plot(iterations, s3_ratio, 
                           marker='o', label=model_name, 
                           color=colors[model_name], linewidth=2)
            
            # cos(S2, S1)
            cos_s2_s1 = [d['cos_s2_s1'] for d in data]
            axes[1, 0].plot(iterations, cos_s2_s1, 
                           marker='o', label=model_name, 
                           color=colors[model_name], linewidth=2)
            
            # cos(S2, S3)
            cos_s2_s3 = [d['cos_s2_s3'] for d in data]
            axes[1, 1].plot(iterations, cos_s2_s3, 
                           marker='o', label=model_name, 
                           color=colors[model_name], linewidth=2)
        
        axes[0, 0].set_ylabel('Mean S2 Bias')
        axes[0, 0].set_title('S2 Bias Evolution (0=S1, 1=S3)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(0.5, color='black', linestyle='--', alpha=0.5)
        
        axes[0, 1].set_ylabel('S3-biased Ratio')
        axes[0, 1].set_title('Fraction of S3-biased S2 Nodes')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_ylabel('cos(μ_S2, μ_S1)')
        axes[1, 0].set_title('S2-S1 Cosine Similarity')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_ylabel('cos(μ_S2, μ_S3)')
        axes[1, 1].set_title('S2-S3 Cosine Similarity')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'cosine_distance_deep_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 绘制偏向分布直方图
        self.plot_bias_distribution()
        
        # 保存数值结果
        self.save_numerical_results()
    
    def plot_bias_distribution(self):
        """绘制S2偏向分布"""
        iterations = [5000, 25000, 50000]
        
        fig, axes = plt.subplots(len(iterations), len(self.results), 
                                figsize=(5*len(self.results), 4*len(iterations)))
        
        if len(iterations) == 1:
            axes = axes.reshape(1, -1)
        if len(self.results) == 1:
            axes = axes.reshape(-1, 1)
        
        for iter_idx, target_iter in enumerate(iterations):
            for model_idx, (model_name, data) in enumerate(self.results.items()):
                ax = axes[iter_idx, model_idx]
                
                # 找到对应迭代的数据
                iter_data = None
                for d in data:
                    if d['iteration'] == target_iter:
                        iter_data = d
                        break
                
                if iter_data and 'all_biases' in iter_data:
                    biases = iter_data['all_biases']
                    
                    ax.hist(biases, bins=20, alpha=0.7, edgecolor='black', color='steelblue')
                    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Neutral')
                    ax.axvline(x=np.mean(biases), color='green', linestyle='-', linewidth=2, label=f'Mean: {np.mean(biases):.3f}')
                    
                    ax.set_xlabel('Bias Score (0=S1, 1=S3)')
                    ax.set_ylabel('Count')
                    ax.set_title(f'{model_name} - {target_iter} iter')
                    ax.legend()
                    ax.set_xlim(0, 1)
                else:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{model_name} - {target_iter} iter')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 's2_bias_distribution.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_numerical_results(self):
        """保存数值结果"""
        output_file = os.path.join(self.output_dir, 'cosine_distance_results.txt')
        
        with open(output_file, 'w') as f:
            f.write("Deep Cosine Distance Analysis Results\n")
            f.write("="*60 + "\n\n")
            
            for model_name, data in self.results.items():
                f.write(f"{model_name}:\n")
                f.write("-"*30 + "\n")
                
                if data:
                    initial = data[0]
                    final = data[-1]
                    
                    f.write(f"Initial ({initial['iteration']}k):\n")
                    f.write(f"  cos(S2,S1): {initial['cos_s2_s1']:.3f}\n")
                    f.write(f"  cos(S2,S3): {initial['cos_s2_s3']:.3f}\n")
                    f.write(f"  Mean Bias: {initial['mean_bias']:.3f}\n")
                    f.write(f"  S3-biased ratio: {initial['s3_biased_ratio']:.2%}\n")
                    
                    f.write(f"\nFinal ({final['iteration']}k):\n")
                    f.write(f"  cos(S2,S1): {final['cos_s2_s1']:.3f}\n")
                    f.write(f"  cos(S2,S3): {final['cos_s2_s3']:.3f}\n")
                    f.write(f"  Mean Bias: {final['mean_bias']:.3f}\n")
                    f.write(f"  S3-biased ratio: {final['s3_biased_ratio']:.2%}\n")
                    
                    f.write(f"\nBias Change: {final['mean_bias'] - initial['mean_bias']:.3f}\n")
                    f.write(f"S3-biased Change: {(final['s3_biased_ratio'] - initial['s3_biased_ratio'])*100:.1f}%\n")
                
                f.write("\n")
        
        print(f"Numerical results saved to: {output_file}")

def main():
    # 创建统一的输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'deep_mechanism_analysis_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    analyzer = CosineDistanceAnalyzer(output_dir=output_dir)
    analyzer.analyze_all_models()
    analyzer.plot_results()
    
    print("\nCosine distance analysis complete!")
    
    return output_dir

if __name__ == "__main__":
    output_dir = main()