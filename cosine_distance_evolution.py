# cosine_distance_evolution.py
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from model import GPTConfig, GPT
from tqdm import tqdm

class CosineDistanceAnalyzer:
    def __init__(self, device='cuda:0'):
        self.device = device
        self.results = {}
        
    def load_model_and_data(self, checkpoint_path, data_dir):
        """加载模型和数据"""
        # 加载模型
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config = GPTConfig(**checkpoint['model_args'])
        model = GPT(config).to(self.device)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        
        # 加载阶段信息
        with open(os.path.join(data_dir, 'stage_info.pkl'), 'rb') as f:
            stage_info = pickle.load(f)
        
        return model, stage_info['stages']
    
    def get_node_embeddings(self, model, nodes):
        """获取节点的嵌入表示"""
        embeddings = []
        
        with torch.no_grad():
            for node in nodes:
                # 使用词嵌入层
                node_id = node + 2  # 因为0是PAD，1是换行符
                node_tensor = torch.tensor([node_id], device=self.device)
                embedding = model.transformer.wte(node_tensor)
                embeddings.append(embedding.squeeze(0))
        
        return torch.stack(embeddings)
    
    def compute_centroid_distances(self, model, stages):
        """计算S2质心与S1/S3质心的余弦距离"""
        S1, S2, S3 = stages
        
        # 获取各阶段的嵌入
        s1_embeddings = self.get_node_embeddings(model, S1)
        s2_embeddings = self.get_node_embeddings(model, S2)
        s3_embeddings = self.get_node_embeddings(model, S3)
        
        # 计算质心
        s1_centroid = s1_embeddings.mean(dim=0)
        s2_centroid = s2_embeddings.mean(dim=0)
        s3_centroid = s3_embeddings.mean(dim=0)
        
        # 计算余弦相似度
        cos_s2_s1 = F.cosine_similarity(s2_centroid.unsqueeze(0), 
                                       s1_centroid.unsqueeze(0)).item()
        cos_s2_s3 = F.cosine_similarity(s2_centroid.unsqueeze(0), 
                                       s3_centroid.unsqueeze(0)).item()
        
        # 计算偏向度（-1到1，正值表示偏向S3）
        bias = cos_s2_s3 - cos_s2_s1
        
        # 归一化偏向度到[0,1]
        normalized_bias = (bias + 2) / 4  # 从[-2,2]映射到[0,1]
        
        return {
            'cos_s2_s1': cos_s2_s1,
            'cos_s2_s3': cos_s2_s3,
            'bias': bias,
            'normalized_bias': normalized_bias
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
        
        print("Computing Cosine Distances...")
        print("="*60)
        
        for model_name, config in model_configs.items():
            print(f"\nAnalyzing {model_name}...")
            self.results[model_name] = []
            
            for iteration in tqdm(iterations):
                checkpoint_path = os.path.join(config['checkpoint_dir'], f'ckpt_{iteration}.pt')
                
                if not os.path.exists(checkpoint_path):
                    print(f"  Warning: {checkpoint_path} not found")
                    continue
                
                # 加载模型和数据
                model, stages = self.load_model_and_data(checkpoint_path, config['data_dir'])
                
                # 计算距离
                distances = self.compute_centroid_distances(model, stages)
                distances['iteration'] = iteration
                
                self.results[model_name].append(distances)
                
                print(f"  Iter {iteration}: bias={distances['normalized_bias']:.3f}")
    
    def plot_results(self):
        """绘制结果"""
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        
        colors = {'original': 'blue', '5% mixed': 'orange', '10% mixed': 'green'}
        
        for model_name, data in self.results.items():
            if not data:
                continue
                
            iterations = [d['iteration'] for d in data]
            
            # cos(S2, S1)
            cos_s2_s1 = [d['cos_s2_s1'] for d in data]
            axes[0].plot(iterations, cos_s2_s1, 
                        marker='o', label=model_name, 
                        color=colors[model_name], linewidth=2)
            
            # cos(S2, S3)
            cos_s2_s3 = [d['cos_s2_s3'] for d in data]
            axes[1].plot(iterations, cos_s2_s3, 
                        marker='o', label=model_name, 
                        color=colors[model_name], linewidth=2)
            
            # Normalized Bias
            bias = [d['normalized_bias'] for d in data]
            axes[2].plot(iterations, bias, 
                        marker='o', label=model_name, 
                        color=colors[model_name], linewidth=2)
        
        axes[0].set_ylabel('cos(μ_S2, μ_S1)')
        axes[0].set_title('S2-S1 Cosine Similarity Evolution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_ylabel('cos(μ_S2, μ_S3)')
        axes[1].set_title('S2-S3 Cosine Similarity Evolution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        axes[2].set_ylabel('S2 Bias (0=S1, 1=S3)')
        axes[2].set_title('S2 Position Bias Evolution')
        axes[2].set_xlabel('Iteration')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].axhline(0.5, color='black', linestyle='--', alpha=0.5, label='Neutral')
        
        plt.tight_layout()
        plt.savefig('cosine_distance_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存数值结果
        self.save_numerical_results()
    
    def save_numerical_results(self):
        """保存数值结果"""
        with open('cosine_distance_results.txt', 'w') as f:
            f.write("Cosine Distance Analysis Results\n")
            f.write("="*60 + "\n\n")
            
            for model_name, data in self.results.items():
                f.write(f"{model_name}:\n")
                f.write("-"*30 + "\n")
                
                if data:
                    initial = data[0]
                    final = data[-1]
                    
                    f.write(f"Initial (5k):\n")
                    f.write(f"  cos(S2,S1): {initial['cos_s2_s1']:.3f}\n")
                    f.write(f"  cos(S2,S3): {initial['cos_s2_s3']:.3f}\n")
                    f.write(f"  Bias: {initial['normalized_bias']:.3f}\n")
                    
                    f.write(f"Final (50k):\n")
                    f.write(f"  cos(S2,S1): {final['cos_s2_s1']:.3f}\n")
                    f.write(f"  cos(S2,S3): {final['cos_s2_s3']:.3f}\n")
                    f.write(f"  Bias: {final['normalized_bias']:.3f}\n")
                    
                    f.write(f"Bias Change: {final['normalized_bias'] - initial['normalized_bias']:.3f}\n")
                
                f.write("\n")

def run_cosine_analysis():
    analyzer = CosineDistanceAnalyzer()
    analyzer.analyze_all_models()
    analyzer.plot_results()
    print("\nAnalysis complete! Results saved to:")
    print("  - cosine_distance_evolution.png")
    print("  - cosine_distance_results.txt")

if __name__ == "__main__":
    run_cosine_analysis()