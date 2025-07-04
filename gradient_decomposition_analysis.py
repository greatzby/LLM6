#!/usr/bin/env python3
# gradient_decomposition_analysis.py

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import os
import networkx as nx
from model import GPTConfig, GPT

class GradientDecompositionAnalyzer:
    def __init__(self, device='cuda:0'):
        self.device = device
        self.gradient_history = defaultdict(lambda: defaultdict(list))
        
    def load_model(self, checkpoint_path):
        """加载模型checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 创建模型
        config = GPTConfig(**checkpoint['model_args'])
        model = GPT(config).to(self.device)
        model.load_state_dict(checkpoint['model'])
        
        return model
    
    def analyze_gradient_contributions(self, model, data_dir, model_name, iteration):
        """分析不同路径类型对梯度的贡献"""
        # 加载数据
        with open(os.path.join(data_dir, 'stage_info.pkl'), 'rb') as f:
            stage_info = pickle.load(f)
        S1, S2, S3 = stage_info['stages']
        
        # 加载训练数据样本
        with open(os.path.join(data_dir, 'train_10.txt'), 'r') as f:
            lines = f.readlines()[:100]  # 使用前100个样本
        
        # 分类数据
        s1_s2_batch = []
        s2_s3_batch = []
        s1_s3_batch = []  # 如果有的话
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 2:
                source, target = int(parts[0]), int(parts[1])
                if source in S1 and target in S2:
                    s1_s2_batch.append(line)
                elif source in S2 and target in S3:
                    s2_s3_batch.append(line)
                elif source in S1 and target in S3:
                    s1_s3_batch.append(line)
        
        # 加载编码器
        with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
        stoi = meta['stoi']
        block_size = meta['block_size']
        
        # 计算各类型梯度
        def compute_gradient_norm(batch_lines, path_type):
            if not batch_lines:
                return 0.0
            
            model.train()
            model.zero_grad()
            
            total_loss = 0
            for line in batch_lines[:10]:  # 每类使用10个样本
                # 编码
                tokens = []
                for token in line.strip().split():
                    if token in stoi:
                        tokens.append(stoi[token])
                tokens.append(1)  # EOS
                
                # Padding
                tokens = tokens[:block_size]
                tokens.extend([0] * (block_size - len(tokens)))
                
                # 准备输入
                x = torch.tensor(tokens[:-1], device=self.device).unsqueeze(0)
                y = torch.tensor(tokens[1:], device=self.device).unsqueeze(0)
                
                # 前向传播
                logits, loss = model(x, y)
                total_loss += loss
            
            # 反向传播
            if total_loss > 0:
                total_loss.backward()
                
                # 收集S2相关参数的梯度范数
                s2_grad_norm = 0
                total_grad_norm = 0
                
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        total_grad_norm += grad_norm
                        
                        # 检查是否是S2相关的参数（这里简化处理）
                        if 'transformer.h' in name:  # 所有层都可能影响S2
                            s2_grad_norm += grad_norm
                
                return s2_grad_norm / (total_grad_norm + 1e-8)
            
            return 0.0
        
        # 计算梯度贡献
        s1_s2_grad = compute_gradient_norm(s1_s2_batch, 'S1->S2')
        s2_s3_grad = compute_gradient_norm(s2_s3_batch, 'S2->S3')
        s1_s3_grad = compute_gradient_norm(s1_s3_batch, 'S1->S3')
        
        # 计算比率
        ratio = s2_s3_grad / (s1_s2_grad + 1e-8) if s1_s2_grad > 0 else 0
        
        # 保存结果
        self.gradient_history[model_name]['iteration'].append(iteration)
        self.gradient_history[model_name]['s1_s2_grad'].append(s1_s2_grad)
        self.gradient_history[model_name]['s2_s3_grad'].append(s2_s3_grad)
        self.gradient_history[model_name]['s1_s3_grad'].append(s1_s3_grad)
        self.gradient_history[model_name]['ratio'].append(ratio)
        
        return {
            's1_s2_grad': s1_s2_grad,
            's2_s3_grad': s2_s3_grad,
            's1_s3_grad': s1_s3_grad,
            'ratio': ratio
        }
    
    def plot_results(self):
        """绘制梯度演化图"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        colors = {'original': 'blue', '5% mixed': 'orange', '10% mixed': 'green'}
        
        for model_name, data in self.gradient_history.items():
            iterations = data['iteration']
            
            # 梯度比率
            ax1.plot(iterations, data['ratio'], 
                    label=model_name, color=colors[model_name], 
                    marker='o', markersize=8, linewidth=2)
            
            # 各类型梯度贡献
            ax2.plot(iterations, data['s1_s2_grad'], 
                    label=f'{model_name} S1→S2', 
                    color=colors[model_name], linestyle='-', marker='s')
            ax2.plot(iterations, data['s2_s3_grad'], 
                    label=f'{model_name} S2→S3', 
                    color=colors[model_name], linestyle='--', marker='^')
            if any(data['s1_s3_grad']):
                ax2.plot(iterations, data['s1_s3_grad'], 
                        label=f'{model_name} S1→S3', 
                        color=colors[model_name], linestyle=':', marker='d')
        
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Gradient Ratio (S2→S3 / S1→S2)')
        ax1.set_title('Gradient Contribution Ratio Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(bottom=0)
        
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Normalized Gradient Contribution')
        ax2.set_title('Individual Path Type Gradient Contributions')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('gradient_decomposition_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """主函数"""
    analyzer = GradientDecompositionAnalyzer()
    
    # 模型配置 - 修正了10% mixed的路径
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
            'checkpoint_dir': 'out/composition_20250703_011304',  # 已修正！
            'data_dir': 'data/simple_graph/composition_90_mixed_10'
        }
    }
    
    iterations = [5000, 15000, 25000, 35000, 50000]
    
    print("Running Gradient Decomposition Analysis...")
    print("="*60)
    
    for model_name, config in model_configs.items():
        print(f"\nAnalyzing {model_name}...")
        
        for iteration in iterations:
            checkpoint_path = os.path.join(config['checkpoint_dir'], f'ckpt_{iteration}.pt')
            
            if not os.path.exists(checkpoint_path):
                print(f"  Warning: {checkpoint_path} not found, skipping...")
                continue
            
            print(f"  Iteration {iteration}...")
            
            # 加载模型
            model = analyzer.load_model(checkpoint_path)
            
            # 分析梯度
            results = analyzer.analyze_gradient_contributions(
                model, config['data_dir'], model_name, iteration
            )
            
            print(f"    S1→S2 grad: {results['s1_s2_grad']:.4f}")
            print(f"    S2→S3 grad: {results['s2_s3_grad']:.4f}")
            print(f"    Ratio: {results['ratio']:.2f}")
    
    # 绘制结果
    analyzer.plot_results()
    print("\nAnalysis complete! Results saved to gradient_decomposition_analysis.png")

if __name__ == "__main__":
    main()