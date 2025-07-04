#!/usr/bin/env python3
# gradient_response_analysis.py

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
from datetime import datetime

class GradientResponseAnalyzer:
    def __init__(self, device='cuda:0', output_dir=None):
        self.device = device
        self.gradient_history = defaultdict(lambda: defaultdict(list))
        self.output_dir = output_dir
        
    def load_model(self, checkpoint_path):
        """加载模型checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config = GPTConfig(**checkpoint['model_args'])
        model = GPT(config).to(self.device)
        model.load_state_dict(checkpoint['model'])
        return model
    
    def analyze_gradient_attribution(self, model, data_dir, model_name, iteration):
        """分析S2对不同阶段的响应（作为梯度的代理）"""
        model.eval()
        
        # 加载数据
        with open(os.path.join(data_dir, 'stage_info.pkl'), 'rb') as f:
            stage_info = pickle.load(f)
        S1, S2, S3 = stage_info['stages']
        
        with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
        stoi = meta['stoi']
        
        # 分析所有S2节点的平均响应
        all_s1_responses = []
        all_s3_responses = []
        s2_individual_biases = {}
        
        with torch.no_grad():
            for s2 in S2[:10]:  # 采样10个S2节点
                s2_id = stoi[str(s2)]
                
                # S2对S1的响应
                s1_responses = []
                for s1 in S1[:5]:  # 每个S2测试5个S1
                    # 构造S2->S1的序列
                    s1_id = stoi[str(s1)]
                    x = torch.tensor([[s2_id, s1_id]], device=self.device)
                    logits, _ = model(x)
                    
                    # 看模型预测S1的概率
                    prob = torch.softmax(logits[0, 0], dim=-1)[s1_id].item()
                    s1_responses.append(prob)
                
                # S2对S3的响应
                s3_responses = []
                for s3 in S3[:5]:  # 每个S2测试5个S3
                    # 构造S2->S3的序列
                    s3_id = stoi[str(s3)]
                    x = torch.tensor([[s2_id, s3_id]], device=self.device)
                    logits, _ = model(x)
                    
                    # 看模型预测S3的概率
                    prob = torch.softmax(logits[0, 0], dim=-1)[s3_id].item()
                    s3_responses.append(prob)
                
                avg_s1_response = np.mean(s1_responses)
                avg_s3_response = np.mean(s3_responses)
                
                all_s1_responses.extend(s1_responses)
                all_s3_responses.extend(s3_responses)
                
                s2_individual_biases[s2] = {
                    's1_response': avg_s1_response,
                    's3_response': avg_s3_response,
                    'bias': avg_s3_response / (avg_s1_response + 1e-8)
                }
        
        # 计算整体统计
        avg_s1_response = np.mean(all_s1_responses)
        avg_s3_response = np.mean(all_s3_responses)
        
        return {
            's1_response': avg_s1_response,
            's3_response': avg_s3_response,
            'bias_ratio': avg_s3_response / (avg_s1_response + 1e-8),
            'individual_biases': s2_individual_biases
        }
    
    def analyze_path_specific_gradients(self, model, data_dir, model_name, iteration):
        """分析路径特定的梯度贡献"""
        model.train()
        
        # 加载数据
        with open(os.path.join(data_dir, 'stage_info.pkl'), 'rb') as f:
            stage_info = pickle.load(f)
        S1, S2, S3 = stage_info['stages']
        
        with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
        stoi = meta['stoi']
        
        # 为不同路径类型准备损失
        path_losses = {'s1_s2': [], 's2_s3': [], 's1_s3': []}
        
        # S1->S2路径
        for _ in range(10):
            s1 = np.random.choice(S1)
            s2 = np.random.choice(S2)
            
            tokens = [stoi[str(s1)], stoi[str(s2)], stoi[str(s1)]]
            x = torch.tensor(tokens[:-1], device=self.device).unsqueeze(0)
            y = torch.tensor(tokens[1:], device=self.device).unsqueeze(0)
            
            logits, loss = model(x, y)
            path_losses['s1_s2'].append(loss.item())
        
        # S2->S3路径
        for _ in range(10):
            s2 = np.random.choice(S2)
            s3 = np.random.choice(S3)
            
            tokens = [stoi[str(s2)], stoi[str(s3)], stoi[str(s2)]]
            x = torch.tensor(tokens[:-1], device=self.device).unsqueeze(0)
            y = torch.tensor(tokens[1:], device=self.device).unsqueeze(0)
            
            logits, loss = model(x, y)
            path_losses['s2_s3'].append(loss.item())
        
        return {
            's1_s2_loss': np.mean(path_losses['s1_s2']),
            's2_s3_loss': np.mean(path_losses['s2_s3'])
        }
    
    def run_analysis(self):
        """运行完整分析"""
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
        
        print("\nRunning Gradient Response Analysis...")
        print("="*60)
        
        for model_name, config in model_configs.items():
            print(f"\nAnalyzing {model_name}...")
            
            for iteration in iterations:
                checkpoint_path = os.path.join(config['checkpoint_dir'], f'ckpt_{iteration}.pt')
                
                if not os.path.exists(checkpoint_path):
                    continue
                
                print(f"  Iteration {iteration}...")
                model = self.load_model(checkpoint_path)
                
                # 响应分析
                response_analysis = self.analyze_gradient_attribution(
                    model, config['data_dir'], model_name, iteration
                )
                
                # 路径损失分析
                path_losses = self.analyze_path_specific_gradients(
                    model, config['data_dir'], model_name, iteration
                )
                
                # 保存结果
                self.gradient_history[model_name]['iteration'].append(iteration)
                self.gradient_history[model_name]['s1_s2_grad'].append(response_analysis['s1_response'])
                self.gradient_history[model_name]['s2_s3_grad'].append(response_analysis['s3_response'])
                self.gradient_history[model_name]['ratio'].append(response_analysis['bias_ratio'])
                self.gradient_history[model_name]['s1_s2_loss'].append(path_losses['s1_s2_loss'])
                self.gradient_history[model_name]['s2_s3_loss'].append(path_losses['s2_s3_loss'])
                
                print(f"    S2→S1 response: {response_analysis['s1_response']:.4f}")
                print(f"    S2→S3 response: {response_analysis['s3_response']:.4f}")
                print(f"    Bias ratio: {response_analysis['bias_ratio']:.2f}")
                print(f"    S1→S2 loss: {path_losses['s1_s2_loss']:.4f}")
                print(f"    S2→S3 loss: {path_losses['s2_s3_loss']:.4f}")
    
    def plot_results(self):
        """绘制结果"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        colors = {'original': 'blue', '5% mixed': 'orange', '10% mixed': 'green'}
        
        for model_name, data in self.gradient_history.items():
            iterations = data['iteration']
            
            # 偏向比率
            ax1.plot(iterations, data['ratio'], 
                    label=model_name, color=colors[model_name], 
                    marker='o', markersize=8, linewidth=2)
            
            # 响应强度
            ax2.plot(iterations, data['s1_s2_grad'], 
                    label=f'{model_name} S2→S1', 
                    color=colors[model_name], linestyle='-', marker='s')
            ax2.plot(iterations, data['s2_s3_grad'], 
                    label=f'{model_name} S2→S3', 
                    color=colors[model_name], linestyle='--', marker='^')
            
            # 路径损失
            ax3.plot(iterations, data['s1_s2_loss'], 
                    label=f'{model_name} S1→S2', 
                    color=colors[model_name], linestyle='-', marker='o')
            ax3.plot(iterations, data['s2_s3_loss'], 
                    label=f'{model_name} S2→S3', 
                    color=colors[model_name], linestyle='--', marker='o')
            
            # 损失比率
            loss_ratio = [s23/s12 if s12 > 0 else 0 
                         for s12, s23 in zip(data['s1_s2_loss'], data['s2_s3_loss'])]
            ax4.plot(iterations, loss_ratio, 
                    label=model_name, color=colors[model_name], 
                    marker='o', markersize=8, linewidth=2)
        
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('S2 Bias Ratio (S3/S1)')
        ax1.set_title('S2 Representation Bias Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(bottom=0)
        
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Response Strength')
        ax2.set_title('S2 Response to Different Stages')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Loss')
        ax3.set_title('Path-specific Loss Values')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Loss Ratio (S2→S3 / S1→S2)')
        ax4.set_title('Relative Training Signal Strength')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'gradient_response_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存数值结果
        self.save_numerical_results()
    
    def save_numerical_results(self):
        """保存数值结果"""
        output_file = os.path.join(self.output_dir, 'gradient_response_results.txt')
        
        with open(output_file, 'w') as f:
            f.write("Gradient Response Analysis Results\n")
            f.write("="*60 + "\n\n")
            
            for model_name, data in self.gradient_history.items():
                f.write(f"{model_name}:\n")
                f.write("-"*30 + "\n")
                
                if data['iteration']:
                    # 初始和最终值
                    initial_idx = 0
                    final_idx = -1
                    
                    f.write(f"Initial ({data['iteration'][initial_idx]}k):\n")
                    f.write(f"  S2→S1 response: {data['s1_s2_grad'][initial_idx]:.4f}\n")
                    f.write(f"  S2→S3 response: {data['s2_s3_grad'][initial_idx]:.4f}\n")
                    f.write(f"  Bias ratio: {data['ratio'][initial_idx]:.2f}\n")
                    
                    f.write(f"\nFinal ({data['iteration'][final_idx]}k):\n")
                    f.write(f"  S2→S1 response: {data['s1_s2_grad'][final_idx]:.4f}\n")
                    f.write(f"  S2→S3 response: {data['s2_s3_grad'][final_idx]:.4f}\n")
                    f.write(f"  Bias ratio: {data['ratio'][final_idx]:.2f}\n")
                    
                    # 变化
                    f.write(f"\nChanges:\n")
                    f.write(f"  Bias ratio change: {data['ratio'][final_idx] - data['ratio'][initial_idx]:.2f}\n")
                
                f.write("\n")
        
        # 保存原始数据
        with open(os.path.join(self.output_dir, 'gradient_history.pkl'), 'wb') as f:
            pickle.dump(dict(self.gradient_history), f)
        
        print(f"Results saved to: {output_file}")

def main(output_dir=None):
    if output_dir is None:
        # 如果没有提供输出目录，创建一个新的
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f'deep_mechanism_analysis_{timestamp}'
        os.makedirs(output_dir, exist_ok=True)
    
    analyzer = GradientResponseAnalyzer(output_dir=output_dir)
    analyzer.run_analysis()
    analyzer.plot_results()
    
    print("\nGradient response analysis complete!")
    
    return output_dir

if __name__ == "__main__":
    main()