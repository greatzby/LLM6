import os
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import networkx as nx
from datetime import datetime
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import json
import random
from tqdm import tqdm

# 导入你的模型
from model import GPTConfig, GPT

class EnhancedCompositionAnalyzer:
    def __init__(self, config, device='cuda:0'):
        """
        config: {
            'model_name': {
                'checkpoint_dir': path,
                'data_dir': path
            }
        }
        """
        self.device = device
        self.config = config
        self.models = {}
        self.data_info = {}
        self.results = defaultdict(lambda: defaultdict(dict))
        
        # 为每个模型加载对应的数据信息
        for model_name, paths in config.items():
            data_dir = paths['data_dir']
            
            # 加载元信息
            with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
                meta = pickle.load(f)
            
            # 加载阶段信息
            with open(os.path.join(data_dir, 'stage_info.pkl'), 'rb') as f:
                stage_info = pickle.load(f)
            
            # 加载图
            G = nx.read_graphml(os.path.join(data_dir, 'composition_graph.graphml'))
            
            self.data_info[model_name] = {
                'meta': meta,
                'stage_info': stage_info,
                'stages': stage_info['stages'],
                'G': G,
                'test_file': os.path.join(data_dir, 'test.txt'),
                'stoi': meta['stoi'],
                'itos': meta['itos'],
                'vocab_size': meta['vocab_size'],
                'block_size': meta['block_size']
            }
            
            print(f"{model_name}: Loaded data from {data_dir}")
            print(f"  Graph nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")
    
    def load_checkpoint(self, model_name, iteration):
        """加载特定的checkpoint"""
        ckpt_path = os.path.join(
            self.config[model_name]['checkpoint_dir'], 
            f'ckpt_{iteration}.pt'
        )
        
        if not os.path.exists(ckpt_path):
            return None, None
            
        print(f"Loading {model_name} - iteration {iteration}")
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        
        # 初始化模型
        model_args = checkpoint['model_args']
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf).to(self.device)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        
        return model, checkpoint.get('iter_num', iteration)
    
    def extract_node_embeddings(self, model, nodes, model_name):
        """提取节点的嵌入表示"""
        embeddings = {}
        stoi = self.data_info[model_name]['stoi']
        
        with torch.no_grad():
            for node in nodes:
                if str(node) not in stoi:
                    continue
                    
                token_id = stoi[str(node)]
                token_tensor = torch.tensor([token_id], device=self.device)
                
                # 获取token embedding
                emb = model.transformer.wte(token_tensor)
                
                # 添加位置编码（位置0）
                pos = torch.zeros(1, 1, dtype=torch.long, device=self.device)
                pos_emb = model.transformer.wpe(pos)
                x = emb + pos_emb
                
                # 通过transformer层
                hidden_states = [x.squeeze().cpu().numpy()]
                for block in model.transformer.h:
                    x = block(x)
                    hidden_states.append(x.squeeze().cpu().numpy())
                
                embeddings[node] = {
                    'initial': hidden_states[0],
                    'final': hidden_states[-1],
                    'all_layers': hidden_states
                }
        
        return embeddings
    
    def analyze_s2_diversity(self, embeddings, S2_nodes):
        """分析S2内部的多样性"""
        s2_embs = []
        for node in S2_nodes:
            if node in embeddings:
                s2_embs.append(embeddings[node]['final'])
        
        if len(s2_embs) < 2:
            return None
        
        # 计算所有S2节点对之间的距离
        pairwise_distances = []
        pairwise_similarities = []
        
        for i in range(len(s2_embs)):
            for j in range(i+1, len(s2_embs)):
                # 欧氏距离
                dist = np.linalg.norm(s2_embs[i] - s2_embs[j])
                pairwise_distances.append(dist)
                
                # 余弦相似度
                sim = np.dot(s2_embs[i], s2_embs[j]) / (
                    np.linalg.norm(s2_embs[i]) * np.linalg.norm(s2_embs[j])
                )
                pairwise_similarities.append(sim)
        
        # 计算有效维度（PCA）
        s2_embs_array = np.array(s2_embs)
        pca = PCA()
        pca.fit(s2_embs_array)
        
        # 找到解释90%方差所需的维度数
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        effective_dim = np.argmax(cumsum >= 0.9) + 1
        
        return {
            'mean_distance': np.mean(pairwise_distances),
            'std_distance': np.std(pairwise_distances),
            'min_distance': np.min(pairwise_distances),
            'max_distance': np.max(pairwise_distances),
            'mean_similarity': np.mean(pairwise_similarities),
            'std_similarity': np.std(pairwise_similarities),
            'effective_dimension': effective_dim,
            'explained_variance_90': cumsum[effective_dim-1] if effective_dim > 0 else 0,
            'num_nodes': len(s2_embs)
        }
    
    def test_path_generation(self, model, source, target, model_name, max_attempts=3):
        """测试是否能生成有效路径"""
        data = self.data_info[model_name]
        
        for _ in range(max_attempts):
            # 生成路径
            prompt = f"{source} {target} {source}"
            prompt_ids = [data['stoi'][t] for t in prompt.split() if t in data['stoi']]
            x = torch.tensor(prompt_ids, dtype=torch.long, device=self.device).unsqueeze(0)
            
            with torch.no_grad():
                y = model.generate(x, max_new_tokens=30, temperature=0.1, top_k=10)
            
            # 解码
            generated = []
            for tid in y[0].tolist():
                if tid == 1:  # EOS
                    break
                if tid in data['itos']:
                    try:
                        generated.append(int(data['itos'][tid]))
                    except:
                        pass
            
            # 提取路径
            path = generated[2:] if len(generated) >= 3 else []
            
            # 验证路径
            if len(path) >= 2 and path[0] == source and path[-1] == target:
                # 检查边的有效性
                edges_valid = all(
                    data['G'].has_edge(str(path[i]), str(path[i+1]))
                    for i in range(len(path)-1)
                )
                if edges_valid:
                    return True, path
        
        return False, None
    
    def analyze_s2_functionality(self, model, model_name):
        """测试S2节点的功能性"""
        stages = self.data_info[model_name]['stages']
        S1, S2, S3 = stages
        
        s2_functionality = {}
        
        # 对每个S2节点进行测试
        for s2 in tqdm(S2, desc="Testing S2 functionality"):
            # 测试从S1到S2的可达性
            s1_sample = random.sample(S1, min(5, len(S1)))
            reachable_from_s1 = 0
            for s1 in s1_sample:
                success, _ = self.test_path_generation(model, s1, s2, model_name)
                if success:
                    reachable_from_s1 += 1
            
            # 测试从S2到S3的可达性
            s3_sample = random.sample(S3, min(5, len(S3)))
            can_reach_s3 = 0
            for s3 in s3_sample:
                success, _ = self.test_path_generation(model, s2, s3, model_name)
                if success:
                    can_reach_s3 += 1
            
            # 测试作为桥梁的功能（S1->S2->S3）
            bridge_success = 0
            bridge_attempts = 5
            for _ in range(bridge_attempts):
                s1 = random.choice(S1)
                s3 = random.choice(S3)
                
                # 检查是否能通过这个S2节点
                success1, path1 = self.test_path_generation(model, s1, s2, model_name)
                success2, path2 = self.test_path_generation(model, s2, s3, model_name)
                
                if success1 and success2:
                    # 测试直接S1->S3是否经过这个S2
                    success3, path3 = self.test_path_generation(model, s1, s3, model_name)
                    if success3 and s2 in path3:
                        bridge_success += 1
            
            s2_functionality[s2] = {
                'reachability_from_s1': reachable_from_s1 / len(s1_sample),
                'reachability_to_s3': can_reach_s3 / len(s3_sample),
                'bridge_success_rate': bridge_success / bridge_attempts
            }
        
        # 计算整体统计
        overall_stats = {
            'mean_s1_reachability': np.mean([f['reachability_from_s1'] for f in s2_functionality.values()]),
            'mean_s3_reachability': np.mean([f['reachability_to_s3'] for f in s2_functionality.values()]),
            'mean_bridge_success': np.mean([f['bridge_success_rate'] for f in s2_functionality.values()]),
            'functional_nodes': sum(1 for f in s2_functionality.values() 
                                  if f['reachability_from_s1'] > 0.5 and f['reachability_to_s3'] > 0.5)
        }
        
        return s2_functionality, overall_stats
    
    def analyze_path_success_rates(self, model, model_name, num_samples=50):
        """分析不同类型路径的成功率"""
        data = self.data_info[model_name]
        stages = data['stages']
        S1, S2, S3 = stages
        
        # 读取测试集
        with open(data['test_file'], 'r') as f:
            test_lines = [line.strip() for line in f if line.strip()]
        
        # 按类型分组
        test_cases = {'S1->S2': [], 'S2->S3': [], 'S1->S3': []}
        for line in test_lines:
            parts = line.split()
            if len(parts) >= 2:
                source, target = int(parts[0]), int(parts[1])
                if source in S1 and target in S2:
                    test_cases['S1->S2'].append((source, target))
                elif source in S2 and target in S3:
                    test_cases['S2->S3'].append((source, target))
                elif source in S1 and target in S3:
                    test_cases['S1->S3'].append((source, target))
        
        # 分析每种类型
        results = {}
        error_analysis = {}
        
        for path_type, cases in test_cases.items():
            # 采样
            sample_cases = random.sample(cases, min(num_samples, len(cases)))
            
            successes = 0
            errors = defaultdict(int)
            successful_paths = []
            
            for source, target in tqdm(sample_cases, desc=f"Testing {path_type}"):
                success, path = self.test_path_generation(model, source, target, model_name)
                
                if success:
                    successes += 1
                    successful_paths.append(path)
                    
                    # 对于S1->S3，分析S2使用情况
                    if path_type == 'S1->S3':
                        s2_nodes = [n for n in path[1:-1] if n in S2]
                        errors['s2_count_' + str(len(s2_nodes))] += 1
                else:
                    errors['generation_failed'] += 1
            
            results[path_type] = {
                'success_rate': successes / len(sample_cases),
                'total_tested': len(sample_cases),
                'successful_paths': successful_paths
            }
            error_analysis[path_type] = dict(errors)
        
        return results, error_analysis
    
    def analyze_representation_collapse(self, embeddings, stages):
        """分析表示空间的坍缩程度"""
        S1, S2, S3 = stages
        
        # 收集各阶段的嵌入
        stage_embeddings = {
            'S1': [embeddings[n]['final'] for n in S1 if n in embeddings],
            'S2': [embeddings[n]['final'] for n in S2 if n in embeddings],
            'S3': [embeddings[n]['final'] for n in S3 if n in embeddings]
        }
        
        collapse_metrics = {}
        
        for stage_name, embs in stage_embeddings.items():
            if len(embs) < 2:
                continue
                
            embs_array = np.array(embs)
            
            # 1. 计算有效秩（通过SVD）
            U, s, Vt = np.linalg.svd(embs_array.T, full_matrices=False)
            
            # 相对于最大奇异值的阈值
            threshold = s[0] * 1e-3
            effective_rank = np.sum(s > threshold)
            
            # 2. 计算各维度的方差
            dim_variances = np.var(embs_array, axis=0)
            active_dims = np.sum(dim_variances > 1e-6)
            
            # 3. 计算平均最近邻距离
            min_distances = []
            for i, emb in enumerate(embs):
                distances = [np.linalg.norm(emb - embs[j]) 
                           for j in range(len(embs)) if i != j]
                if distances:
                    min_distances.append(min(distances))
            
            collapse_metrics[stage_name] = {
                'effective_rank': effective_rank,
                'active_dimensions': active_dims,
                'mean_nearest_neighbor_dist': np.mean(min_distances) if min_distances else 0,
                'singular_values': s[:10].tolist(),  # 前10个奇异值
                'dimension_variance_ratio': active_dims / len(dim_variances)
            }
        
        return collapse_metrics
    
    def run_enhanced_analysis(self, iterations):
        """运行增强的完整分析"""
        for model_name in self.config.keys():
            print(f"\n{'='*60}")
            print(f"Analyzing {model_name}")
            print('='*60)
            
            for iteration in iterations:
                print(f"\nIteration {iteration}:")
                
                model, _ = self.load_checkpoint(model_name, iteration)
                if model is None:
                    continue
                
                # 基础信息
                stages = self.data_info[model_name]['stages']
                S1, S2, S3 = stages
                all_nodes = list(range(90))
                
                # 1. 提取嵌入
                print("  Extracting embeddings...")
                embeddings = self.extract_node_embeddings(model, all_nodes, model_name)
                
                # 2. S2多样性分析
                print("  Analyzing S2 diversity...")
                diversity_metrics = self.analyze_s2_diversity(embeddings, S2)
                
                # 3. 表示坍缩分析
                print("  Analyzing representation collapse...")
                collapse_metrics = self.analyze_representation_collapse(embeddings, stages)
                
                # 4. 路径成功率（每5000次迭代详细测试）
                if iteration % 5000 == 0:
                    print("  Testing path generation success rates...")
                    path_results, error_analysis = self.analyze_path_success_rates(model, model_name)
                    
                    # S2功能性分析（比较耗时，只在关键迭代进行）
                    if iteration in [5000, 25000, 50000]:
                        print("  Analyzing S2 functionality (this may take a while)...")
                        s2_func, overall_func = self.analyze_s2_functionality(model, model_name)
                    else:
                        s2_func, overall_func = None, None
                else:
                    path_results = None
                    error_analysis = None
                    s2_func = None
                    overall_func = None
                
                # 保存结果
                self.results[model_name][iteration] = {
                    'diversity_metrics': diversity_metrics,
                    'collapse_metrics': collapse_metrics,
                    'path_results': path_results,
                    'error_analysis': error_analysis,
                    's2_functionality': s2_func,
                    's2_overall_functionality': overall_func,
                    'embeddings': embeddings  # 保存用于可视化
                }
                
                # 打印关键指标
                if diversity_metrics:
                    print(f"  S2 diversity - mean distance: {diversity_metrics['mean_distance']:.3f}")
                    print(f"  S2 effective dimension: {diversity_metrics['effective_dimension']}")
                
                if collapse_metrics and 'S2' in collapse_metrics:
                    print(f"  S2 effective rank: {collapse_metrics['S2']['effective_rank']}")
                
                if overall_func:
                    print(f"  S2 functional nodes: {overall_func['functional_nodes']}/{len(S2)}")
                    print(f"  S2 bridge success rate: {overall_func['mean_bridge_success']:.2%}")
    
    def create_enhanced_visualizations(self, output_dir):
        """创建增强的可视化"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. S2多样性演化
        self.plot_s2_diversity_evolution(output_dir)
        
        # 2. 表示坍缩分析
        self.plot_representation_collapse(output_dir)
        
        # 3. 路径成功率
        self.plot_path_success_rates(output_dir)
        
        # 4. S2功能性热图
        self.plot_s2_functionality_heatmap(output_dir)
        
        # 5. 综合仪表板
        self.create_comprehensive_dashboard(output_dir)
    
    def plot_s2_diversity_evolution(self, output_dir):
        """绘制S2多样性演化"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        metrics_to_plot = [
            ('mean_distance', 'Mean Pairwise Distance'),
            ('std_distance', 'Std of Pairwise Distance'),
            ('effective_dimension', 'Effective Dimension (90% variance)'),
            ('mean_similarity', 'Mean Pairwise Similarity')
        ]
        
        for idx, (metric, title) in enumerate(metrics_to_plot):
            ax = axes[idx // 2, idx % 2]
            
            for model_name in self.config.keys():
                iterations = []
                values = []
                
                for iter_num in sorted(self.results[model_name].keys()):
                    if 'diversity_metrics' in self.results[model_name][iter_num]:
                        div_metrics = self.results[model_name][iter_num]['diversity_metrics']
                        if div_metrics and metric in div_metrics:
                            iterations.append(iter_num)
                            values.append(div_metrics[metric])
                
                if iterations:
                    ax.plot(iterations, values, 'o-', label=model_name, linewidth=2)
            
            ax.set_xlabel('Iteration')
            ax.set_ylabel(title)
            ax.set_title(f'S2 {title} Evolution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 's2_diversity_evolution.png'), dpi=300)
        plt.close()
    
    def plot_representation_collapse(self, output_dir):
        """绘制表示坍缩分析"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for stage_idx, stage_name in enumerate(['S1', 'S2', 'S3']):
            ax = axes[stage_idx]
            
            for model_name in self.config.keys():
                iterations = []
                effective_ranks = []
                
                for iter_num in sorted(self.results[model_name].keys()):
                    if 'collapse_metrics' in self.results[model_name][iter_num]:
                        collapse = self.results[model_name][iter_num]['collapse_metrics']
                        if stage_name in collapse:
                            iterations.append(iter_num)
                            effective_ranks.append(collapse[stage_name]['effective_rank'])
                
                if iterations:
                    ax.plot(iterations, effective_ranks, 'o-', label=model_name, linewidth=2)
            
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Effective Rank')
            ax.set_title(f'{stage_name} Representation Effective Rank')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'representation_collapse.png'), dpi=300)
        plt.close()
    
    def plot_path_success_rates(self, output_dir):
        """绘制路径成功率"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        path_types = ['S1->S2', 'S2->S3', 'S1->S3']
        
        for idx, path_type in enumerate(path_types):
            ax = axes[idx]
            
            for model_name in self.config.keys():
                iterations = []
                success_rates = []
                
                for iter_num in sorted(self.results[model_name].keys()):
                    if 'path_results' in self.results[model_name][iter_num]:
                        path_res = self.results[model_name][iter_num]['path_results']
                        if path_res and path_type in path_res:
                            iterations.append(iter_num)
                            success_rates.append(path_res[path_type]['success_rate'])
                
                if iterations:
                    ax.plot(iterations, success_rates, 'o-', label=model_name, linewidth=2, markersize=8)
            
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Success Rate')
            ax.set_title(f'{path_type} Generation Success Rate')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.05)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'path_success_rates.png'), dpi=300)
        plt.close()
    
    def plot_s2_functionality_heatmap(self, output_dir):
        """绘制S2功能性热图"""
        key_iterations = [5000, 25000, 50000]
        
        fig, axes = plt.subplots(len(self.config), len(key_iterations), 
                                figsize=(4*len(key_iterations), 4*len(self.config)))
        
        if len(self.config) == 1:
            axes = axes.reshape(1, -1)
        
        for model_idx, model_name in enumerate(self.config.keys()):
            for iter_idx, iteration in enumerate(key_iterations):
                ax = axes[model_idx, iter_idx]
                
                if iteration in self.results[model_name] and \
                   self.results[model_name][iteration]['s2_functionality']:
                    
                    s2_func = self.results[model_name][iteration]['s2_functionality']
                    S2 = self.data_info[model_name]['stages'][1]
                    
                    # 创建功能性矩阵
                    func_matrix = []
                    node_labels = []
                    
                    for node in sorted(S2):
                        if node in s2_func:
                            func_matrix.append([
                                s2_func[node]['reachability_from_s1'],
                                s2_func[node]['reachability_to_s3'],
                                s2_func[node]['bridge_success_rate']
                            ])
                            node_labels.append(str(node))
                    
                    if func_matrix:
                        func_matrix = np.array(func_matrix).T
                        
                        sns.heatmap(func_matrix, 
                                   xticklabels=node_labels,
                                   yticklabels=['From S1', 'To S3', 'Bridge'],
                                   cmap='RdYlGn',
                                   vmin=0, vmax=1,
                                   ax=ax,
                                   cbar_kws={'label': 'Success Rate'})
                        
                        ax.set_title(f'{model_name} - Iter {iteration}')
                else:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                    ax.set_title(f'{model_name} - Iter {iteration}')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 's2_functionality_heatmap.png'), dpi=300)
        plt.close()
    
    def create_comprehensive_dashboard(self, output_dir):
        """创建综合仪表板"""
        fig = plt.figure(figsize=(20, 16))
        
        # 定义布局
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # 1. S2内聚性 vs 多样性
        ax1 = fig.add_subplot(gs[0, :])
        self.plot_cohesion_vs_diversity(ax1)
        
        # 2. 有效维度
        ax2 = fig.add_subplot(gs[1, 0])
        self.plot_effective_dimensions(ax2)
        
        # 3. S2功能性得分
        ax3 = fig.add_subplot(gs[1, 1])
        self.plot_functionality_scores(ax3)
        
        # 4. 路径成功率（S1->S3）
        ax4 = fig.add_subplot(gs[1, 2])
        self.plot_s1_s3_success(ax4)
        
        # 5. 关键发现文本
        ax5 = fig.add_subplot(gs[2:, :])
        self.add_key_findings_text(ax5)
        
        plt.savefig(os.path.join(output_dir, 'comprehensive_dashboard.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_cohesion_vs_diversity(self, ax):
        """绘制内聚性vs多样性"""
        for model_name in self.config.keys():
            iterations = []
            diversity = []
            
            for iter_num in sorted(self.results[model_name].keys()):
                if 'diversity_metrics' in self.results[model_name][iter_num]:
                    div_metrics = self.results[model_name][iter_num]['diversity_metrics']
                    if div_metrics:
                        iterations.append(iter_num)
                        # 使用mean_distance的倒数作为内聚性的代理
                        diversity.append(div_metrics['mean_distance'])
            
            if iterations:
                ax.plot(iterations, diversity, 'o-', label=f'{model_name} diversity', 
                       linewidth=2, markersize=6)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('S2 Mean Pairwise Distance')
        ax.set_title('S2 Diversity Evolution (Lower = More Cohesive/Less Diverse)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 添加关键区域
        ax.axvspan(0, 10000, alpha=0.1, color='green', label='Early success')
        ax.axvspan(20000, 35000, alpha=0.1, color='red', label='Degradation')
    
    def plot_effective_dimensions(self, ax):
        """绘制有效维度"""
        for model_name in self.config.keys():
            iterations = []
            eff_dims = []
            
            for iter_num in sorted(self.results[model_name].keys()):
                if 'diversity_metrics' in self.results[model_name][iter_num]:
                    div_metrics = self.results[model_name][iter_num]['diversity_metrics']
                    if div_metrics and 'effective_dimension' in div_metrics:
                        iterations.append(iter_num)
                        eff_dims.append(div_metrics['effective_dimension'])
            
            if iterations:
                ax.plot(iterations, eff_dims, 'o-', label=model_name, linewidth=2)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Effective Dimension')
        ax.set_title('S2 Effective Dimension (90% variance)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_functionality_scores(self, ax):
        """绘制S2功能性得分"""
        key_iters = [5000, 25000, 50000]
        
        data_for_plot = defaultdict(list)
        
        for model_name in self.config.keys():
            for iter_num in key_iters:
                if iter_num in self.results[model_name]:
                    overall_func = self.results[model_name][iter_num].get('s2_overall_functionality')
                    if overall_func:
                        data_for_plot[model_name].append(overall_func['mean_bridge_success'])
                    else:
                        data_for_plot[model_name].append(0)
        
        # 绘制
        x = np.arange(len(key_iters))
        width = 0.25
        
        for i, model_name in enumerate(self.config.keys()):
            if model_name in data_for_plot:
                ax.bar(x + i*width, data_for_plot[model_name], 
                      width, label=model_name)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Mean Bridge Success Rate')
        ax.set_title('S2 Bridge Functionality')
        ax.set_xticks(x + width)
        ax.set_xticklabels(key_iters)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    def plot_s1_s3_success(self, ax):
        """绘制S1->S3成功率"""
        for model_name in self.config.keys():
            iterations = []
            success_rates = []
            
            for iter_num in sorted(self.results[model_name].keys()):
                if 'path_results' in self.results[model_name][iter_num]:
                    path_res = self.results[model_name][iter_num]['path_results']
                    if path_res and 'S1->S3' in path_res:
                        iterations.append(iter_num)
                        success_rates.append(path_res['S1->S3']['success_rate'])
            
            if iterations:
                ax.plot(iterations, success_rates, 'o-', label=model_name, 
                       linewidth=3, markersize=10)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Success Rate')
        ax.set_title('S1→S3 Compositional Success Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        
        # 添加重要阈值线
        ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='80% threshold')
    
    def add_key_findings_text(self, ax):
        """添加关键发现文本"""
        ax.axis('off')
        
        findings = []
        findings.append("KEY FINDINGS:\n")
        
        # 分析每个模型在关键迭代的表现
        for model_name in self.config.keys():
            findings.append(f"\n{model_name.upper()}:")
            
            # 早期（5k）vs 晚期（50k）对比
            if 5000 in self.results[model_name] and 50000 in self.results[model_name]:
                early = self.results[model_name][5000]
                late = self.results[model_name][50000]
                
                # S2多样性变化
                if 'diversity_metrics' in early and 'diversity_metrics' in late:
                    early_div = early['diversity_metrics']
                    late_div = late['diversity_metrics']
                    
                    if early_div and late_div:
                        div_change = (late_div['mean_distance'] - early_div['mean_distance']) / early_div['mean_distance'] * 100
                        findings.append(f"  • S2 diversity change: {div_change:+.1f}%")
                        findings.append(f"  • Effective dimension: {early_div['effective_dimension']} → {late_div['effective_dimension']}")
                
                # 路径成功率
                if 'path_results' in early and 'path_results' in late:
                    early_s13 = early['path_results'].get('S1->S3', {}).get('success_rate', 0)
                    late_s13 = late['path_results'].get('S1->S3', {}).get('success_rate', 0)
                    findings.append(f"  • S1→S3 success: {early_s13:.1%} → {late_s13:.1%}")
        
        # 添加解释
        findings.append("\n\nINTERPRETATION:")
        findings.append("• High cohesion (low diversity) in S2 indicates representation collapse")
        findings.append("• Reduced effective dimension suggests loss of representational capacity")
        findings.append("• Mixed training maintains S2 diversity and functionality")
        
        text = '\n'.join(findings)
        ax.text(0.05, 0.95, text, transform=ax.transAxes, 
               fontsize=11, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def generate_enhanced_report(self, output_dir):
        """生成增强的分析报告"""
        report = []
        report.append("# Enhanced Transformer Composition Analysis Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 执行摘要
        report.append("## Executive Summary\n")
        
        # 1. S2多样性分析
        report.append("## 1. S2 Diversity Analysis\n")
        
        for model_name in self.config.keys():
            report.append(f"### {model_name}")
            
            # 比较5k和50k
            if 5000 in self.results[model_name] and 50000 in self.results[model_name]:
                early = self.results[model_name][5000].get('diversity_metrics', {})
                late = self.results[model_name][50000].get('diversity_metrics', {})
                
                if early and late:
                    report.append(f"\n**Diversity Metrics Evolution (5k → 50k):**")
                    report.append(f"- Mean pairwise distance: {early['mean_distance']:.3f} → {late['mean_distance']:.3f}")
                    report.append(f"- Effective dimension: {early['effective_dimension']} → {late['effective_dimension']}")
                    report.append(f"- Distance std: {early['std_distance']:.3f} → {late['std_distance']:.3f}")
            
            report.append("")
        
        # 2. 表示坍缩分析
        report.append("## 2. Representation Collapse Analysis\n")
        
        for model_name in self.config.keys():
            if 50000 in self.results[model_name]:
                collapse = self.results[model_name][50000].get('collapse_metrics', {})
                if 'S2' in collapse:
                    report.append(f"### {model_name} (at 50k iterations)")
                    report.append(f"- S2 effective rank: {collapse['S2']['effective_rank']}")
                    report.append(f"- S2 active dimensions: {collapse['S2']['active_dimensions']}")
                    report.append(f"- S2 dimension utilization: {collapse['S2']['dimension_variance_ratio']:.1%}")
                    report.append("")
        
        # 3. 功能性分析
        report.append("## 3. S2 Functionality Analysis\n")
        
        for model_name in self.config.keys():
            report.append(f"### {model_name}")
            
            # 收集所有功能性数据
            functionality_timeline = []
            for iter_num in [5000, 25000, 50000]:
                if iter_num in self.results[model_name]:
                    overall_func = self.results[model_name][iter_num].get('s2_overall_functionality')
                    if overall_func:
                        functionality_timeline.append((iter_num, overall_func))
            
            if functionality_timeline:
                report.append("\n| Iteration | Functional Nodes | Bridge Success | S1 Reach | S3 Reach |")
                report.append("|-----------|------------------|----------------|----------|----------|")
                
                for iter_num, func in functionality_timeline:
                    report.append(f"| {iter_num} | {func['functional_nodes']}/30 | "
                                f"{func['mean_bridge_success']:.1%} | "
                                f"{func['mean_s1_reachability']:.1%} | "
                                f"{func['mean_s3_reachability']:.1%} |")
            
            report.append("")
        
        # 4. 路径生成成功率
        report.append("## 4. Path Generation Success Rates\n")
        
        # 创建对比表
        report.append("### Final Performance (50k iterations)\n")
        report.append("| Model | S1→S2 | S2→S3 | S1→S3 |")
        report.append("|-------|-------|-------|-------|")
        
        for model_name in self.config.keys():
            if 50000 in self.results[model_name]:
                path_res = self.results[model_name][50000].get('path_results', {})
                if path_res:
                    s12 = path_res.get('S1->S2', {}).get('success_rate', 0)
                    s23 = path_res.get('S2->S3', {}).get('success_rate', 0)
                    s13 = path_res.get('S1->S3', {}).get('success_rate', 0)
                    report.append(f"| {model_name} | {s12:.1%} | {s23:.1%} | {s13:.1%} |")
        
        # 5. 结论
        report.append("\n## 5. Conclusions\n")
        report.append("1. **S2 Cohesion Paradox**: Higher cohesion indicates collapse, not strength")
        report.append("2. **Diversity is Key**: S2 nodes need diversity to bridge S1 and S3")
        report.append("3. **Functional Degradation**: S2 nodes lose their bridging capability over training")
        report.append("4. **Mixed Training Benefits**: Maintains S2 diversity and functionality")
        
        # 保存报告
        report_path = os.path.join(output_dir, 'enhanced_analysis_report.md')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"\nEnhanced report saved to: {report_path}")
    
    def save_all_results(self, output_dir):
        """保存所有结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存原始数据
        with open(os.path.join(output_dir, 'enhanced_results.pkl'), 'wb') as f:
            pickle.dump(dict(self.results), f)
        
        # 创建可视化
        print("\nCreating visualizations...")
        self.create_enhanced_visualizations(output_dir)
        
        # 生成报告
        print("Generating report...")
        self.generate_enhanced_report(output_dir)
        
        # 保存关键指标的CSV
        self.export_key_metrics_csv(output_dir)
    
    def export_key_metrics_csv(self, output_dir):
        """导出关键指标到CSV"""
        rows = []
        
        for model_name in self.config.keys():
            for iter_num in sorted(self.results[model_name].keys()):
                row = {
                    'model': model_name,
                    'iteration': iter_num
                }
                
                # 添加多样性指标
                if 'diversity_metrics' in self.results[model_name][iter_num]:
                    div = self.results[model_name][iter_num]['diversity_metrics']
                    if div:
                        row.update({
                            's2_mean_distance': div['mean_distance'],
                            's2_effective_dim': div['effective_dimension'],
                            's2_mean_similarity': div['mean_similarity']
                        })
                
                # 添加路径成功率
                if 'path_results' in self.results[model_name][iter_num]:
                    path = self.results[model_name][iter_num]['path_results']
                    if path:
                        for ptype in ['S1->S2', 'S2->S3', 'S1->S3']:
                            if ptype in path:
                                row[f'success_{ptype}'] = path[ptype]['success_rate']
                
                rows.append(row)
        
        # 保存为CSV
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(output_dir, 'key_metrics.csv'), index=False)
        print(f"Key metrics exported to: {os.path.join(output_dir, 'key_metrics.csv')}")

def main():
    # 配置
    config = {
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
    
    # 要分析的迭代
    iterations = [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000]
    
    # 输出目录
    output_dir = f'enhanced_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    
    print("="*60)
    print("Enhanced Composition Analysis")
    print("="*60)
    
    # 创建分析器
    analyzer = EnhancedCompositionAnalyzer(config, device='cuda:0')
    
    # 运行增强分析
    analyzer.run_enhanced_analysis(iterations)
    
    # 保存结果和生成报告
    analyzer.save_all_results(output_dir)
    
    print(f"\n{'='*60}")
    print(f"Analysis complete! Results saved to: {output_dir}")
    print(f"Check the following files:")
    print(f"  - enhanced_analysis_report.md: Detailed findings")
    print(f"  - *.png: Visualization plots")
    print(f"  - key_metrics.csv: Metrics in spreadsheet format")
    print(f"  - enhanced_results.pkl: Raw data for further analysis")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()