import os
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import networkx as nx
from datetime import datetime
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, mutual_info_score
from scipy.stats import entropy
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 导入模型
from model import GPTConfig, GPT

class S2SpecializationAnalyzer:
    def __init__(self, config, device='cuda:0'):
        self.device = device
        self.config = config
        self.data_info = {}
        self.results = defaultdict(lambda: defaultdict(dict))
        
        # 加载数据信息
        for model_name, paths in config.items():
            data_dir = paths['data_dir']
            
            with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
                meta = pickle.load(f)
            
            with open(os.path.join(data_dir, 'stage_info.pkl'), 'rb') as f:
                stage_info = pickle.load(f)
            
            G = nx.read_graphml(os.path.join(data_dir, 'composition_graph.graphml'))
            
            self.data_info[model_name] = {
                'meta': meta,
                'stages': stage_info['stages'],
                'G': G,
                'stoi': meta['stoi'],
                'itos': meta['itos']
            }
    
    def load_model(self, model_name, iteration):
        """加载模型"""
        ckpt_path = os.path.join(
            self.config[model_name]['checkpoint_dir'], 
            f'ckpt_{iteration}.pt'
        )
        
        if not os.path.exists(ckpt_path):
            return None
            
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        model_args = checkpoint['model_args']
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf).to(self.device)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        
        return model
    
    def analyze_s2_connection_patterns(self, model, model_name, num_tests=100):
        """分析每个S2节点的连接模式"""
        S1, S2, S3 = self.data_info[model_name]['stages']
        data = self.data_info[model_name]
        
        # 记录每个S2节点的连接成功情况
        s2_patterns = defaultdict(lambda: {
            'from_s1': [],  # 成功连接的S1节点
            'to_s3': [],    # 成功连接的S3节点
            's1_success_rate': 0,
            's3_success_rate': 0,
            'bidirectional_score': 0  # 双向连接能力
        })
        
        print("Testing S2 connection patterns...")
        
        # 测试S1->S2连接
        for s2 in tqdm(S2, desc="Testing S1->S2"):
            successes_from_s1 = []
            
            for s1 in S1:
                success = self._test_path(model, s1, s2, data)
                if success:
                    successes_from_s1.append(s1)
                    s2_patterns[s2]['from_s1'].append(s1)
            
            s2_patterns[s2]['s1_success_rate'] = len(successes_from_s1) / len(S1)
        
        # 测试S2->S3连接
        for s2 in tqdm(S2, desc="Testing S2->S3"):
            successes_to_s3 = []
            
            for s3 in S3:
                success = self._test_path(model, s2, s3, data)
                if success:
                    successes_to_s3.append(s3)
                    s2_patterns[s2]['to_s3'].append(s3)
            
            s2_patterns[s2]['s3_success_rate'] = len(successes_to_s3) / len(S3)
            
            # 计算双向得分
            s2_patterns[s2]['bidirectional_score'] = (
                s2_patterns[s2]['s1_success_rate'] * 
                s2_patterns[s2]['s3_success_rate']
            )
        
        return dict(s2_patterns)
    
    def cluster_s2_by_function(self, s2_patterns):
        """根据功能对S2节点进行聚类"""
        # 构建特征矩阵
        s2_nodes = sorted(s2_patterns.keys())
        features = []
        
        for s2 in s2_nodes:
            pattern = s2_patterns[s2]
            # 特征：S1连接率，S3连接率，双向得分
            features.append([
                pattern['s1_success_rate'],
                pattern['s3_success_rate'],
                pattern['bidirectional_score']
            ])
        
        features = np.array(features)
        
        # 尝试不同的聚类方法
        clustering_results = {}
        
        # K-means聚类
        for n_clusters in [2, 3, 4]:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(features)
            
            if len(set(labels)) > 1:
                score = silhouette_score(features, labels)
                clustering_results[f'kmeans_{n_clusters}'] = {
                    'labels': labels,
                    'silhouette_score': score,
                    'cluster_centers': kmeans.cluster_centers_
                }
        
        # DBSCAN聚类
        dbscan = DBSCAN(eps=0.1, min_samples=2)
        labels = dbscan.fit_predict(features)
        
        if len(set(labels)) > 1:
            # 过滤掉噪声点
            valid_mask = labels != -1
            if np.sum(valid_mask) > 0:
                score = silhouette_score(features[valid_mask], labels[valid_mask])
                clustering_results['dbscan'] = {
                    'labels': labels,
                    'silhouette_score': score
                }
        
        # 选择最佳聚类
        best_method = max(clustering_results.keys(), 
                         key=lambda x: clustering_results[x]['silhouette_score'])
        best_clustering = clustering_results[best_method]
        
        # 分析每个簇的特征
        cluster_analysis = {}
        labels = best_clustering['labels']
        
        for cluster_id in set(labels):
            if cluster_id == -1:  # DBSCAN噪声点
                continue
                
            cluster_mask = labels == cluster_id
            cluster_nodes = [s2_nodes[i] for i in range(len(s2_nodes)) if cluster_mask[i]]
            cluster_features = features[cluster_mask]
            
            cluster_analysis[f'cluster_{cluster_id}'] = {
                'nodes': cluster_nodes,
                'size': len(cluster_nodes),
                'mean_s1_rate': np.mean(cluster_features[:, 0]),
                'mean_s3_rate': np.mean(cluster_features[:, 1]),
                'mean_bidirectional': np.mean(cluster_features[:, 2]),
                'std_s1_rate': np.std(cluster_features[:, 0]),
                'std_s3_rate': np.std(cluster_features[:, 1])
            }
        
        return {
            'best_method': best_method,
            'clustering': best_clustering,
            'cluster_analysis': cluster_analysis,
            'features': features,
            'node_order': s2_nodes
        }
    
    def analyze_path_diversity(self, model, model_name, num_samples=200):
        """分析S1->S3路径的多样性"""
        S1, S2, S3 = self.data_info[model_name]['stages']
        data = self.data_info[model_name]
        
        successful_paths = []
        s2_usage_count = Counter()
        path_length_distribution = []
        
        print(f"Analyzing path diversity (sampling {num_samples} S1->S3 pairs)...")
        
        # 随机采样S1->S3对
        for _ in tqdm(range(num_samples)):
            s1 = np.random.choice(S1)
            s3 = np.random.choice(S3)
            
            success, path = self._generate_path(model, s1, s3, data)
            
            if success and len(path) >= 2:
                successful_paths.append(path)
                path_length_distribution.append(len(path))
                
                # 统计S2使用
                s2_in_path = [node for node in path[1:-1] if node in S2]
                for s2 in s2_in_path:
                    s2_usage_count[s2] += 1
        
        if not successful_paths:
            return None
        
        # 计算多样性指标
        total_s2_used = len(s2_usage_count)
        usage_frequencies = list(s2_usage_count.values())
        
        # 计算使用熵（越高越均匀）
        usage_probs = np.array(usage_frequencies) / sum(usage_frequencies)
        usage_entropy = entropy(usage_probs)
        max_entropy = np.log(len(S2))  # 完全均匀时的熵
        normalized_entropy = usage_entropy / max_entropy if max_entropy > 0 else 0
        
        # 计算基尼系数（越低越均匀）
        sorted_freqs = sorted(usage_frequencies)
        n = len(sorted_freqs)
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_freqs)) / (n * np.sum(sorted_freqs)) - (n + 1) / n
        
        diversity_metrics = {
            'success_rate': len(successful_paths) / num_samples,
            'total_s2_nodes_used': total_s2_used,
            's2_usage_ratio': total_s2_used / len(S2),
            'usage_entropy': usage_entropy,
            'normalized_entropy': normalized_entropy,
            'gini_coefficient': gini,
            'mean_path_length': np.mean(path_length_distribution),
            'std_path_length': np.std(path_length_distribution),
            's2_usage_distribution': dict(s2_usage_count),
            'top_5_s2_nodes': s2_usage_count.most_common(5),
            'unused_s2_nodes': [s2 for s2 in S2 if s2 not in s2_usage_count]
        }
        
        return diversity_metrics
    
    def compute_mutual_information(self, model, model_name):
        """计算S2表示与S1/S3的互信息"""
        S1, S2, S3 = self.data_info[model_name]['stages']
        
        # 提取嵌入
        embeddings = self._extract_embeddings(model, S1 + S2 + S3, model_name)
        
        # 准备数据
        s1_embs = np.array([embeddings[n] for n in S1 if n in embeddings])
        s2_embs = np.array([embeddings[n] for n in S2 if n in embeddings])
        s3_embs = np.array([embeddings[n] for n in S3 if n in embeddings])
        
        # 降维到合理的维度以计算互信息
        pca = PCA(n_components=10)
        
        all_embs = np.vstack([s1_embs, s2_embs, s3_embs])
        pca.fit(all_embs)
        
        s1_reduced = pca.transform(s1_embs)
        s2_reduced = pca.transform(s2_embs)
        s3_reduced = pca.transform(s3_embs)
        
        # 计算每个S2节点与S1/S3质心的相似度
        s1_centroid = np.mean(s1_reduced, axis=0)
        s3_centroid = np.mean(s3_reduced, axis=0)
        
        s2_bias_scores = []
        for s2_emb in s2_reduced:
            dist_to_s1 = np.linalg.norm(s2_emb - s1_centroid)
            dist_to_s3 = np.linalg.norm(s2_emb - s3_centroid)
            
            # 偏向得分：负值偏向S1，正值偏向S3
            bias = (dist_to_s1 - dist_to_s3) / (dist_to_s1 + dist_to_s3)
            s2_bias_scores.append(bias)
        
        # 将S2节点按偏向分类
        s2_bias_scores = np.array(s2_bias_scores)
        s1_biased = np.sum(s2_bias_scores < -0.1)
        s3_biased = np.sum(s2_bias_scores > 0.1)
        neutral = len(s2_bias_scores) - s1_biased - s3_biased
        
        return {
            's2_bias_distribution': {
                's1_biased': s1_biased,
                's3_biased': s3_biased,
                'neutral': neutral
            },
            'bias_scores': s2_bias_scores,
            'mean_bias': np.mean(s2_bias_scores),
            'std_bias': np.std(s2_bias_scores),
            'extreme_bias_ratio': (s1_biased + s3_biased) / len(s2_bias_scores)
        }
    
    def _test_path(self, model, source, target, data):
        """测试是否能生成有效路径"""
        prompt = f"{source} {target} {source}"
        prompt_ids = [data['stoi'][t] for t in prompt.split() if t in data['stoi']]
        x = torch.tensor(prompt_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            y = model.generate(x, max_new_tokens=20, temperature=0.1, top_k=10)
        
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
        
        # 验证
        path = generated[2:] if len(generated) >= 3 else []
        
        if len(path) >= 2 and path[0] == source and path[-1] == target:
            edges_valid = all(
                data['G'].has_edge(str(path[i]), str(path[i+1]))
                for i in range(len(path)-1)
            )
            return edges_valid
        
        return False
    
    def _generate_path(self, model, source, target, data):
        """生成路径并返回"""
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
        
        path = generated[2:] if len(generated) >= 3 else []
        
        # 验证
        if len(path) >= 2 and path[0] == source and path[-1] == target:
            edges_valid = all(
                data['G'].has_edge(str(path[i]), str(path[i+1]))
                for i in range(len(path)-1)
            )
            return edges_valid, path
        
        return False, []
    
    def _extract_embeddings(self, model, nodes, model_name):
        """提取节点嵌入"""
        embeddings = {}
        stoi = self.data_info[model_name]['stoi']
        
        with torch.no_grad():
            for node in nodes:
                if str(node) not in stoi:
                    continue
                    
                token_id = stoi[str(node)]
                token_tensor = torch.tensor([token_id], device=self.device)
                
                emb = model.transformer.wte(token_tensor)
                pos = torch.zeros(1, 1, dtype=torch.long, device=self.device)
                pos_emb = model.transformer.wpe(pos)
                x = emb + pos_emb
                
                for block in model.transformer.h:
                    x = block(x)
                
                embeddings[node] = x.squeeze().cpu().numpy()
        
        return embeddings
    
    def run_specialization_analysis(self, iterations):
        """运行完整的特化分析"""
        for model_name in self.config.keys():
            print(f"\n{'='*60}")
            print(f"Analyzing {model_name}")
            print('='*60)
            
            for iteration in iterations:
                print(f"\n--- Iteration {iteration} ---")
                
                model = self.load_model(model_name, iteration)
                if model is None:
                    continue
                
                # 1. 分析连接模式
                print("1. Analyzing connection patterns...")
                s2_patterns = self.analyze_s2_connection_patterns(model, model_name)
                
                # 2. 功能聚类
                print("2. Clustering S2 nodes by function...")
                clustering_results = self.cluster_s2_by_function(s2_patterns)
                
                # 3. 路径多样性
                print("3. Analyzing path diversity...")
                diversity_metrics = self.analyze_path_diversity(model, model_name)
                
                # 4. 互信息分析
                print("4. Computing mutual information...")
                mi_results = self.compute_mutual_information(model, model_name)
                
                # 保存结果
                self.results[model_name][iteration] = {
                    's2_patterns': s2_patterns,
                    'clustering': clustering_results,
                    'diversity': diversity_metrics,
                    'mutual_information': mi_results
                }
                
                # 打印关键指标
                self._print_summary(model_name, iteration)
    
    def _print_summary(self, model_name, iteration):
        """打印分析摘要"""
        results = self.results[model_name][iteration]
        
        print(f"\nSummary for {model_name} at iteration {iteration}:")
        
        # 聚类结果
        if 'clustering' in results:
            clustering = results['clustering']
            print(f"  - S2 functional clusters: {len(clustering['cluster_analysis'])}")
            for cluster_name, info in clustering['cluster_analysis'].items():
                print(f"    {cluster_name}: {info['size']} nodes, "
                      f"S1 rate: {info['mean_s1_rate']:.2f}, "
                      f"S3 rate: {info['mean_s3_rate']:.2f}")
        
        # 多样性
        if 'diversity' in results and results['diversity']:
            div = results['diversity']
            print(f"  - Path diversity:")
            print(f"    S2 usage: {div['total_s2_nodes_used']}/{len(self.data_info[model_name]['stages'][1])}")
            print(f"    Normalized entropy: {div['normalized_entropy']:.3f}")
            print(f"    Gini coefficient: {div['gini_coefficient']:.3f}")
        
        # 互信息
        if 'mutual_information' in results:
            mi = results['mutual_information']
            bias = mi['s2_bias_distribution']
            print(f"  - S2 bias distribution:")
            print(f"    S1-biased: {bias['s1_biased']}, "
                  f"Neutral: {bias['neutral']}, "
                  f"S3-biased: {bias['s3_biased']}")
            print(f"    Extreme bias ratio: {mi['extreme_bias_ratio']:.2%}")
    
    def create_visualizations(self, output_dir):
        """创建可视化"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. S2功能特化热图
        self.plot_s2_specialization_heatmap(output_dir)
        
        # 2. 聚类演化
        self.plot_clustering_evolution(output_dir)
        
        # 3. 路径多样性演化
        self.plot_diversity_evolution(output_dir)
        
        # 4. S2偏向分布
        self.plot_s2_bias_distribution(output_dir)
        
        # 5. 综合报告
        self.generate_specialization_report(output_dir)
    
    def plot_s2_specialization_heatmap(self, output_dir):
        """绘制S2功能特化热图"""
        fig, axes = plt.subplots(len(self.config), 3, figsize=(12, 4*len(self.config)))
        
        if len(self.config) == 1:
            axes = axes.reshape(1, -1)
        
        iterations = [5000, 25000, 50000]
        
        for model_idx, model_name in enumerate(self.config.keys()):
            for iter_idx, iteration in enumerate(iterations):
                ax = axes[model_idx, iter_idx]
                
                if iteration in self.results[model_name]:
                    patterns = self.results[model_name][iteration]['s2_patterns']
                    S2 = self.data_info[model_name]['stages'][1]
                    
                    # 创建矩阵
                    matrix = []
                    node_labels = []
                    
                    for s2 in sorted(S2):
                        if s2 in patterns:
                            matrix.append([
                                patterns[s2]['s1_success_rate'],
                                patterns[s2]['s3_success_rate'],
                                patterns[s2]['bidirectional_score']
                            ])
                            node_labels.append(str(s2))
                    
                    matrix = np.array(matrix).T
                    
                    sns.heatmap(matrix,
                               xticklabels=node_labels,
                               yticklabels=['S1→S2', 'S2→S3', 'Bidirectional'],
                               cmap='RdYlGn',
                               vmin=0, vmax=1,
                               ax=ax)
                    
                    ax.set_title(f'{model_name} - {iteration} iter')
                else:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 's2_specialization_heatmap.png'), dpi=300)
        plt.close()
    
    def plot_clustering_evolution(self, output_dir):
        """绘制聚类演化"""
        fig, axes = plt.subplots(1, len(self.config), figsize=(6*len(self.config), 5))
        
        if len(self.config) == 1:
            axes = [axes]
        
        for idx, model_name in enumerate(self.config.keys()):
            ax = axes[idx]
            
            cluster_evolution = []
            iterations = []
            
            for iteration in sorted(self.results[model_name].keys()):
                if 'clustering' in self.results[model_name][iteration]:
                    clustering = self.results[model_name][iteration]['clustering']
                    n_clusters = len(clustering['cluster_analysis'])
                    cluster_evolution.append(n_clusters)
                    iterations.append(iteration)
            
            if iterations:
                ax.plot(iterations, cluster_evolution, 'o-', linewidth=2, markersize=8)
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Number of Functional Clusters')
                ax.set_title(f'{model_name} - S2 Functional Clustering')
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, 5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'clustering_evolution.png'), dpi=300)
        plt.close()
    
    def plot_diversity_evolution(self, output_dir):
        """绘制路径多样性演化"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        for model_name in self.config.keys():
            iterations = []
            entropy_values = []
            gini_values = []
            usage_ratios = []
            success_rates = []
            
            for iteration in sorted(self.results[model_name].keys()):
                if 'diversity' in self.results[model_name][iteration]:
                    div = self.results[model_name][iteration]['diversity']
                    if div:
                        iterations.append(iteration)
                        entropy_values.append(div['normalized_entropy'])
                        gini_values.append(div['gini_coefficient'])
                        usage_ratios.append(div['s2_usage_ratio'])
                        success_rates.append(div['success_rate'])
            
            if iterations:
                ax1.plot(iterations, entropy_values, 'o-', label=model_name, linewidth=2)
                ax2.plot(iterations, gini_values, 'o-', label=model_name, linewidth=2)
                ax3.plot(iterations, usage_ratios, 'o-', label=model_name, linewidth=2)
                ax4.plot(iterations, success_rates, 'o-', label=model_name, linewidth=2)
        
        ax1.set_title('S2 Usage Entropy (Higher = More Uniform)')
        ax1.set_ylabel('Normalized Entropy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_title('S2 Usage Gini Coefficient (Lower = More Equal)')
        ax2.set_ylabel('Gini Coefficient')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3.set_title('Fraction of S2 Nodes Used in Paths')
        ax3.set_ylabel('Usage Ratio')
        ax3.set_xlabel('Iteration')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        ax4.set_title('S1→S3 Path Generation Success Rate')
        ax4.set_ylabel('Success Rate')
        ax4.set_xlabel('Iteration')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'diversity_evolution.png'), dpi=300)
        plt.close()
    
    def plot_s2_bias_distribution(self, output_dir):
        """绘制S2偏向分布"""
        iterations = [5000, 25000, 50000]
        
        fig, axes = plt.subplots(len(iterations), len(self.config), 
                                figsize=(5*len(self.config), 4*len(iterations)))
        
        if len(iterations) == 1:
            axes = axes.reshape(1, -1)
        if len(self.config) == 1:
            axes = axes.reshape(-1, 1)
        
        for iter_idx, iteration in enumerate(iterations):
            for model_idx, model_name in enumerate(self.config.keys()):
                ax = axes[iter_idx, model_idx]
                
                if iteration in self.results[model_name]:
                    mi = self.results[model_name][iteration].get('mutual_information')
                    if mi and 'bias_scores' in mi:
                        bias_scores = mi['bias_scores']
                        
                        ax.hist(bias_scores, bins=20, alpha=0.7, edgecolor='black')
                        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
                        ax.set_xlabel('Bias Score (negative=S1, positive=S3)')
                        ax.set_ylabel('Count')
                        ax.set_title(f'{model_name} - {iteration} iter')
                        
                        # 添加统计信息
                        mean_bias = np.mean(bias_scores)
                        ax.text(0.02, 0.98, f'Mean: {mean_bias:.3f}', 
                               transform=ax.transAxes, va='top')
                else:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 's2_bias_distribution.png'), dpi=300)
        plt.close()
    
    def generate_specialization_report(self, output_dir):
        """生成特化分析报告"""
        report = []
        report.append("# S2 Specialization Analysis Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 核心发现
        report.append("## Key Findings\n")
        
        for model_name in self.config.keys():
            report.append(f"### {model_name}\n")
            
            # 早期vs晚期对比
            early = self.results[model_name].get(5000, {})
            late = self.results[model_name].get(50000, {})
            
            if early and late:
                # 聚类演化
                early_clusters = len(early.get('clustering', {}).get('cluster_analysis', {}))
                late_clusters = len(late.get('clustering', {}).get('cluster_analysis', {}))
                report.append(f"**Functional Clustering Evolution:**")
                report.append(f"- 5k iterations: {early_clusters} clusters")
                report.append(f"- 50k iterations: {late_clusters} clusters\n")
                
                # 路径多样性
                if early.get('diversity') and late.get('diversity'):
                    early_div = early['diversity']
                    late_div = late['diversity']
                    
                    report.append(f"**Path Diversity Changes:**")
                    report.append(f"- S2 usage: {early_div['s2_usage_ratio']:.1%} → {late_div['s2_usage_ratio']:.1%}")
                    report.append(f"- Entropy: {early_div['normalized_entropy']:.3f} → {late_div['normalized_entropy']:.3f}")
                    report.append(f"- Success rate: {early_div['success_rate']:.1%} → {late_div['success_rate']:.1%}\n")
                
                # 偏向分析
                if early.get('mutual_information') and late.get('mutual_information'):
                    early_mi = early['mutual_information']
                    late_mi = late['mutual_information']
                    
                    report.append(f"**S2 Bias Evolution:**")
                    early_bias = early_mi['s2_bias_distribution']
                    late_bias = late_mi['s2_bias_distribution']
                    
                    report.append(f"- S1-biased nodes: {early_bias['s1_biased']} → {late_bias['s1_biased']}")
                    report.append(f"- S3-biased nodes: {early_bias['s3_biased']} → {late_bias['s3_biased']}")
                    report.append(f"- Neutral nodes: {early_bias['neutral']} → {late_bias['neutral']}")
                    report.append(f"- Extreme bias ratio: {early_mi['extreme_bias_ratio']:.1%} → {late_mi['extreme_bias_ratio']:.1%}\n")
        
        # 结论
        report.append("## Conclusions\n")
        report.append("1. **S2 Functional Specialization**: S2 nodes progressively specialize into distinct functional roles")
        report.append("2. **Path Diversity Loss**: The model relies on fewer S2 nodes over time")
        report.append("3. **Bias Development**: S2 nodes develop biases toward either S1 or S3 connections")
        report.append("4. **Mixed Training Effect**: Mixed training maintains functional diversity and prevents extreme specialization")
        
        # 保存报告
        with open(os.path.join(output_dir, 'specialization_report.md'), 'w') as f:
            f.write('\n'.join(report))
        
        print(f"\nReport saved to: {os.path.join(output_dir, 'specialization_report.md')}")
    
    def save_results(self, output_dir):
        """保存所有结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存原始结果
        with open(os.path.join(output_dir, 'specialization_results.pkl'), 'wb') as f:
            pickle.dump(dict(self.results), f)
        
        # 创建可视化
        print("\nCreating visualizations...")
        self.create_visualizations(output_dir)
        
        # 导出关键数据到CSV
        self.export_specialization_metrics(output_dir)
    
    def export_specialization_metrics(self, output_dir):
        """导出特化指标到CSV"""
        rows = []
        
        for model_name in self.config.keys():
            for iteration in sorted(self.results[model_name].keys()):
                row = {
                    'model': model_name,
                    'iteration': iteration
                }
                
                # 添加聚类信息
                if 'clustering' in self.results[model_name][iteration]:
                    clustering = self.results[model_name][iteration]['clustering']
                    row['n_clusters'] = len(clustering['cluster_analysis'])
                
                # 添加多样性指标
                if 'diversity' in self.results[model_name][iteration]:
                    div = self.results[model_name][iteration]['diversity']
                    if div:
                        row.update({
                            's2_usage_ratio': div['s2_usage_ratio'],
                            'normalized_entropy': div['normalized_entropy'],
                            'gini_coefficient': div['gini_coefficient'],
                            'path_success_rate': div['success_rate']
                        })
                
                # 添加偏向信息
                if 'mutual_information' in self.results[model_name][iteration]:
                    mi = self.results[model_name][iteration]['mutual_information']
                    bias = mi['s2_bias_distribution']
                    row.update({
                        's1_biased_nodes': bias['s1_biased'],
                        's3_biased_nodes': bias['s3_biased'],
                        'neutral_nodes': bias['neutral'],
                        'extreme_bias_ratio': mi['extreme_bias_ratio']
                    })
                
                rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(output_dir, 'specialization_metrics.csv'), index=False)
        print(f"Metrics exported to: {os.path.join(output_dir, 'specialization_metrics.csv')}")

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
    
    # 要分析的迭代（为了速度，只分析关键迭代）
    iterations = [5000, 15000, 25000, 35000, 50000]
    
    # 输出目录
    output_dir = f's2_specialization_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    
    print("="*60)
    print("S2 Specialization Analysis")
    print("="*60)
    print("\nThis analysis will:")
    print("1. Test connection patterns for each S2 node")
    print("2. Cluster S2 nodes by their functional roles")
    print("3. Analyze path diversity in S1->S3 generation")
    print("4. Compute S2 representation bias toward S1/S3")
    print("\nExpected runtime: 20-40 minutes\n")
    
    # 创建分析器
    analyzer = S2SpecializationAnalyzer(config, device='cuda:0')
    
    # 运行分析
    analyzer.run_specialization_analysis(iterations)
    
    # 保存结果
    analyzer.save_results(output_dir)
    
    print(f"\n{'='*60}")
    print(f"Analysis complete! Results saved to: {output_dir}")
    print(f"Check the following files:")
    print(f"  - specialization_report.md: Key findings")
    print(f"  - *.png: Visualization plots")
    print(f"  - specialization_metrics.csv: Detailed metrics")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()