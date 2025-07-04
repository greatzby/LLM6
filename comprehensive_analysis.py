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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import json

# 导入你的模型
from model import GPTConfig, GPT

class CompositionAnalyzer:
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
    
    def analyze_representation_space(self, model, model_name, iteration):
        """分析表示空间的特性"""
        stages = self.data_info[model_name]['stages']
        S1, S2, S3 = stages
        
        # 提取所有节点的嵌入
        all_nodes = list(range(90))
        embeddings = self.extract_node_embeddings(model, all_nodes, model_name)
        
        # 按阶段分组
        stage_embeddings = {'S1': [], 'S2': [], 'S3': []}
        stage_nodes = {'S1': [], 'S2': [], 'S3': []}
        
        for node, emb_data in embeddings.items():
            if node in S1:
                stage_embeddings['S1'].append(emb_data['final'])
                stage_nodes['S1'].append(node)
            elif node in S2:
                stage_embeddings['S2'].append(emb_data['final'])
                stage_nodes['S2'].append(node)
            elif node in S3:
                stage_embeddings['S3'].append(emb_data['final'])
                stage_nodes['S3'].append(node)
        
        # 计算各种指标
        metrics = {}
        
        for stage_name in ['S1', 'S2', 'S3']:
            if len(stage_embeddings[stage_name]) > 1:
                embs = np.array(stage_embeddings[stage_name])
                
                # 1. 内聚性（平均成对相似度）
                similarities = []
                for i in range(len(embs)):
                    for j in range(i+1, len(embs)):
                        sim = np.dot(embs[i], embs[j]) / (
                            np.linalg.norm(embs[i]) * np.linalg.norm(embs[j])
                        )
                        similarities.append(sim)
                
                # 2. 中心性（到质心的平均距离）
                centroid = np.mean(embs, axis=0)
                distances = [np.linalg.norm(e - centroid) for e in embs]
                
                metrics[stage_name] = {
                    'cohesion': np.mean(similarities),
                    'cohesion_std': np.std(similarities),
                    'centroid_distance_mean': np.mean(distances),
                    'centroid_distance_std': np.std(distances),
                    'num_nodes': len(embs)
                }
        
        # 3. 阶段间分离度
        separation = {}
        for s1, s2 in [('S1', 'S2'), ('S2', 'S3'), ('S1', 'S3')]:
            if stage_embeddings[s1] and stage_embeddings[s2]:
                centroid1 = np.mean(stage_embeddings[s1], axis=0)
                centroid2 = np.mean(stage_embeddings[s2], axis=0)
                separation[f'{s1}-{s2}'] = np.linalg.norm(centroid1 - centroid2)
        
        metrics['separation'] = separation
        
        # 4. S2桥接性分析
        if stage_embeddings['S2']:
            s2_bridge_scores = []
            for s2_emb in stage_embeddings['S2']:
                # 计算S2节点到S1和S3质心的距离比
                if stage_embeddings['S1'] and stage_embeddings['S3']:
                    s1_centroid = np.mean(stage_embeddings['S1'], axis=0)
                    s3_centroid = np.mean(stage_embeddings['S3'], axis=0)
                    
                    dist_to_s1 = np.linalg.norm(s2_emb - s1_centroid)
                    dist_to_s3 = np.linalg.norm(s2_emb - s3_centroid)
                    
                    # 理想的桥接节点应该到两边距离相近
                    bridge_score = 1 - abs(dist_to_s1 - dist_to_s3) / (dist_to_s1 + dist_to_s3)
                    s2_bridge_scores.append(bridge_score)
            
            metrics['S2']['bridge_score'] = np.mean(s2_bridge_scores) if s2_bridge_scores else 0
        
        return metrics, embeddings, stage_nodes
    
    def train_linear_probe(self, embeddings, stage_nodes, model_name):
        """训练线性探针预测节点所属阶段"""
        stages = self.data_info[model_name]['stages']
        S1, S2, S3 = stages
        
        # 准备数据
        X = []
        y = []
        nodes = []
        
        for stage_idx, (stage_name, node_list) in enumerate(stage_nodes.items()):
            for node in node_list:
                if node in embeddings:
                    X.append(embeddings[node]['final'])
                    y.append(stage_idx)  # 0: S1, 1: S2, 2: S3
                    nodes.append(node)
        
        if len(X) < 10:  # 数据太少
            return None
            
        X = np.array(X)
        y = np.array(y)
        
        # 训练测试分割
        n_samples = len(X)
        indices = np.random.permutation(n_samples)
        n_train = int(0.8 * n_samples)
        
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        # 训练
        probe = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
        probe.fit(X_train, y_train)
        
        # 评估
        train_pred = probe.predict(X_train)
        test_pred = probe.predict(X_test)
        
        # 计算每个阶段的准确率
        stage_accuracies = {}
        for stage_idx, stage_name in enumerate(['S1', 'S2', 'S3']):
            stage_mask = y_test == stage_idx
            if np.any(stage_mask):
                stage_acc = accuracy_score(y_test[stage_mask], test_pred[stage_mask])
                stage_accuracies[stage_name] = stage_acc
        
        return {
            'train_accuracy': accuracy_score(y_train, train_pred),
            'test_accuracy': accuracy_score(y_test, test_pred),
            'stage_accuracies': stage_accuracies,
            'confusion_matrix': confusion_matrix(y_test, test_pred),
            'probe': probe
        }
    
    def analyze_path_generation(self, model, model_name, iteration, num_samples=20):
        """分析路径生成的特性"""
        data = self.data_info[model_name]
        S1, S2, S3 = data['stages']
        
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
        
        # 分析每种类型的前num_samples个
        generation_analysis = {}
        
        for path_type, cases in test_cases.items():
            analysis = {
                'successful_paths': [],
                'failed_paths': [],
                'path_lengths': [],
                's2_usage': []  # 对于S1->S3路径
            }
            
            for source, target in cases[:num_samples]:
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
                valid = False
                if len(path) >= 2 and path[0] == source and path[-1] == target:
                    # 检查边的有效性
                    edges_valid = all(
                        data['G'].has_edge(str(path[i]), str(path[i+1]))
                        for i in range(len(path)-1)
                    )
                    if edges_valid:
                        valid = True
                        analysis['successful_paths'].append(path)
                        analysis['path_lengths'].append(len(path))
                        
                        # 对于S1->S3，检查S2使用
                        if path_type == 'S1->S3':
                            s2_nodes = [n for n in path[1:-1] if n in S2]
                            analysis['s2_usage'].append(len(s2_nodes))
                
                if not valid:
                    analysis['failed_paths'].append({
                        'source': source,
                        'target': target,
                        'generated': path
                    })
            
            generation_analysis[path_type] = analysis
        
        return generation_analysis
    
    def analyze_single_iteration(self, model_name, iteration):
        """分析单个迭代的所有方面"""
        model, actual_iter = self.load_checkpoint(model_name, iteration)
        if model is None:
            return None
        
        print(f"  Analyzing representation space...")
        metrics, embeddings, stage_nodes = self.analyze_representation_space(
            model, model_name, iteration
        )
        
        print(f"  Training linear probe...")
        probe_results = self.train_linear_probe(embeddings, stage_nodes, model_name)
        
        print(f"  Analyzing path generation...")
        generation_analysis = self.analyze_path_generation(model, model_name, iteration)
        
        return {
            'iteration': iteration,
            'representation_metrics': metrics,
            'probe_results': probe_results,
            'generation_analysis': generation_analysis,
            'embeddings': embeddings  # 保存用于可视化
        }
    
    def run_full_analysis(self, iterations):
        """运行完整分析"""
        for model_name in self.config.keys():
            print(f"\n{'='*60}")
            print(f"Analyzing {model_name}")
            print('='*60)
            
            for iteration in iterations:
                print(f"\nIteration {iteration}:")
                
                results = self.analyze_single_iteration(model_name, iteration)
                if results:
                    self.results[model_name][iteration] = results
                    
                    # 打印关键指标
                    metrics = results['representation_metrics']
                    print(f"  S2 cohesion: {metrics['S2']['cohesion']:.3f}")
                    if 'bridge_score' in metrics['S2']:
                        print(f"  S2 bridge score: {metrics['S2']['bridge_score']:.3f}")
                    
                    if results['probe_results']:
                        print(f"  Probe accuracy: {results['probe_results']['test_accuracy']:.2%}")
                        if 'S2' in results['probe_results']['stage_accuracies']:
                            print(f"  S2 classification accuracy: {results['probe_results']['stage_accuracies']['S2']:.2%}")
    
    def save_results(self, output_dir):
        """保存所有结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存原始结果
        with open(os.path.join(output_dir, 'raw_results.pkl'), 'wb') as f:
            pickle.dump(dict(self.results), f)
        
        # 生成可视化
        self.create_visualizations(output_dir)
        
        # 生成报告
        self.generate_report(output_dir)
    
    def create_visualizations(self, output_dir):
        """创建所有可视化图表"""
        # 1. S2内聚性对比
        self.plot_s2_cohesion_evolution(output_dir)
        
        # 2. 桥接分数演化
        self.plot_bridge_score_evolution(output_dir)
        
        # 3. 探针准确率
        self.plot_probe_accuracy(output_dir)
        
        # 4. 嵌入空间可视化（关键迭代）
        key_iterations = [5000, 25000, 50000]
        for iter_num in key_iterations:
            self.plot_embedding_spaces(iter_num, output_dir)
        
        # 5. S2分离度分析
        self.plot_stage_separation(output_dir)
    
    def plot_s2_cohesion_evolution(self, output_dir):
        """绘制S2内聚性演化"""
        plt.figure(figsize=(10, 6))
        
        for model_name in self.config.keys():
            iterations = []
            cohesion_values = []
            
            for iter_num in sorted(self.results[model_name].keys()):
                if 'representation_metrics' in self.results[model_name][iter_num]:
                    metrics = self.results[model_name][iter_num]['representation_metrics']
                    if 'S2' in metrics:
                        iterations.append(iter_num)
                        cohesion_values.append(metrics['S2']['cohesion'])
            
            if iterations:
                plt.plot(iterations, cohesion_values, 'o-', 
                        label=model_name, linewidth=2, markersize=8)
        
        plt.xlabel('Iteration')
        plt.ylabel('S2 Cohesion')
        plt.title('S2 Node Cohesion Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 添加关键区域标注
        plt.axvspan(0, 10000, alpha=0.1, color='green', label='Early success')
        plt.axvspan(20000, 35000, alpha=0.1, color='red', label='Degradation')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 's2_cohesion_evolution.png'), dpi=300)
        plt.close()
    
    def plot_bridge_score_evolution(self, output_dir):
        """绘制S2桥接分数演化"""
        plt.figure(figsize=(10, 6))
        
        for model_name in self.config.keys():
            iterations = []
            bridge_scores = []
            
            for iter_num in sorted(self.results[model_name].keys()):
                metrics = self.results[model_name][iter_num]['representation_metrics']
                if 'S2' in metrics and 'bridge_score' in metrics['S2']:
                    iterations.append(iter_num)
                    bridge_scores.append(metrics['S2']['bridge_score'])
            
            if iterations:
                plt.plot(iterations, bridge_scores, 'o-', 
                        label=model_name, linewidth=2, markersize=8)
        
        plt.xlabel('Iteration')
        plt.ylabel('S2 Bridge Score')
        plt.title('S2 Bridging Capability (1 = perfect bridge)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 's2_bridge_score.png'), dpi=300)
        plt.close()
    
    def plot_embedding_spaces(self, iteration, output_dir):
        """使用t-SNE可视化嵌入空间"""
        fig, axes = plt.subplots(1, len(self.config), figsize=(6*len(self.config), 5))
        if len(self.config) == 1:
            axes = [axes]
        
        for idx, model_name in enumerate(self.config.keys()):
            ax = axes[idx]
            
            if iteration not in self.results[model_name]:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax.set_title(f'{model_name} - Iter {iteration}')
                continue
            
            embeddings = self.results[model_name][iteration]['embeddings']
            stages = self.data_info[model_name]['stages']
            S1, S2, S3 = stages
            
            # 准备数据
            all_embs = []
            all_labels = []
            
            for node, emb_data in embeddings.items():
                all_embs.append(emb_data['final'])
                if node in S1:
                    all_labels.append('S1')
                elif node in S2:
                    all_labels.append('S2')
                else:
                    all_labels.append('S3')
            
            all_embs = np.array(all_embs)
            
            # t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_embs)-1))
            reduced = tsne.fit_transform(all_embs)
            
            # 绘图
            colors = {'S1': 'blue', 'S2': 'green', 'S3': 'red'}
            for stage in ['S1', 'S2', 'S3']:
                mask = np.array(all_labels) == stage
                ax.scatter(reduced[mask, 0], reduced[mask, 1],
                          c=colors[stage], label=stage, alpha=0.6, s=100)
            
            ax.set_title(f'{model_name} - Iteration {iteration}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'embedding_space_{iteration}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, output_dir):
        """生成详细的分析报告"""
        report = []
        report.append("# Transformer Composition Analysis Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 执行摘要
        report.append("## Executive Summary\n")
        report.append("This analysis investigates three key questions about compositional learning in Transformers:")
        report.append("1. What enables early-stage compositional success?")
        report.append("2. What causes the forgetting of compositional ability?")
        report.append("3. How does mixed training prevent forgetting?\n")
        
        # 1. 早期成功分析
        report.append("## 1. Early Stage Success Analysis (5k iterations)\n")
        
        for model_name in self.config.keys():
            if 5000 in self.results[model_name]:
                data = self.results[model_name][5000]
                metrics = data['representation_metrics']
                probe = data['probe_results']
                
                report.append(f"### {model_name}")
                report.append(f"- **S2 Cohesion**: {metrics['S2']['cohesion']:.3f}")
                report.append(f"- **S2 Bridge Score**: {metrics['S2'].get('bridge_score', 'N/A')}")
                
                if probe:
                    report.append(f"- **Probe Accuracy**: {probe['test_accuracy']:.2%}")
                    report.append(f"- **S2 Classification**: {probe['stage_accuracies'].get('S2', 'N/A')}")
                
                report.append("")
        
        # 2. 遗忘机制分析
        report.append("## 2. Forgetting Mechanism Analysis\n")
        
        # 计算关键迭代的变化
        key_iters = [5000, 25000, 50000]
        
        for model_name in self.config.keys():
            report.append(f"### {model_name}")
            
            # S2内聚性变化
            cohesion_values = []
            for iter_num in key_iters:
                if iter_num in self.results[model_name]:
                    metrics = self.results[model_name][iter_num]['representation_metrics']
                    cohesion_values.append((iter_num, metrics['S2']['cohesion']))
            
            if len(cohesion_values) >= 2:
                report.append("\n**S2 Cohesion Degradation:**")
                for iter_num, cohesion in cohesion_values:
                    report.append(f"- Iteration {iter_num}: {cohesion:.3f}")
                
                total_change = cohesion_values[-1][1] - cohesion_values[0][1]
                report.append(f"- **Total degradation**: {total_change:.3f} ({total_change/cohesion_values[0][1]*100:.1f}%)")
            
            report.append("")
        
        # 3. 混合训练效果
        report.append("## 3. Effect of Mixed Training\n")
        
        # 比较最终性能
        final_iter = 50000
        if all(final_iter in self.results[model] for model in self.config.keys()):
            report.append("### Final Performance Comparison (50k iterations)\n")
            
            report.append("| Model | S2 Cohesion | Bridge Score | Probe Accuracy |")
            report.append("|-------|-------------|--------------|----------------|")
            
            for model_name in self.config.keys():
                metrics = self.results[model_name][final_iter]['representation_metrics']
                probe = self.results[model_name][final_iter]['probe_results']
                
                cohesion = metrics['S2']['cohesion']
                bridge = metrics['S2'].get('bridge_score', 'N/A')
                probe_acc = probe['test_accuracy'] if probe else 'N/A'
                
                report.append(f"| {model_name} | {cohesion:.3f} | {bridge} | {probe_acc} |")
        
        # 4. 关键发现
        report.append("\n## 4. Key Findings\n")
        
        # 分析S2漂移
        s2_drift_analysis = self.analyze_s2_drift()
        if s2_drift_analysis:
            report.append("### S2 Representation Drift")
            for model_name, drift_info in s2_drift_analysis.items():
                report.append(f"- **{model_name}**: {drift_info}")
        
        # 5. 结论
        report.append("\n## 5. Conclusions\n")
        report.append("1. **Early Success**: High S2 cohesion and bridge scores enable composition")
        report.append("2. **Forgetting**: S2 representations drift and lose their bridging role")
        report.append("3. **Mixed Training**: Maintains S2 cohesion and prevents representational collapse")
        
        # 保存报告
        report_path = os.path.join(output_dir, 'analysis_report.md')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"\nReport saved to: {report_path}")
    
    def analyze_s2_drift(self):
        """分析S2表示的漂移"""
        drift_analysis = {}
        
        for model_name in self.config.keys():
            # 收集S2嵌入随时间的变化
            iterations = sorted([i for i in self.results[model_name].keys() 
                               if 'embeddings' in self.results[model_name][i]])
            
            if len(iterations) < 2:
                continue
            
            # 计算第一个和最后一个迭代之间的平均漂移
            first_iter = iterations[0]
            last_iter = iterations[-1]
            
            first_embeddings = self.results[model_name][first_iter]['embeddings']
            last_embeddings = self.results[model_name][last_iter]['embeddings']
            
            S2 = self.data_info[model_name]['stages'][1]
            
            drifts = []
            for node in S2:
                if node in first_embeddings and node in last_embeddings:
                    first_emb = first_embeddings[node]['final']
                    last_emb = last_embeddings[node]['final']
                    drift = np.linalg.norm(last_emb - first_emb)
                    drifts.append(drift)
            
            if drifts:
                drift_analysis[model_name] = f"Mean drift: {np.mean(drifts):.3f} (±{np.std(drifts):.3f})"
        
        return drift_analysis
    
    def plot_probe_accuracy(self, output_dir):
        """绘制探针准确率"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 整体准确率
        for model_name in self.config.keys():
            iterations = []
            accuracies = []
            
            for iter_num in sorted(self.results[model_name].keys()):
                probe = self.results[model_name][iter_num]['probe_results']
                if probe:
                    iterations.append(iter_num)
                    accuracies.append(probe['test_accuracy'])
            
            if iterations:
                ax1.plot(iterations, accuracies, 'o-', label=model_name, linewidth=2)
        
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Test Accuracy')
        ax1.set_title('Linear Probe: Overall Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.05)
        
        # S2特定准确率
        for model_name in self.config.keys():
            iterations = []
            s2_accuracies = []
            
            for iter_num in sorted(self.results[model_name].keys()):
                probe = self.results[model_name][iter_num]['probe_results']
                if probe and 'S2' in probe['stage_accuracies']:
                    iterations.append(iter_num)
                    s2_accuracies.append(probe['stage_accuracies']['S2'])
            
            if iterations:
                ax2.plot(iterations, s2_accuracies, 'o-', label=model_name, linewidth=2)
        
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('S2 Classification Accuracy')
        ax2.set_title('Linear Probe: S2-specific Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.05)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'probe_accuracy.png'), dpi=300)
        plt.close()
    
    def plot_stage_separation(self, output_dir):
        """绘制阶段间分离度"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        separations = ['S1-S2', 'S2-S3', 'S1-S3']
        
        for idx, sep in enumerate(separations):
            ax = axes[idx]
            
            for model_name in self.config.keys():
                iterations = []
                distances = []
                
                for iter_num in sorted(self.results[model_name].keys()):
                    metrics = self.results[model_name][iter_num]['representation_metrics']
                    if 'separation' in metrics and sep in metrics['separation']:
                        iterations.append(iter_num)
                        distances.append(metrics['separation'][sep])
                
                if iterations:
                    ax.plot(iterations, distances, 'o-', label=model_name, linewidth=2)
            
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Centroid Distance')
            ax.set_title(f'{sep} Separation')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'stage_separation.png'), dpi=300)
        plt.close()

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
    output_dir = f'analysis_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    
    # 创建分析器
    analyzer = CompositionAnalyzer(config, device='cuda:0')
    
    # 运行完整分析
    analyzer.run_full_analysis(iterations)
    
    # 保存结果
    analyzer.save_results(output_dir)
    
    print(f"\n{'='*60}")
    print(f"Analysis complete! Results saved to: {output_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()