# compute_brc_evolution_fixed.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, cdist
import pickle
from model import GPTConfig, GPT
from tabulate import tabulate

def load_model_from_checkpoint(checkpoint_path, device='cuda'):
    """从checkpoint加载模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_args = checkpoint['model_args']
    
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    
    return model

def get_deep_node_embeddings_in_context(model, nodes, stages, device='cuda'):
    """获取节点在典型路径中的深层嵌入"""
    S1, S2, S3 = stages
    embeddings = []
    
    with torch.no_grad():
        for node in nodes:
            # 收集包含该节点的多个路径上下文
            contexts = []
            
            # 如果是S2节点，构造S1->S2->S3路径
            if node in S2:
                # 找几个代表性路径
                for s1 in S1[:3]:  # 取前3个S1节点
                    for s3 in S3[:3]:  # 取前3个S3节点
                        # 路径：s1 -> node -> s3，加上+2偏移
                        path = torch.tensor([s1+2, node+2, s3+2], dtype=torch.long, device=device).unsqueeze(0)
                        
                        # 前向传播
                        tok_emb = model.transformer.wte(path)
                        pos_emb = model.transformer.wpe(torch.arange(path.size(1), device=device))
                        x = model.transformer.drop(tok_emb + pos_emb)
                        
                        for block in model.transformer.h:
                            x = block(x)
                        
                        x = model.transformer.ln_f(x)
                        
                        # 取node在位置1的嵌入（S1在位置0，S2在位置1）
                        node_emb = x[0, 1, :].cpu().numpy()
                        contexts.append(node_emb)
            
            else:
                # 对于S1/S3节点，使用单token输入（加偏移）
                x = torch.tensor([node+2], dtype=torch.long, device=device).unsqueeze(0)
                
                tok_emb = model.transformer.wte(x)
                pos_emb = model.transformer.wpe(torch.zeros(1, dtype=torch.long, device=device))
                x = model.transformer.drop(tok_emb + pos_emb)
                
                for block in model.transformer.h:
                    x = block(x)
                
                x = model.transformer.ln_f(x)
                node_emb = x[0, 0, :].cpu().numpy()
                contexts = [node_emb]
            
            # 平均多个上下文的嵌入
            avg_emb = np.mean(contexts, axis=0)
            embeddings.append(avg_emb)
    
    return np.array(embeddings)

def compute_cosine_bias(node_emb, S1_centroid, S3_centroid):
    """使用余弦相似度计算偏向（与之前的图一致）"""
    # 归一化
    node_norm = node_emb / (np.linalg.norm(node_emb) + 1e-9)
    S1_norm = S1_centroid / (np.linalg.norm(S1_centroid) + 1e-9)
    S3_norm = S3_centroid / (np.linalg.norm(S3_centroid) + 1e-9)
    
    # 余弦相似度
    cos_to_S1 = np.dot(node_norm, S1_norm)
    cos_to_S3 = np.dot(node_norm, S3_norm)
    
    # 偏向值：(cos_S3 - cos_S1 + 1) / 2，范围[0,1]
    bias = (cos_to_S3 - cos_to_S1 + 1) / 2
    
    return bias

def compute_brc_metrics_fixed(model, stages, device='cuda'):
    """修正后的BRC计算"""
    S1, S2, S3 = stages
    
    # 获取嵌入（考虑上下文）
    S1_embeddings = get_deep_node_embeddings_in_context(model, S1, stages, device)
    S2_embeddings = get_deep_node_embeddings_in_context(model, S2, stages, device)
    S3_embeddings = get_deep_node_embeddings_in_context(model, S3, stages, device)
    
    # 1. 计算余弦偏向
    S1_centroid = S1_embeddings.mean(axis=0)
    S3_centroid = S3_embeddings.mean(axis=0)
    
    biases = []
    for emb in S2_embeddings:
        bias = compute_cosine_bias(emb, S1_centroid, S3_centroid)
        biases.append(bias)
    
    biases = np.array(biases)
    sigma_bias = np.std(biases)
    mean_bias = np.mean(biases)
    
    # 2. 正确计算有效维度（先中心化）
    S2_centered = S2_embeddings - S2_embeddings.mean(axis=0)
    
    # SVD计算
    U, S, Vt = np.linalg.svd(S2_centered, full_matrices=False)
    eigenvalues = S**2 / (len(S2_embeddings) - 1)
    
    # 累积方差
    total_var = eigenvalues.sum()
    cumsum_var = np.cumsum(eigenvalues) / total_var
    ED_90 = np.searchsorted(cumsum_var, 0.9) + 1
    
    # 3. 参与率
    PR = (eigenvalues.sum())**2 / (eigenvalues**2).sum()
    
    # 4. 方向多样性（新指标）
    S2_normalized = S2_embeddings / (np.linalg.norm(S2_embeddings, axis=1, keepdims=True) + 1e-9)
    cos_sim_matrix = np.dot(S2_normalized, S2_normalized.T)
    upper_tri_indices = np.triu_indices_from(cos_sim_matrix, k=1)
    avg_cos_sim = np.mean(cos_sim_matrix[upper_tri_indices])
    direction_diversity = 1 - avg_cos_sim
    
    # 5. S2内部距离
    if len(S2) > 1:
        pairwise_dists = pdist(S2_embeddings)
        dist_std = np.std(pairwise_dists)
        dist_mean = np.mean(pairwise_dists)
    else:
        dist_std = 0
        dist_mean = 0
    
    # 6. BRC指数（修正版）
    BRC = sigma_bias / ED_90 if ED_90 > 0 else 0
    
    # 7. 替代指标
    BRC_alt = (1 - direction_diversity) * mean_bias  # 方向趋同 × 偏向强度
    
    return {
        'sigma_bias': sigma_bias,
        'mean_bias': mean_bias,
        'ED_90': ED_90,
        'PR': PR,
        'direction_diversity': direction_diversity,
        'avg_cos_sim': avg_cos_sim,
        'dist_std': dist_std,
        'dist_mean': dist_mean,
        'BRC': BRC,
        'BRC_alt': BRC_alt,
        'biases': biases
    }

def analyze_all_checkpoints():
    """分析所有模型的BRC演化"""
    
    # 加载阶段信息
    with open('data/simple_graph/composition_90/stage_info.pkl', 'rb') as f:
        stage_info = pickle.load(f)
    stages = stage_info['stages']
    
    print(f"Stages: S1={len(stages[0])}, S2={len(stages[1])}, S3={len(stages[2])}")
    
    models = {
        'original': 'out/composition_20250702_063926',
        '5% mixed': 'out/composition_20250703_004537', 
        '10% mixed': 'out/composition_20250703_011304'
    }
    
    # 手动添加成功率数据
    success_rates_manual = {
        'original': [0.70, 0.65, 0.55, 0.45, 0.40, 0.35, 0.33, 0.32, 0.32, 0.32],
        '5% mixed': [0.92, 0.85, 0.70, 0.62, 0.75, 0.80, 0.83, 0.85, 0.86, 0.86],
        '10% mixed': [0.92, 0.90, 0.90, 0.89, 0.89, 0.89, 0.90, 0.90, 0.90, 0.90]
    }
    
    checkpoints = [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000]
    
    table_data = []
    results = {}
    
    for name, out_dir in models.items():
        print(f"\n{'='*80}")
        print(f"Analyzing {name}")
        print('='*80)
        
        model_results = {
            'iters': [], 'BRC': [], 'ED_90': [], 'sigma_bias': [], 
            'mean_bias': [], 'direction_diversity': [], 'BRC_alt': [],
            'success_rate': [], 'dist_std': []
        }
        
        for idx, ckpt in enumerate(checkpoints):
            ckpt_path = f"{out_dir}/ckpt_{ckpt}.pt"
            if not os.path.exists(ckpt_path):
                continue
            
            print(f"\n  Processing iteration {ckpt}...")
            
            try:
                # 加载模型
                model = load_model_from_checkpoint(ckpt_path)
                
                # 计算BRC指标
                metrics = compute_brc_metrics_fixed(model, stages)
                
                # 获取成功率
                success_rate = success_rates_manual[name][idx]
                
                # 保存结果
                model_results['iters'].append(ckpt)
                model_results['BRC'].append(metrics['BRC'])
                model_results['ED_90'].append(metrics['ED_90'])
                model_results['sigma_bias'].append(metrics['sigma_bias'])
                model_results['mean_bias'].append(metrics['mean_bias'])
                model_results['direction_diversity'].append(metrics['direction_diversity'])
                model_results['BRC_alt'].append(metrics['BRC_alt'])
                model_results['success_rate'].append(success_rate)
                model_results['dist_std'].append(metrics['dist_std'])
                
                # 添加到表格
                table_data.append([
                    name, 
                    f"{ckpt/1000:.0f}k",
                    f"{metrics['BRC']:.3f}",
                    f"{metrics['ED_90']:.0f}",
                    f"{metrics['sigma_bias']:.3f}",
                    f"{metrics['mean_bias']:.3f}",
                    f"{metrics['direction_diversity']:.3f}",
                    f"{success_rate:.2f}"
                ])
                
                print(f"    BRC={metrics['BRC']:.3f}, ED={metrics['ED_90']}, "
                      f"σ={metrics['sigma_bias']:.3f}, diversity={metrics['direction_diversity']:.3f}")
                
                # 清理内存
                del model
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"    Error: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        results[name] = model_results
    
    # 打印表格
    print("\n" + "="*100)
    print("BRC Analysis Summary (Fixed)")
    print("="*100)
    
    headers = ["Model", "Iter", "BRC", "ED_90", "σ_bias", "Mean_bias", "Dir_Div", "Success"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # 保存结果
    with open('brc_analysis_results_fixed.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    return results

def plot_brc_evolution_with_success(results):
    """绘制修正后的BRC分析（包含成功率）"""
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    colors = {'original': '#e74c3c', '5% mixed': '#f39c12', '10% mixed': '#27ae60'}
    
    # 1. BRC演化 vs 成功率
    ax1 = axes[0, 0]
    ax1_twin = ax1.twinx()
    
    for name in ['original', '5% mixed', '10% mixed']:
        if name in results and results[name]['iters']:
            iters_k = [x/1000 for x in results[name]['iters']]
            
            # BRC (左轴)
            ax1.plot(iters_k, results[name]['BRC'], 
                    marker='o', label=f'{name} BRC', linewidth=2.5, 
                    color=colors[name], linestyle='-')
            
            # 成功率 (右轴)
            ax1_twin.plot(iters_k, results[name]['success_rate'],
                         marker='s', label=f'{name} Success', linewidth=2,
                         color=colors[name], linestyle='--', alpha=0.7)
    
    ax1.set_xlabel('Iteration (k)', fontsize=12)
    ax1.set_ylabel('BRC Index', fontsize=12)
    ax1_twin.set_ylabel('Success Rate', fontsize=12)
    ax1.set_title('BRC vs Success Rate Evolution', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 2. 有效维度
    ax2 = axes[0, 1]
    for name in ['original', '5% mixed', '10% mixed']:
        if name in results and results[name]['iters']:
            iters_k = [x/1000 for x in results[name]['iters']]
            ax2.plot(iters_k, results[name]['ED_90'],
                    marker='s', label=name, linewidth=2.5,
                    color=colors[name])
    ax2.set_xlabel('Iteration (k)', fontsize=12)
    ax2.set_ylabel('Effective Dimension (90%)', fontsize=12)
    ax2.set_title('S2 Effective Dimension', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 方向多样性
    ax3 = axes[0, 2]
    for name in ['original', '5% mixed', '10% mixed']:
        if name in results and results[name]['iters']:
            iters_k = [x/1000 for x in results[name]['iters']]
            ax3.plot(iters_k, results[name]['direction_diversity'],
                    marker='^', label=name, linewidth=2.5,
                    color=colors[name])
    ax3.set_xlabel('Iteration (k)', fontsize=12)
    ax3.set_ylabel('Direction Diversity', fontsize=12)
    ax3.set_title('S2 Direction Diversity', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. BRC vs 成功率散点图
    ax4 = axes[1, 0]
    for name in ['original', '5% mixed', '10% mixed']:
        if name in results and results[name]['BRC']:
            ax4.scatter(results[name]['BRC'], results[name]['success_rate'],
                       label=name, color=colors[name], s=100, alpha=0.7)
    ax4.set_xlabel('BRC Index', fontsize=12)
    ax4.set_ylabel('Success Rate', fontsize=12)
    ax4.set_title('BRC vs Success Rate Correlation', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Bias标准差
    ax5 = axes[1, 1]
    for name in ['original', '5% mixed', '10% mixed']:
        if name in results and results[name]['iters']:
            iters_k = [x/1000 for x in results[name]['iters']]
            ax5.plot(iters_k, results[name]['sigma_bias'],
                    marker='D', label=name, linewidth=2.5,
                    color=colors[name])
    ax5.set_xlabel('Iteration (k)', fontsize=12)
    ax5.set_ylabel('Bias Std Dev (σ)', fontsize=12)
    ax5.set_title('S2 Bias Distribution Width', fontsize=14, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 最终指标对比
    ax6 = axes[1, 2]
    model_names = []
    final_brc = []
    final_success = []
    
    for name in ['original', '5% mixed', '10% mixed']:
        if name in results and results[name]['BRC']:
            model_names.append(name)
            final_brc.append(results[name]['BRC'][-1])
            final_success.append(results[name]['success_rate'][-1])
    
    x = np.arange(len(model_names))
    width = 0.35
    
    bars1 = ax6.bar(x - width/2, final_brc, width, label='BRC', color='#3498db')
    bars2 = ax6.bar(x + width/2, final_success, width, label='Success Rate', color='#e74c3c')
    
    ax6.set_xlabel('Model', fontsize=12)
    ax6.set_ylabel('Value', fontsize=12)
    ax6.set_title('Final Metrics (iter=50k)', fontsize=14, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels(model_names)
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('brc_analysis_fixed.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主函数"""
    print("="*80)
    print("BRC Evolution Analysis (Fixed Version)")
    print("="*80)
    
    # 分析
    results = analyze_all_checkpoints()
    
    # 绘图
    plot_brc_evolution_with_success(results)
    
    # 相关性分析
    print("\n" + "="*60)
    print("Correlation Analysis:")
    print("="*60)
    
    all_brc = []
    all_success = []
    
    for name in results:
        if 'BRC' in results[name] and 'success_rate' in results[name]:
            all_brc.extend(results[name]['BRC'])
            all_success.extend(results[name]['success_rate'])
    
    if all_brc and all_success:
        from scipy.stats import pearsonr, spearmanr
        pearson_r, pearson_p = pearsonr(all_brc, all_success)
        spearman_r, spearman_p = spearmanr(all_brc, all_success)
        
        print(f"BRC vs Success Rate:")
        print(f"  Pearson r = {pearson_r:.3f} (p = {pearson_p:.3e})")
        print(f"  Spearman r = {spearman_r:.3f} (p = {spearman_p:.3e})")

if __name__ == "__main__":
    main()