import torch
import numpy as np
import matplotlib.pyplot as plt

def comprehensive_layer_analysis(ckpt_20k_path, ckpt_50k_path):
    """
    全面分析20k和50k checkpoint的所有层
    """
    print("="*80)
    print("🔬 Comprehensive Layer Analysis")
    print("="*80)
    
    # 加载checkpoints
    ckpt_20k = torch.load(ckpt_20k_path, map_location='cpu')
    ckpt_50k = torch.load(ckpt_50k_path, map_location='cpu')
    
    state_20k = ckpt_20k['model'] if 'model' in ckpt_20k else ckpt_20k
    state_50k = ckpt_50k['model'] if 'model' in ckpt_50k else ckpt_50k
    
    # 分析结果存储
    layer_analysis = {}
    
    print("\n📊 Layer-by-Layer Comparison")
    print("-"*80)
    print(f"{'Layer Name':<50} {'Shape':<20} {'Norm_20k':<12} {'Norm_50k':<12} {'Change%':<10}")
    print("-"*80)
    
    # 遍历所有层
    for key in state_20k.keys():
        if key in state_50k:
            w_20k = state_20k[key]
            w_50k = state_50k[key]
            
            # 计算范数
            norm_20k = torch.norm(w_20k).item()
            norm_50k = torch.norm(w_50k).item()
            norm_change = (norm_50k - norm_20k) / norm_20k * 100
            
            # 计算相似度
            w_20k_flat = w_20k.flatten()
            w_50k_flat = w_50k.flatten()
            cosine_sim = torch.nn.functional.cosine_similarity(
                w_20k_flat.unsqueeze(0), 
                w_50k_flat.unsqueeze(0)
            ).item()
            
            # 存储分析结果
            layer_analysis[key] = {
                'shape': tuple(w_20k.shape),
                'norm_20k': norm_20k,
                'norm_50k': norm_50k,
                'norm_change_%': norm_change,
                'cosine_similarity': cosine_sim,
                'has_nan_20k': torch.isnan(w_20k).any().item(),
                'has_inf_20k': torch.isinf(w_20k).any().item(),
                'has_nan_50k': torch.isnan(w_50k).any().item(),
                'has_inf_50k': torch.isinf(w_50k).any().item(),
            }
            
            # 打印结果
            shape_str = str(tuple(w_20k.shape))
            print(f"{key:<50} {shape_str:<20} {norm_20k:<12.4f} {norm_50k:<12.4f} {norm_change:<10.2f}")
            
            # 标记异常
            if abs(norm_change) > 50:
                print(f"  ⚠️ Large change detected! Cosine similarity: {cosine_sim:.4f}")
            if layer_analysis[key]['has_nan_50k'] or layer_analysis[key]['has_inf_50k']:
                print(f"  🔴 NaN or Inf detected in 50k!")
    
    print("-"*80)
    
    # 特别关注的层
    print("\n🎯 Key Layers Analysis")
    print("-"*80)
    
    important_layers = {
        'embedding': [],
        'lm_head': [],
        'attention': [],
        'mlp': [],
        'norm': []
    }
    
    for key in layer_analysis:
        if 'wte' in key or 'wpe' in key:
            important_layers['embedding'].append(key)
        elif 'lm_head' in key or 'head' in key.lower():
            important_layers['lm_head'].append(key)
        elif 'attn' in key:
            important_layers['attention'].append(key)
        elif 'mlp' in key or 'fc' in key:
            important_layers['mlp'].append(key)
        elif 'norm' in key or 'ln' in key:
            important_layers['norm'].append(key)
    
    for category, layers in important_layers.items():
        if layers:
            print(f"\n{category.upper()} Layers:")
            for layer in layers:
                info = layer_analysis[layer]
                print(f"  {layer}: sim={info['cosine_similarity']:.4f}, change={info['norm_change_%']:.2f}%")
    
    return layer_analysis

def analyze_specific_layers(ckpt_20k_path, ckpt_50k_path):
    """
    深入分析特定层（embedding vs lm_head）
    """
    print("\n" + "="*80)
    print("🔬 Embedding vs LM_Head Analysis")
    print("="*80)
    
    ckpt_20k = torch.load(ckpt_20k_path, map_location='cpu')
    ckpt_50k = torch.load(ckpt_50k_path, map_location='cpu')
    
    state_20k = ckpt_20k['model'] if 'model' in ckpt_20k else ckpt_20k
    state_50k = ckpt_50k['model'] if 'model' in ckpt_50k else ckpt_50k
    
    # 查找embedding和lm_head层
    embedding_key = None
    lm_head_key = None
    
    for key in state_20k.keys():
        if 'wte' in key and 'weight' in key:
            embedding_key = key
        if 'lm_head' in key or ('head' in key and 'weight' in key):
            lm_head_key = key
    
    print(f"\nFound layers:")
    print(f"  Embedding: {embedding_key}")
    print(f"  LM_Head: {lm_head_key}")
    
    # 分析Embedding层
    if embedding_key:
        print(f"\n📊 Embedding Layer Analysis ({embedding_key}):")
        emb_20k = state_20k[embedding_key].cpu().numpy()
        emb_50k = state_50k[embedding_key].cpu().numpy()
        
        print(f"  Shape: {emb_20k.shape}")
        print(f"  20k - Mean: {emb_20k.mean():.6f}, Std: {emb_20k.std():.6f}")
        print(f"  50k - Mean: {emb_50k.mean():.6f}, Std: {emb_50k.std():.6f}")
        
        # SVD分析
        _, s_20k, _ = np.linalg.svd(emb_20k, full_matrices=False)
        _, s_50k, _ = np.linalg.svd(emb_50k, full_matrices=False)
        
        print(f"  Effective rank (90% energy) - 20k: {np.argmax(np.cumsum(s_20k**2)/np.sum(s_20k**2) >= 0.9) + 1}")
        print(f"  Effective rank (90% energy) - 50k: {np.argmax(np.cumsum(s_50k**2)/np.sum(s_50k**2) >= 0.9) + 1}")
    
    # 分析LM_Head层
    if lm_head_key:
        print(f"\n📊 LM_Head Layer Analysis ({lm_head_key}):")
        head_20k = state_20k[lm_head_key].cpu().numpy()
        head_50k = state_50k[lm_head_key].cpu().numpy()
        
        print(f"  Shape: {head_20k.shape}")
        print(f"  20k - Mean: {head_20k.mean():.6f}, Std: {head_20k.std():.6f}")
        print(f"  50k - Mean: {head_50k.mean():.6f}, Std: {head_50k.std():.6f}")
        
        # 相似度
        cosine_sim = np.dot(head_20k.flatten(), head_50k.flatten()) / (
            np.linalg.norm(head_20k.flatten()) * np.linalg.norm(head_50k.flatten())
        )
        print(f"  Cosine similarity: {cosine_sim:.4f}")
        
        # 如果形状允许，做SVD
        if len(head_20k.shape) == 2:
            _, s_20k, _ = np.linalg.svd(head_20k, full_matrices=False)
            _, s_50k, _ = np.linalg.svd(head_50k, full_matrices=False)
            
            print(f"  Top 5 singular values - 20k: {s_20k[:5]}")
            print(f"  Top 5 singular values - 50k: {s_50k[:5]}")
    
    # 检查是否有权重共享
    if embedding_key and lm_head_key:
        print(f"\n🔗 Weight Sharing Check:")
        emb_weight = state_20k[embedding_key]
        head_weight = state_20k[lm_head_key]
        
        # 检查是否是转置关系
        if emb_weight.T.shape == head_weight.shape:
            if torch.allclose(emb_weight.T, head_weight):
                print("  ✅ Weights are shared (transposed)")
            else:
                print("  ❌ Weights are NOT shared")
        elif emb_weight.shape == head_weight.shape:
            if torch.allclose(emb_weight, head_weight):
                print("  ✅ Weights are shared (same)")
            else:
                print("  ❌ Weights are NOT shared")
        else:
            print(f"  Shape mismatch: emb={emb_weight.shape}, head={head_weight.shape}")

# 运行分析
if __name__ == "__main__":
    ckpt_20k = "out_d92/composition_mix0_seed42_20250801_054758/ckpt_mix0_seed42_iter20000.pt"
    ckpt_50k = "out_d92/composition_mix0_seed42_20250801_054758/ckpt_mix0_seed42_iter50000.pt"
    
    # 全面分析
    layer_analysis = comprehensive_layer_analysis(ckpt_20k, ckpt_50k)
    
    # 专门分析embedding和lm_head
    analyze_specific_layers(ckpt_20k, ckpt_50k)