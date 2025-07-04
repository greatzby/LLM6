# direct_model_test.py
import torch
import os
import pickle
from model import GPTConfig, GPT
import networkx as nx

def test_model_directly(checkpoint_path, data_dir):
    """直接测试模型的S1->S3性能"""
    print(f"\nTesting: {checkpoint_path}")
    
    # 加载模型
    checkpoint = torch.load(checkpoint_path)
    config = GPTConfig(**checkpoint['model_args'])
    model = GPT(config).to('cuda')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # 加载数据
    with open(os.path.join(data_dir, 'stage_info.pkl'), 'rb') as f:
        stage_info = pickle.load(f)
    S1, S2, S3 = stage_info['stages']
    
    with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    stoi = meta['stoi']
    
    # 加载图
    G = nx.read_graphml(os.path.join(data_dir, 'composition_graph.graphml'))
    
    # 测试S1->S3
    success = 0
    total = 0
    
    # 随机选择10个S1->S3对
    import random
    pairs = [(s1, s3) for s1 in S1 for s3 in S3 if nx.has_path(G, str(s1), str(s3))]
    test_pairs = random.sample(pairs, min(10, len(pairs)))
    
    for source, target in test_pairs:
        # 生成路径
        prompt = f"{source} {target} {source}"
        prompt_ids = [stoi[t] for t in prompt.split() if t in stoi]
        
        x = torch.tensor(prompt_ids, dtype=torch.long, device='cuda').unsqueeze(0)
        
        # 生成
        with torch.no_grad():
            y = model.generate(x, max_new_tokens=30, temperature=0.1, top_k=10)
        
        # 解码
        generated = []
        for tid in y[0].tolist():
            if tid == 1:  # EOS
                break
            if tid in meta['itos'] and tid > 1:
                try:
                    generated.append(int(meta['itos'][tid]))
                except:
                    pass
        
        # 验证路径
        if len(generated) >= 3:
            path = generated[2:]  # 跳过prompt
            if len(path) >= 2 and path[0] == source and path[-1] == target:
                # 检查是否经过S2
                has_s2 = any(node in S2 for node in path[1:-1])
                if has_s2:
                    success += 1
        
        total += 1
        print(f"  {source}→{target}: {'✓' if success == total else '✗'} Path: {generated[2:] if len(generated) >= 3 else 'None'}")
    
    accuracy = success / total if total > 0 else 0
    print(f"S1→S3 Success Rate: {accuracy:.2%} ({success}/{total})")
    return accuracy

# 测试关键checkpoint
if __name__ == "__main__":
    configs = [
        ('original-50k', 'out/composition_20250702_063926/ckpt_50000.pt', 'data/simple_graph/composition_90'),
        ('10% mixed-50k', 'out/composition_20250703_011304/ckpt_50000.pt', 'data/simple_graph/composition_90_mixed_10'),
    ]
    
    for name, ckpt, data in configs:
        test_model_directly(ckpt, data)