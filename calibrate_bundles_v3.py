# 文件名: calibrate_bundles_v3.py
# 描述: 最终修复版。严格处理 bias=False 的情况，确保不创建不存在的偏置项。

import torch
import os
import argparse
import glob
from tqdm import tqdm
import numpy as np

# --- 从 model.py 导入 ---
from model import GPT, GPTConfig

# --- 辅助函数与核心逻辑 ---

def get_bundle_checkpoint_path(bundle_type, seed):
    path = f"hybrid_models/hybrid_bundle_{bundle_type}_bundle_transplant_seed{seed}.pt"
    if not os.path.exists(path): raise FileNotFoundError(f"错误：未找到预期的 'bundle' 模型: {path}")
    print(f"  > 定位到 Bundle 模型: {path}")
    return path

def get_base_model_path(seed):
    pattern = f"out_d92/composition_mix0_seed{seed}_*"
    dirs = glob.glob(pattern)
    if not dirs: raise FileNotFoundError(f"错误：未找到匹配的目录: {pattern}")
    latest_dir = sorted(dirs)[-1]
    path = os.path.join(latest_dir, f"ckpt_mix0_seed{seed}_iter50000.pt")
    if not os.path.exists(path): raise FileNotFoundError(f"错误：在目录 {latest_dir} 中未找到预期的基线模型")
    print(f"  > 定位到基线 (Host) 模型: {path}")
    return path

def get_activations(model, data_loader, hook_fn):
    model.eval()
    activations = []
    with torch.no_grad():
        for X, _ in tqdm(data_loader, desc="  - 采集数据中"):
            _ = model(X.to(device))
            activations.append(hook_fn.output.cpu())
    return torch.cat(activations, dim=0)

class ActivationHook:
    def __init__(self): self.output = None
    def __call__(self, module, module_in, module_out): self.output = module_out.detach()

def calculate_calibration_params(host_stats, bundle_stats, args):
    mu_h, sigma_h = host_stats['mean'], host_stats['std']
    mu_b, sigma_b = bundle_stats['mean'], bundle_stats['std']
    a = sigma_h / (sigma_b + args.eps)
    a = torch.clamp(a, min=args.clamp_min, max=args.clamp_max)
    b = mu_h - a * mu_b
    print(f"  > 计算得到的 'a' (缩放) 范围: [{a.min().item():.4f}, {a.max().item():.4f}]")
    print(f"  > 计算得到的 'b' (偏移) 范围: [{b.min().item():.4f}, {b.max().item():.4f}]")
    return a, b

# --- 核心修复逻辑在此函数中 ---
def fold_parameters_v3(a, b, g, proj_weight, proj_bias):
    """
    V3 修复版：将 a, b, g 折叠进 c_proj 层的权重和偏置。
    关键修复：只有在 proj_bias 不是 None 的情况下，才计算和返回 new_bias。
    """
    a = a * g
    b = b * g # 偏移 b 也需要被门控 g 缩放
    
    # 折叠权重 (这部分不变)
    new_weight = torch.diag(a) @ proj_weight
    
    # --- 关键修复逻辑 ---
    new_bias = None
    if proj_bias is not None:
        # 只有当原始层有偏置时，我们才计算新的偏置
        print("    - 检测到存在偏置项，将进行折叠。")
        new_bias = a * proj_bias + b
    else:
        # 如果原始层没有偏置，我们什么都不做，并告知用户
        print("    - 未检测到偏置项。校准偏移 'b' 将被忽略。")
        # new_bias 保持为 None
        
    return new_weight, new_bias

# --- 主执行逻辑 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="路线图 A+ (v3)：对 Bundle 移植模型进行统计校准 (修复 bias 问题)")
    # ... (参数定义与 v2 完全相同)
    parser.add_argument('--seed', type=int, default=42, help='实验用的随机种子')
    parser.add_argument('--bundle_type', type=str, required=True, choices=['attn', 'mlp'], help='要校准的 Bundle 类型')
    parser.add_argument('--data_path', type=str, default='data/simple_graph/composition_90/train_10.bin', help='用于统计的未标注数据路径')
    parser.add_argument('--batch_size', type=int, default=64, help='数据加载的批次大小')
    parser.add_argument('--num_tokens', type=int, default=20000, help='用于统计的总 token 数')
    parser.add_argument('--eps', type=float, default=1e-6, help='防止除以零的小常数')
    parser.add_argument('--clamp_min', type=float, default=0.1, help='缩放参数 a 的下限')
    parser.add_argument('--clamp_max', type=float, default=10.0, help='缩放参数 a 的上限')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    print("\n步骤 1: 加载模型和数据...")
    base_model_path = get_base_model_path(args.seed)
    ckpt_base = torch.load(base_model_path, map_location='cpu')
    gpt_config = GPTConfig(**ckpt_base['model_args'])
    
    model_base = GPT(gpt_config)
    model_base.load_state_dict(ckpt_base['model'])
    model_base.to(device)

    bundle_model_path = get_bundle_checkpoint_path(args.bundle_type, args.seed)
    ckpt_bundle = torch.load(bundle_model_path, map_location='cpu')
    model_bundle = GPT(gpt_config)
    model_bundle.load_state_dict(ckpt_bundle['model'])
    model_bundle.to(device)
    
    data = np.memmap(args.data_path, dtype=np.uint16, mode='r')
    num_samples = args.num_tokens // gpt_config.block_size
    
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, data, block_size, num_samples):
            self.data = data
            self.block_size = block_size
            self.num_samples = min(num_samples, (len(data) - 1) // block_size)
        def __len__(self): return self.num_samples
        def __getitem__(self, idx):
            chunk = self.data[idx*self.block_size:(idx+1)*self.block_size + 1]
            x = torch.from_numpy(chunk[:-1].astype(np.int64))
            y = torch.from_numpy(chunk[1:].astype(np.int64))
            return x, y

    dataset = SimpleDataset(data, gpt_config.block_size, num_samples)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)
    print(f"数据加载完成。将使用 {len(dataset)} 个样本 (~{len(dataset)*gpt_config.block_size} tokens) 进行统计。")

    print("\n步骤 2: 采集激活值统计...")
    target_module_path = f'transformer.h.0.{args.bundle_type}.c_proj'
    
    print("-> 正在处理 Host 模型 (mix0)...")
    hook_base = ActivationHook()
    handle_base = model_base.get_submodule(target_module_path).register_forward_hook(hook_base)
    activations_base = get_activations(model_base, loader, hook_base)
    handle_base.remove()
    host_stats = {'mean': activations_base.mean(dim=[0, 1]), 'std': activations_base.std(dim=[0, 1])}
    
    print(f"-> 正在处理 {args.bundle_type} Bundle 模型...")
    hook_bundle = ActivationHook()
    handle_bundle = model_bundle.get_submodule(target_module_path).register_forward_hook(hook_bundle)
    activations_bundle = get_activations(model_bundle, loader, hook_bundle)
    handle_bundle.remove()
    bundle_stats = {'mean': activations_bundle.mean(dim=[0, 1]), 'std': activations_bundle.std(dim=[0, 1])}
    
    print("\n步骤 3: 计算并折叠校准参数...")
    a, b = calculate_calibration_params(host_stats, bundle_stats, args)
    
    output_dir = "hybrid_models"
    os.makedirs(output_dir, exist_ok=True)
    generated_files = []
    
    gating_factors = [0.2, 0.3, 0.5, 1.0]
    
    for g in gating_factors:
        print(f"--> 处理残差门控 g = {g}")
        
        proj_layer = model_bundle.get_submodule(target_module_path)
        proj_weight = proj_layer.weight.data
        proj_bias = proj_layer.bias.data if hasattr(proj_layer, 'bias') and proj_layer.bias is not None else None
        
        # 使用 v3 版本的折叠函数
        new_weight, new_bias = fold_parameters_v3(a.to(device), b.to(device), g, proj_weight, proj_bias)
        
        calibrated_state_dict = ckpt_bundle['model'].copy()
        calibrated_state_dict[f'{target_module_path}.weight'] = new_weight
        
        # --- 关键修复逻辑 ---
        if new_bias is not None:
            # 只有在 new_bias 被成功计算出来时，才更新 state_dict
            calibrated_state_dict[f'{target_module_path}.bias'] = new_bias
        elif f'{target_module_path}.bias' in calibrated_state_dict:
            # 如果原始 state_dict 中意外存在 bias (理论上不应发生)，我们最好移除它以匹配模型结构
            del calibrated_state_dict[f'{target_module_path}.bias']
        
        new_ckpt = ckpt_bundle.copy()
        new_ckpt['model'] = calibrated_state_dict
        
        output_filename = f"hybrid_bundle_{args.bundle_type}_calibrated_g{g}_seed{args.seed}.pt"
        output_path = os.path.join(output_dir, output_filename)
        torch.save(new_ckpt, output_path)
        print(f"    ✅ 校准后的模型已保存至: {output_path}")
        generated_files.append(output_path)

    print("\n步骤 4: 生成批量评估脚本...")
    script_content = "#!/bin/bash\n"
    script_content += f"echo '🚀 开始批量评估 {args.bundle_type} Bundle 的校准模型 (路线图 A+ v3)... 🚀'\n"
    
    for model_path in generated_files:
        script_content += f"\necho '\n--- 正在评估: {os.path.basename(model_path)} ---'\n"
        script_content += f"python evaluate_hybrid_model.py --model_path {model_path} --data_dir data/simple_graph/composition_90 --temperature 0\n"
        
    script_filename = f"run_{args.bundle_type}_bundle_calibration_eval.sh"
    with open(script_filename, 'w') as f:
        f.write(script_content)
    
    print("-" * 60)
    print(f"🎉 校准模型已全部生成！(v3 版)")
    print(f"   评估脚本 '{script_filename}' 也已更新。")