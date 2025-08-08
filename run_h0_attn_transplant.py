import torch
import os
import argparse
import glob

def get_final_checkpoint_path(ratio, seed, checkpoint_dir="out_d92"):
    """辅助函数，用于查找并验证模型检查点的路径。"""
    pattern = f"{checkpoint_dir}/composition_mix{ratio}_seed{seed}_*"
    dirs = glob.glob(pattern)
    if not dirs: raise FileNotFoundError(f"错误：未找到匹配的目录: {pattern}")
    latest_dir = sorted(dirs)[-1]
    iteration = 50000
    expected_filename = f"ckpt_mix{ratio}_seed{seed}_iter{iteration}.pt"
    path = os.path.join(latest_dir, expected_filename)
    if not os.path.exists(path): raise FileNotFoundError(f"错误：在目录 {latest_dir} 中未找到预期的最终 checkpoint 文件 '{expected_filename}'")
    print(f"  > 定位到模型: {path}")
    return path

def attention_module_transplant(state_0, state_20):
    """
    完整移植h.0层中与Attention模块相关的所有权重。
    采纳了您的建议，动态检查键的存在，以提高鲁棒性。
    """
    state_hybrid = {key: val.clone() for key, val in state_0.items()}
    
    # 根据您的建议，定义一个更完整的潜在权重列表
    attn_keys = [
        # Attention核心权重
        'transformer.h.0.attn.c_attn.weight',
        'transformer.h.0.attn.c_proj.weight',
        # Attention之前的LayerNorm（其功能与Attention紧密耦合）
        'transformer.h.0.ln_1.weight',
        'transformer.h.0.ln_1.bias',  # 动态检查此项是否存在
    ]
    
    print("  🎯 目标模块: h.0.attn (包括其前面的LayerNorm ln_1)")
    print("      > 正在动态检查并移植以下权重:")
    
    transplanted_count = 0
    for key in attn_keys:
        if key in state_20:  # 只移植源模型中存在的键
            state_hybrid[key] = state_20[key].clone()
            print(f"        ✓ {key}")
            transplanted_count += 1
        else:
            print(f"        - {key} (在源模型中不存在，跳过)")
    
    print(f"      > 成功移植 {transplanted_count} 个权重。")
    return state_hybrid

# --- 主执行逻辑 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="对h.0 Attention模块进行整体移植。")
    parser.add_argument('--seed', type=int, default=42, help='指定实验用的随机种子')
    args = parser.parse_args()

    SEED_TO_TEST = args.seed

    print("="*60)
    print("     🚀 开始 h.0 Attention 模块移植 (最终决战 v5.1) 🚀     ")
    print(f"     操作种子 (Seed): {SEED_TO_TEST}")
    print("="*60)

    print("\n步骤 1: 加载基准模型...")
    path_0 = get_final_checkpoint_path(0, SEED_TO_TEST)
    path_20 = get_final_checkpoint_path(20, SEED_TO_TEST)
    ckpt_0 = torch.load(path_0, map_location='cpu')
    ckpt_20 = torch.load(path_20, map_location='cpu')
    state_0, state_20 = ckpt_0['model'], ckpt_20['model']
    print("步骤 1 完成。")

    print("\n步骤 2: 生成混合模型...")
    output_dir = f"hybrid_models_h0_attn_transplant_seed{SEED_TO_TEST}"
    os.makedirs(output_dir, exist_ok=True)

    state_hybrid = attention_module_transplant(state_0, state_20)
    
    ckpt_hybrid = ckpt_0.copy()
    ckpt_hybrid['model'] = state_hybrid
    output_path = os.path.join(output_dir, "hybrid_h0_attn_transplant.pt")
    torch.save(ckpt_hybrid, output_path)
    print(f"    ✅ 模型已保存至: {output_path}")

    print("\n步骤 3: 生成评估脚本...")
    script_content = f"#!/bin/bash\necho '🚀 开始评估h.0 Attention模块移植模型... 🚀'\n"
    script_content += f"python evaluate_hybrid_model.py --model_path {output_path} --data_dir data/simple_graph/composition_90\n"
    
    script_filename = f"run_h0_attn_evaluation_seed{SEED_TO_TEST}.sh"
    with open(script_filename, 'w') as f:
        f.write(script_content)
    os.chmod(script_filename, 0o755)
    print(f"    ✅ 评估脚本已生成: ./{script_filename}")

    print("\n" + "="*60)
    print("🎉🎉🎉 最终战场模型已生成！ 🎉🎉🎉")
    print("="*60)
    print(f"下一步：请运行 './{script_filename}' 脚本。这将是对我们最终假说的决定性检验。")