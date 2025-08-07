# create_full_transplant_model.py

import torch
import os
import argparse
import glob

def get_final_checkpoint_path(ratio, seed, checkpoint_dir="out_d92"):
    """复用之前的路径查找函数。"""
    pattern = f"{checkpoint_dir}/composition_mix{ratio}_seed{seed}_*"
    dirs = glob.glob(pattern)
    if not dirs:
        raise FileNotFoundError(f"未找到匹配的目录: {pattern}")
    
    latest_dir = sorted(dirs)[-1]
    iteration = 50000
    expected_filename = f"ckpt_mix{ratio}_seed{seed}_iter{iteration}.pt"
    path = os.path.join(latest_dir, expected_filename)
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"在目录 {latest_dir} 中未找到预期的最终 checkpoint 文件 '{expected_filename}'")

    print(f"  > 自动定位到模型: {path}")
    return path

def create_full_transplant(path_template, path_source):
    """
    执行“全脑移植”：将源模型的所有权重复制到模板模型中。
    """
    print("  > 加载模板模型 (用于结构和非模型元数据):")
    print(f"    - {path_template}")
    ckpt_template = torch.load(path_template, map_location='cpu')

    print("  > 加载源模型 (用于所有权重):")
    print(f"    - {path_source}")
    ckpt_source = torch.load(path_source, map_location='cpu')

    # 获取源模型的权重字典
    state_source = ckpt_source.get('model', ckpt_source)

    print("  > 开始执行“全脑移植”...")
    # 将源模型的权重字典直接替换到模板中
    # 这一步会替换掉 wte, wpe, 所有的 h.*, ln_f, 和 lm_head
    if 'model' in ckpt_template:
        ckpt_template['model'] = state_source
        print("    - 已将源模型的 'model' state_dict 完整替换到模板中。")
    else:
        ckpt_template.update(state_source)
        print("    - 已将源模型的权重直接更新到模板的根级别。")

    return ckpt_template

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="执行“全脑移植”实验，以验证内部通路的重要性。")
    parser.add_argument('--seed', type=int, required=True, help='要操作的随机种子 (例如: 42, 123, 456)')
    args = parser.parse_args()

    SEED = args.seed
    print("="*60)
    print("     🚀 开始执行“全脑移植”实验 (最终诊断) 🚀     ")
    print(f"     操作种子 (Seed): {SEED}")
    print("="*60)

    try:
        print("\n步骤 1: 定位模型文件...")
        path_0 = get_final_checkpoint_path(0, SEED)
        path_20 = get_final_checkpoint_path(20, SEED)

        print("\n步骤 2: 开始执行全脑移植...")
        transplant_ckpt = create_full_transplant(path_template=path_0, path_source=path_20)

        output_dir = "hybrid_models"
        os.makedirs(output_dir, exist_ok=True)
        output_filename = os.path.join(output_dir, f"hybrid_full_transplant_seed{SEED}.pt")
        torch.save(transplant_ckpt, output_filename)

        print("\n" + "="*60)
        print("✅ 全脑移植模型已成功生成！")
        print(f"模型已保存至: {output_filename}")
        print("\n下一步：请评估此模型。其结果应与 mix20 模型完全一致。")

    except Exception as e:
        print(f"\n🚨 致命错误: {e}")
        import traceback
        traceback.print_exc()
        exit(1)