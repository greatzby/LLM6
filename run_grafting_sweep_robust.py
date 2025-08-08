# ----------- 文件: run_grafting_sweep_robust.py (全新) -----------
# 一个健壮的、非侵入式的总控脚本，用于执行 lambda 网格搜索。
# 它通过命令行调用评估脚本，并解析其输出，无需修改评估脚本本身。
# =================================================================

import os
import argparse
import numpy as np
import subprocess
import re

# 导入我们需要的函数
# 确保 graft_advanced.py 在同一目录下
from graft_advanced import create_graft_transplant, get_final_checkpoint_path

def parse_evaluation_output(output_str):
    """
    使用正则表达式从 evaluate_hybrid_model.py 的标准输出中解析准确率。
    """
    results = {}
    try:
        # 解析 S1->S3 的准确率
        s1_s3_match = re.search(r"S1->S3\s*:\s*([\d.]+)%\s*accuracy", output_str)
        if s1_s3_match:
            results['s1_s3_acc'] = float(s1_s3_match.group(1)) / 100.0
        else:
            results['s1_s3_acc'] = 0.0

        # 解析 Overall Accuracy
        overall_match = re.search(r"Overall Accuracy:\s*([\d.]+)%", output_str)
        if overall_match:
            results['overall_acc'] = float(overall_match.group(1)) / 100.0
        else:
            results['overall_acc'] = 0.0
        
        if not s1_s3_match and not overall_match:
            print("    [警告] 无法从评估输出中解析到任何准确率。")
            
    except Exception as e:
        print(f"    [错误] 解析评估输出时发生错误: {e}")
        return {"s1_s3_acc": 0.0, "overall_acc": 0.0}
        
    return results


def run_sweep_robust(seed, data_dir, device):
    """
    执行一个完整的、健壮的网格搜索。
    """
    print("\n🔬 开始执行 Lambda 网格搜索 (健壮模式) 🔬\n")
    
    # --- 1. 定义网格搜索的范围 ---
    LAMBDAS = np.arange(0.0, 1.1, 0.1).tolist()
    LAMBDAS.extend([0.95, 0.98])
    LAMBDAS = sorted(list(set(LAMBDAS)))
    
    print(f"[*] 将要测试的 Lambda 值: {[f'{l:.2f}' for l in LAMBDAS]}")
    
    # --- 2. 定位原始模型 ---
    print("\n[*] 正在定位原始 checkpoint 文件...")
    path_0 = get_final_checkpoint_path(0, seed)
    path_20 = get_final_checkpoint_path(20, seed)
    
    results_summary = []

    # --- 3. 循环执行“生成-评估” ---
    for lam in LAMBDAS:
        lam = round(lam, 2)
        print("\n" + "="*70)
        print(f"[*] 正在处理 Lambda = {lam:.2f}")
        print("="*70)

        # --- 生成模型 ---
        try:
            grafted_model_path = create_graft_transplant(path_0, path_20, lam, seed)
        except Exception as e:
            print(f"    ❌ 生成模型时出错 (lambda={lam}): {e}")
            continue

        # --- 通过子进程调用评估脚本 ---
        print(f"\n[*] 正在通过子进程评估模型: {os.path.basename(grafted_model_path)}...")
        command = [
            "python", "evaluate_hybrid_model.py",
            "--model_path", grafted_model_path,
            "--data_dir", data_dir,
            "--device", device
        ]
        
        try:
            # 执行命令并捕获输出
            process_result = subprocess.run(
                command, 
                capture_output=True, 
                text=True, 
                check=True, # 如果命令返回非零退出码，则抛出异常
                encoding='utf-8'
            )
            
            # 打印评估脚本的完整输出，方便调试
            print("\n--- [评估脚本输出开始] ---")
            print(process_result.stdout)
            print("--- [评估脚本输出结束] ---\n")

            # 解析输出以获取结果
            eval_results = parse_evaluation_output(process_result.stdout)
            
            print(f"    ✅ 评估完成。解析到组合能力 (S1->S3): {eval_results.get('s1_s3_acc', 0.0):.2%}")
            results_summary.append({
                "lambda": lam,
                "s1_s3_acc": eval_results.get('s1_s3_acc', 0.0),
                "overall_acc": eval_results.get('overall_acc', 0.0)
            })

        except FileNotFoundError:
            print(f"    ❌ 错误：找不到 'evaluate_hybrid_model.py'。请确保它在当前目录中。")
            break # 中断整个 sweep
        except subprocess.CalledProcessError as e:
            print(f"    ❌ 评估脚本执行失败 (lambda={lam})，返回码: {e.returncode}")
            print("    --- [评估脚本错误输出] ---")
            print(e.stderr)
            print("    --- [错误输出结束] ---")
            continue # 继续下一个 lambda
        except Exception as e:
            print(f"    ❌ 执行评估时发生未知错误 (lambda={lam}): {e}")
            continue


    # --- 4. 打印最终的总结报告 ---
    print("\n\n" + "#"*70)
    print(" " * 20 + "🔬 网格搜索最终总结报告 🔬")
    print("#"*70 + "\n")
    
    results_summary.sort(key=lambda x: x['lambda'])
    
    print(f"{'Lambda':<10} | {'组合能力 (S1->S3)':<25} | {'总体性能':<20}")
    print("-" * 60)
    for res in results_summary:
        s1_s3_str = f"{res['s1_s3_acc']:.2%}"
        overall_str = f"{res['overall_acc']:.2%}"
        print(f"{res['lambda']:<10.2f} | {s1_s3_str:<25} | {overall_str:<20}")
        
    print("\n🎉 网格搜索已全部完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="执行一个完整的、健壮的 lambda 网格搜索。")
    parser.add_argument('--seed', type=int, default=42, help='指定实验用的随机种子 (例如: 42)')
    parser.add_argument('--data_dir', type=str, default='data/simple_graph/composition_90', help='评估数据的目录')
    parser.add_argument('--device', type=str, default='cuda:0', help="用于评估的设备 (e.g., 'cuda:0' or 'cpu')")
    args = parser.parse_args()
    
    run_sweep_robust(args.seed, args.data_dir, args.device)