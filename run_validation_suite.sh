#!/bin/bash

# ===================================================================
#           🔬 顶会级论文验证套件 (Validation Suite) 🔬
#
#  本脚本将为多个随机种子 (123, 456) 自动执行完整的“剂量效应”
#  实验，包括模型生成和评估，以验证研究结论的普适性。
# ===================================================================

# --- 配置 ---
# 在此数组中添加或修改你想要验证的种子
SEEDS_TO_PROCESS=(123 456)

# 实验参数 (与之前的脚本保持一致)
STRATEGIES=('by_angle_top20' 'by_energy_top20' 'control_stable_bottom20')
K_VALUES=({1..10})
DATA_DIR="data/simple_graph/composition_90"

# 设置脚本在遇到任何错误时立即退出
set -e

# --- 主循环: 遍历所有需要验证的种子 ---
for seed in "${SEEDS_TO_PROCESS[@]}"; do
    echo ""
    echo "================================================================="
    echo "          🚀 开始处理随机种子 (Seed): $seed 🚀"
    echo "================================================================="
    
    # --- 阶段 1: 为当前种子生成全部30个混合模型 ---
    echo ""
    echo "--- [阶段 1/2] 正在为 Seed $seed 生成混合模型... ---"
    python run_dose_response_ablation.py --seed "$seed"
    echo "--- ✅ Seed $seed 的模型生成完毕 ---"
    echo ""

    # --- 阶段 2: 评估刚刚为当前种子生成的所有模型 ---
    echo "--- [阶段 2/2] 正在评估 Seed $seed 的所有混合模型... ---"
    for strategy in "${STRATEGIES[@]}"; do
        echo ""
        echo "-------------------------------------------------"
        echo "📊 开始评估策略 (Seed $seed): $strategy"
        echo "-------------------------------------------------"
        for k in "${K_VALUES[@]}"; do
            MODEL_FILE="hybrid_model_seed${seed}_${strategy}_k${k}.pt"
            
            echo ""
            echo "--- 评估 k=$k ---"
            
            if [ -f "$MODEL_FILE" ]; then
                python evaluate_hybrid_model.py --model_path "${MODEL_FILE}" --data_dir "${DATA_DIR}"
            else
                echo "🚨 错误: 未找到模型文件 ${MODEL_FILE}，跳过评估。"
                echo "🚨 请检查阶段1的生成过程是否出错。"
            fi
        done
    done
    echo ""
    echo "================================================================="
    echo "          🎉 Seed $seed 的全部评估任务均已成功执行完毕！ 🎉"
    echo "================================================================="
    echo ""
done

echo ""
echo "================================================================="
echo "          🎉🎉🎉 所有验证套件均已成功执行完毕！ 🎉🎉🎉"
echo "================================================================="
echo "请仔细检查以上所有输出，为每个种子绘制剂量效应曲线图。"