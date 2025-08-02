#!/bin/bash

# 设置脚本在遇到任何错误时立即退出
set -e

# ===================================================================
#           “定点嫁接”实验自动化套件
#
#  本脚本将自动执行你设计的、针对 by_energy 策略中
#  “黄金维度”(6, 7, 8) 的所有组合嫁接与评估。
# ===================================================================

# --- 关键配置 ---
SEED=42
STRATEGY="by_energy_top20"
STRATEGY_SHORT_NAME="by_energy" # 用于文件名
DATA_DIR="data/simple_graph/composition_90"

# 定义我们要测试的所有维度组合
# 单维度、双维度、三维度
declare -a TARGET_INDICES=("6" "7" "8" "6,7" "6,8" "7,8" "6,7,8")

echo "================================================="
echo "🔬 开始执行“黄金维度”定点嫁接与评估套件..."
echo "================================================="
echo "种子: ${SEED} | 策略: ${STRATEGY}"
echo ""

# --- 第一部分: 循环生成所有7个混合模型 ---
echo "--- [阶段 1/2] 正在生成 7 个定点嫁接模型 ---"
for indices in "${TARGET_INDICES[@]}"; do
    echo ""
    echo ">>> 正在生成模型，目标维度: ${indices}"
    python run_targeted_ablation.py --seed ${SEED} --strategy ${STRATEGY} --indices "${indices}"
done
echo "--- ✅ 所有模型生成完毕 ---"
echo ""


# --- 第二部分: 循环评估所有7个新生成的模型 ---
echo "--- [阶段 2/2] 正在评估 7 个新模型 ---"
for indices in "${TARGET_INDICES[@]}"; do
    indices_str=$(echo ${indices} | tr ',' '_')
    MODEL_FILE="hybrid_model_seed${SEED}_${STRATEGY_SHORT_NAME}_indices_${indices_str}.pt"
    
    echo ""
    echo ">>> 正在评估模型: ${MODEL_FILE}"
    if [ -f "$MODEL_FILE" ]; then
        python evaluate_hybrid_model.py --model_path "${MODEL_FILE}" --data_dir "${DATA_DIR}"
    else
        echo "🚨 错误: 未找到模型文件 ${MODEL_FILE}，跳过评估。"
    fi
done

# --- 打印结束信息 ---
echo ""
echo "================================================="
echo "🎉🎉🎉 “黄金维度”实验套件全部执行完毕！ 🎉🎉🎉"
echo "================================================="
echo "请仔细检查上面的7个评估结果，寻找性能最高的组合。"