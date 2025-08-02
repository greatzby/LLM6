#!/bin/bash

# 设置脚本在遇到任何错误时立即退出
set -e

# ===================================================================
#           “剂量效应”混合模型批量评估脚本
#
#  本脚本将自动评估所有由 'run_dose_response_ablation.py' 
#  生成的 hybrid_model_*_k*.pt 文件。
# ===================================================================

# --- 关键配置 ---
SEED_TO_EVALUATE=42
DATA_DIR="data/simple_graph/composition_90" # 统一使用0% mix的测试集
STRATEGIES=('by_angle_top20' 'by_energy_top20' 'control_stable_bottom20')
K_VALUES=$(seq 1 10) # k 从 1 到 10

echo "================================================="
echo "🚀 开始批量评估“剂量效应”混合模型..."
echo "================================================="
echo "将评估种子: ${SEED_TO_EVALUATE}"
echo "将使用测试数据: ${DATA_DIR}"
echo ""

# --- 循环执行所有评估任务 ---

for strategy in "${STRATEGIES[@]}"; do
    echo ""
    echo "-------------------------------------------------"
    echo "📊 开始评估策略: ${strategy}"
    echo "-------------------------------------------------"
    
    for k in $K_VALUES; do
        MODEL_FILE="hybrid_model_seed${SEED_TO_EVALUATE}_${strategy}_k${k}.pt"
        
        if [ -f "$MODEL_FILE" ]; then
            echo ""
            echo "--- 评估 k=${k} ---"
            python evaluate_hybrid_model.py --model_path "${MODEL_FILE}" --data_dir "${DATA_DIR}"
        else
            echo ""
            echo "--- 跳过 k=${k} (文件未找到: ${MODEL_FILE}) ---"
        fi
    done
done


# --- 打印结束信息 ---
echo ""
echo "================================================="
echo "🎉🎉🎉 全部评估任务均已成功执行完毕！ 🎉🎉🎉"
echo "================================================="
echo "请检查上面的输出，并将每个k值对应的 S1->S3 准确率记录下来，"
echo "以便绘制“剂量-效应”曲线图。"