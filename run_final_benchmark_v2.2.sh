#!/bin/bash

set -e

# --- 配置 ---
LOG_DIR="final_benchmark_logs_subspace"
BASELINE_MODEL_PATH="out_d92/composition_mix0_seed42_20250801_054758/ckpt_mix0_seed42_iter50000.pt"
DATA_DIR="data/simple_graph/composition_90"
HYBRID_MODEL_DIR="hybrid_models"

# --- 脚本开始 ---
mkdir -p $LOG_DIR
echo "=============================================================================="
echo "    📊  最终基准测试: 子空间岭回归模型评估 📊"
echo "=============================================================================="

# --- 步骤 1: 测量基线 ---
echo "🚀 [步骤 1/2] 正在测量真实的基线模型性能..."
BASELINE_LOG_FILE="$LOG_DIR/benchmark_result_BASELINE.log"
python evaluate_hybrid_model.py \
  --model_path "$BASELINE_MODEL_PATH" \
  --data_dir "$DATA_DIR" | tee "$BASELINE_LOG_FILE"
echo "  [✔] 真实基线测量完成。"
echo "------------------------------------------------------------------------------"

# --- 步骤 2: 测量所有子空间模型 ---
echo "🚀 [步骤 2/2] 正在批量评估所有子空间岭回归模型..."
# 查找所有符合命名规则的模型
CANDIDATE_MODELS=($(find "$HYBRID_MODEL_DIR" -name "subspace_ridge_r*_lam*_seed*.pt"))

for model_path in "${CANDIDATE_MODELS[@]}"; do
    model_filename=$(basename "$model_path")
    # 移除了.pt后缀
    model_name="${model_filename%.*}" 
    LOG_FILE="$LOG_DIR/benchmark_result_${model_name}.log"
    
    echo "  [*] 正在评估模型: $model_filename ..."
    python evaluate_hybrid_model.py \
      --model_path "$model_path" \
      --data_dir "$DATA_DIR" | tee "$LOG_FILE"
    echo "  [✔] 评估完成: $model_filename"
done
echo "------------------------------------------------------------------------------"

# --- 步骤 3: 生成最终报告 ---
echo "🚀 [步骤 3/3] 生成最终对比报告..."
echo ""
echo "=============================================================================="
echo "                      🏆  最终基准测试报告 (子空间岭回归) 🏆"
echo "=============================================================================="
echo ""
printf "%-40s | %-15s | %-15s\n" "模型" "S1->S3 Acc." "Overall Acc."
echo "--------------------------------------------------------------------------------"

# 打印基线结果
BASELINE_S1S3=$(grep "S1->S3" "$BASELINE_LOG_FILE" | awk '{print $5}' || echo "N/A")
BASELINE_OVERALL=$(grep "Overall" "$BASELINE_LOG_FILE" | awk '{print $3}' || echo "N/A")
printf "%-40s | %-15s | %-15s\n" "真实基线 (mix0)" "$BASELINE_S1S3" "$BASELINE_OVERALL"

# 打印手动剂量效应的最佳结果作为参考
printf "%-40s | %-15s | %-15s\n" "剂量效应最佳 (by_energy, k=5)" "76.00%" "92.00%"


# 打印所有子空间模型的结果
for model_path in "${CANDIDATE_MODELS[@]}"; do
    model_filename=$(basename "$model_path")
    model_name="${model_filename%.*}"
    LOG_FILE="$LOG_DIR/benchmark_result_${model_name}.log"
    if [ -f "$LOG_FILE" ]; then
        S1S3_ACC=$(grep "S1->S3" "$LOG_FILE" | awk '{print $5}' || echo "N/A")
        OVERALL_ACC=$(grep "Overall" "$LOG_FILE" | awk '{print $3}' || echo "N/A")
        printf "%-40s | %-15s | %-15s\n" "$model_name" "$S1S3_ACC" "$OVERALL_ACC"
    fi
done
echo "=============================================================================="