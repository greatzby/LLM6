#!/bin/bash
set -e

# ==============================================================================
#                  最终基准测试脚本 (v2.4 - 结合了日志记录的最佳实践)
# ==============================================================================

# --- 配置区 ---
LOG_DIR="final_benchmark_logs_all"
BASELINE_MODEL_PATH="out_d92/composition_mix0_seed42_20250801_054758/ckpt_mix0_seed42_iter50000.pt"
DATA_DIR="data/simple_graph/composition_90"
EVAL_SCRIPT="evaluate_hybrid_model.py"

# *** 定义所有需要评估的模型目录 ***
MODEL_DIRS_TO_EVAL=("hybrid_models" "hybrid_models_convex" "hybrid_models_mapping_v2")

# --- 脚本开始 ---
mkdir -p "$LOG_DIR"
echo "=============================================================================="
echo "    📊  最终基准测试: 评估所有模型变体 (v2.4) 📊"
echo "    日志将保存在目录: $LOG_DIR"
echo "=============================================================================="

# --- 步骤 1: 评估基线模型 ---
echo "🚀 [步骤 1/3] 正在测量基线模型性能..."
BASELINE_LOG_FILE="$LOG_DIR/benchmark_result_BASELINE.log"
python "$EVAL_SCRIPT" \
  --model_path "$BASELINE_MODEL_PATH" \
  --data_dir "$DATA_DIR" | tee "$BASELINE_LOG_FILE"

# --- 步骤 2: 批量评估所有生成的模型 ---
echo "🚀 [步骤 2/3] 正在批量评估所有生成的模型..."
# 从所有指定的目录中查找模型，并存入一个总列表
mapfile -t ALL_CANDIDATE_MODELS < <(find "${MODEL_DIRS_TO_EVAL[@]}" -name "*.pt" 2>/dev/null | sort)

if [ ${#ALL_CANDIDATE_MODELS[@]} -eq 0 ]; then
    echo "警告: 在指定的目录中没有找到任何 .pt 模型文件。请检查目录内容。"
else
    for model_path in "${ALL_CANDIDATE_MODELS[@]}"; do
      model_filename=$(basename "$model_path" .pt)
      LOG_FILE="$LOG_DIR/benchmark_result_${model_filename}.log"
      echo "  [*] 正在评估模型: $model_filename ..."
      python "$EVAL_SCRIPT" --model_path "$model_path" --data_dir "$DATA_DIR" | tee "$LOG_FILE"
      echo "  [✔] 评估完成: $model_filename"
    done
fi

# --- 步骤 3: 生成最终对比报告 ---
echo "🚀 [步骤 3/3] 生成最终对比报告..."
echo ""
echo "========================================================================================"
echo "                      🏆  最终基准测试总报告 (v2.4) 🏆"
echo "========================================================================================"
printf "%-60s | %-12s | %-12s\n" "模型" "S1->S3 Acc." "Overall Acc."
echo "----------------------------------------------------------------------------------------"

# 从日志中解析并打印基线结果
BASELINE_S1S3=$(grep "S1->S3" "$BASELINE_LOG_FILE" | awk '{print $4}')
BASELINE_OVERALL=$(grep "Overall Accuracy" "$BASELINE_LOG_FILE" | awk '{print $3}')
printf "%-60s | %-12s | %-12s\n" "真实基线 (mix0)" "${BASELINE_S1S3:-N/A}" "${BASELINE_OVERALL:-N/A}"

# 打印硬编码的最佳历史结果
printf "%-60s | %-12s | %-12s\n" "剂量效应最佳 (by_energy, k=5)" "76.00%" "92.00%"
echo "----------------------------------------------------------------------------------------"

# 从日志中解析并打印所有生成模型的结果
if [ ${#ALL_CANDIDATE_MODELS[@]} -ne 0 ]; then
    for model_path in "${ALL_CANDIDATE_MODELS[@]}"; do
      model_filename=$(basename "$model_path" .pt)
      LOG_FILE="$LOG_DIR/benchmark_result_${model_filename}.log"
      if [ -f "$LOG_FILE" ]; then
        # 使用您脚本中经过验证的 awk 命令来提取数据
        S1S3=$(grep "S1->S3" "$LOG_FILE" | awk '{print $4}')
        OVERALL=$(grep "Overall Accuracy" "$LOG_FILE" | awk '{print $3}')
        printf "%-60s | %-12s | %-12s\n" "$model_filename" "${S1S3:-N/A}" "${OVERALL:-N/A}"
      fi
    done
fi
echo "========================================================================================"