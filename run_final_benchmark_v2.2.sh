#!/bin/bash
set -e

LOG_DIR="final_benchmark_logs_subspace"
BASELINE_MODEL_PATH="out_d92/composition_mix0_seed42_20250801_054758/ckpt_mix0_seed42_iter50000.pt"
DATA_DIR="data/simple_graph/composition_90"
HYBRID_MODEL_DIR="hybrid_models"

mkdir -p "$LOG_DIR"
echo "=============================================================================="
echo "    📊  最终基准测试: 子空间岭回归模型评估 📊"
echo "=============================================================================="

echo "🚀 [步骤 1/3] 正在测量基线模型性能..."
BASELINE_LOG_FILE="$LOG_DIR/benchmark_result_BASELINE.log"
python evaluate_hybrid_model.py \
  --model_path "$BASELINE_MODEL_PATH" \
  --data_dir "$DATA_DIR" | tee "$BASELINE_LOG_FILE"

echo "🚀 [步骤 2/3] 正在批量评估所有子空间模型..."
# 关键修正：使用 mapfile 和 sort 来保证评估顺序稳定
mapfile -t CANDIDATE_MODELS < <(find "$HYBRID_MODEL_DIR" -name "subspace_ridge_r*_lam*_seed*.pt" | sort)

if [ ${#CANDIDATE_MODELS[@]} -eq 0 ]; then
    echo "警告: 在 '$HYBRID_MODEL_DIR' 中没有找到任何 'subspace_ridge' 模型。请先运行生成脚本。"
else
    for model_path in "${CANDIDATE_MODELS[@]}"; do
      model_filename=$(basename "$model_path" .pt)
      LOG_FILE="$LOG_DIR/benchmark_result_${model_filename}.log"
      echo "  [*] 正在评估模型: $model_filename ..."
      # 确保您的评估脚本使用确定性评估 (temperature=0)
      python evaluate_hybrid_model.py --model_path "$model_path" --data_dir "$DATA_DIR" | tee "$LOG_FILE"
      echo "  [✔] 评估完成: $model_filename"
    done
fi

echo "🚀 [步骤 3/3] 生成最终对比报告..."
echo ""
echo "=============================================================================="
echo "                      🏆  最终基准测试报告 (子空间岭回归) 🏆"
echo "=============================================================================="
printf "%-45s | %-12s | %-12s\n" "模型" "S1->S3" "Overall"
echo "--------------------------------------------------------------------------------"

BASELINE_S1S3=$(grep "S1->S3" "$BASELINE_LOG_FILE" | awk '{print $4}')
BASELINE_OVERALL=$(grep "Overall Accuracy" "$BASELINE_LOG_FILE" | awk '{print $3}')
printf "%-45s | %-12s | %-12s\n" "真实基线 (mix0)" "$BASELINE_S1S3" "$BASELINE_OVERALL"
printf "%-45s | %-12s | %-12s\n" "剂量效应最佳 (by_energy, k=5)" "76.00%" "92.00%"

if [ ${#CANDIDATE_MODELS[@]} -ne 0 ]; then
    for model_path in "${CANDIDATE_MODELS[@]}"; do
      model_filename=$(basename "$model_path" .pt)
      LOG_FILE="$LOG_DIR/benchmark_result_${model_filename}.log"
      if [ -f "$LOG_FILE" ]; then
        # 关键修正：awk 使用 $4 来提取 S1->S3 的准确率
        S1S3=$(grep "S1->S3" "$LOG_FILE" | awk '{print $4}')
        OVERALL=$(grep "Overall Accuracy" "$LOG_FILE" | awk '{print $3}')
        printf "%-45s | %-12s | %-12s\n" "$model_filename" "$S1S3" "$OVERALL"
      fi
    done
fi
echo "=============================================================================="