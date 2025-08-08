#!/bin/bash

# ==============================================================================
#           最终基准测试：一键式确定性评估脚本 (v2)
#
#  目标：使用确定性评估脚本，重新测量基线和最佳候选模型的真实性能，
#        并生成最终的对比报告。
#
#  运行此脚本前，请确保:
#  1. evaluate_hybrid_model.py 已更新为确定性版本。
#  2. model.py 中的 generate 函数已修复，可以处理 temperature = 0。
# ==============================================================================

# 如果任何命令失败，脚本将立即退出
set -e

# --- 配置 ---
LOG_DIR="final_benchmark_logs" # 使用新的日志目录名以避免混淆
BASELINE_MODEL_PATH="out_d92/composition_mix0_seed42_20250801_054758/ckpt_mix0_seed42_iter50000.pt"
DATA_DIR="data/simple_graph/composition_90"
CANDIDATE_ALPHAS=("1.28" "1.30")

# --- 脚本开始 ---
mkdir -p $LOG_DIR
echo "=============================================================================="
echo "    📊  开始执行最终基准测试：一键式确定性评估  📊"
echo "    所有日志将保存在 '$LOG_DIR' 目录中。"
echo "=============================================================================="

# --- 步骤 1: 重新测量真实的基线性能 ---
echo "🚀 [步骤 1/3] 正在测量真实的基线模型性能 (使用确定性评估)..."
BASELINE_LOG_FILE="$LOG_DIR/benchmark_result_BASELINE.log"

# 检查基线模型文件是否存在
if [ ! -f "$BASELINE_MODEL_PATH" ]; then
    echo "❌ 错误: 基线模型文件不存在! 路径: $BASELINE_MODEL_PATH"
    exit 1
fi

python evaluate_hybrid_model.py \
  --model_path "$BASELINE_MODEL_PATH" \
  --data_dir "$DATA_DIR" | tee "$BASELINE_LOG_FILE"

echo "  [✔] 真实基线测量完成。日志已保存到 $BASELINE_LOG_FILE"
echo "------------------------------------------------------------------------------"


# --- 步骤 2: 重新测量冠军候选者的性能 ---
echo "🚀 [步骤 2/3] 正在测量冠军候选模型的真实性能..."
for alpha in "${CANDIDATE_ALPHAS[@]}"; do
    MODEL_FILENAME="hybrid_models/grafted_mode-projection_k2_lam0.5_alpha${alpha}_seed42_NO_PROCRUSTES.pt"
    LOG_FILE="$LOG_DIR/benchmark_result_alpha_${alpha}.log"
    
    echo "  [*] 正在评估 Alpha = $alpha ..."

    # 检查候选模型文件是否存在
    if [ ! -f "$MODEL_FILENAME" ]; then
        echo "  [!] 警告: 模型文件不存在，跳过评估: $MODEL_FILENAME"
        # 创建一个空的日志文件，以防报告部分出错
        echo "Model file not found." > "$LOG_FILE"
        continue
    fi

    python evaluate_hybrid_model.py \
      --model_path "$MODEL_FILENAME" \
      --data_dir "$DATA_DIR" | tee "$LOG_FILE"
      
    echo "  [✔] Alpha = $alpha 评估完成。日志已保存到 $LOG_FILE"
done
echo "------------------------------------------------------------------------------"


# --- 步骤 3: 生成并打印最终对比报告 ---
echo "🚀 [步骤 3/3] 生成最终对比报告..."
echo ""
echo "=============================================================================="
echo "                      🏆  最终基准测试报告 (确定性结果) 🏆"
echo "=============================================================================="
echo ""
echo "配置: k=2, λ=0.5, mode=projection, seed=42, no_procrustes"
echo ""
printf "%-15s | %-15s | %-15s\n" "模型" "S1->S3 Acc." "Overall Acc."
echo "--------------------------------------------------------------"

# 提取并打印基线结果
BASELINE_S1S3=$(grep "S1->S3" "$BASELINE_LOG_FILE" | awk '{print $3}' || echo "N/A")
BASELINE_OVERALL=$(grep "Overall" "$BASELINE_LOG_FILE" | awk '{print $3}' || echo "N/A")
printf "%-15s | %-15s | %-15s\n" "真实基线" "$BASELINE_S1S3" "$BASELINE_OVERALL"

# 提取并打印候选者结果
for alpha in "${CANDIDATE_ALPHAS[@]}"; do
    LOG_FILE="$LOG_DIR/benchmark_result_alpha_${alpha}.log"
    if [ -f "$LOG_FILE" ]; then
        S1S3_ACC=$(grep "S1->S3" "$LOG_FILE" | awk '{print $3}' || echo "N/A")
        OVERALL_ACC=$(grep "Overall" "$LOG_FILE" | awk '{print $3}' || echo "N/A")
        printf "%-15s | %-15s | %-15s\n" "Alpha = $alpha" "$S1S3_ACC" "$OVERALL_ACC"
    fi
done
echo "=============================================================================="
echo ""
echo "🎉🎉🎉 所有基准测试已完成！以上为最可信的最终结果。 🎉🎉🎉"