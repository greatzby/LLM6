#!/bin/bash
set -e

# ==============================================================================
#                  最终基准测试脚本 (v2.2)
#   自动发现、评估并报告所有模型变体的性能
# ==============================================================================

# 这是一个辅助函数，用于评估单个模型并格式化输出
# 它会运行 python 评估脚本，然后用 grep 和 awk 提取关键结果
evaluate_model() {
    local model_path=$1
    local display_name=$2

    # 运行评估并捕获输出
    # 假设 eval_simple.py 会打印 "S1->S3: XX.XX%" 和 "Overall: YY.YY%" 这样的行
    local result
    result=$(python eval_simple.py --model_path "$model_path")

    # 解析 S1->S3 和 Overall 的准确率
    local s1_s3
    s1_s3=$(echo "$result" | grep 'S1->S3' | awk '{print $2}')
    local overall
    overall=$(echo "$result" | grep 'Overall' | awk '{print $2}')

    # 如果由于某种原因解析失败，则使用 "N/A" 作为备用
    s1_s3=${s1_s3:-"N/A"}
    overall=${overall:-"N/A"}

    # 打印格式化的表格行
    printf "%-55s | %-12s | %s\n" "$display_name" "$s1_s3" "$overall"
}

# ==============================================================================
#                            脚本主程序
# ==============================================================================

# 打印报告标题
echo "=============================================================================="
echo "                      🏆  最终基准测试报告 (v2.2) 🏆"
echo "=============================================================================="

# 打印表头
printf "%-55s | %-12s | %s\n" "模型" "S1->S3" "Overall"
echo "------------------------------------------------------------------------------"

# --- 评估基线模型 ---
HOST_MODEL_CKPT="out_d92/composition_mix0_seed42_20250801_054758/ckpt_mix0_seed42_iter50000.pt"
evaluate_model "$HOST_MODEL_CKPT" "真实基线 (mix0)"

# 为了方便对比，我们在这里硬编码之前的最佳结果
printf "%-55s | %-12s | %s\n" "剂量效应最佳 (by_energy, k=5)" "76.00%" "92.00%"

echo "------------------------------------------------------------------------------"

# --- 自动发现并评估所有生成的模型 ---
# 定义需要搜索的所有模型目录
MODEL_DIRS=("hybrid_models" "hybrid_models_convex" "hybrid_models_mapping_v2")

for dir in "${MODEL_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        # 使用 find 找到所有 .pt 模型文件，并用 sort 保证顺序一致
        find "$dir" -name "*.pt" | sort | while read -r model_file; do
            # 从文件路径中提取一个干净的显示名称
            display_name=$(basename "$model_file" .pt)
            evaluate_model "$model_file" "$display_name"
        done
    fi
done

echo "=============================================================================="
echo "                             报告结束"
echo "=============================================================================="