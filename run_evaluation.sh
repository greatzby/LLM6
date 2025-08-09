#!/bin/bash
#
# run_evaluation.sh
# 一键式评估脚本，用于测试所有微调后的模型的最终性能。

# 如果任何命令执行时返回非零退出码（即发生错误），则立即退出脚本
set -e

# --- 配置 ---
# 微调后模型所在的目录
FINETUNED_DIR="finetuned_models"
# 包含 test.txt, meta.pkl 等评估数据的目录
# 我们使用与微调时相同的数据源，以确保评估的一致性
DATA_DIR="data/simple_graph/composition_90_mixed_20"

echo "====================================================="
echo "🚀 开始执行一键式模型评估任务..."
echo "====================================================="
echo "将评估目录 '$FINETUNED_DIR' 中的所有 .pt 模型"
echo "使用数据源: $DATA_DIR"
echo

# 检查 finetuned_models 目录是否存在
if [ ! -d "$FINETUNED_DIR" ]; then
    echo "错误: 目录 '$FINETUNED_DIR' 未找到。请先运行微调脚本。"
    exit 1
fi

# 循环遍历 finetuned_models 目录下的每一个 .pt 文件
for model_path in "$FINETUNED_DIR"/*.pt; do
    # 检查是否有文件匹配，如果没有则跳出循环
    [ -e "$model_path" ] || continue

    echo "--- 正在评估模型: $(basename "$model_path") ---"
    
    # 调用您提供的评估脚本
    python evaluate_hybrid_model.py --model_path "$model_path" --data_dir "$DATA_DIR"
    
    echo "--- 评估完成 ---"
    echo
done

echo "====================================================="
echo "✅ 所有评估任务已成功完成！"
echo "====================================================="