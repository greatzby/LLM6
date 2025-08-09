#!/bin/bash
#
# run_finetune.sh
# 一键式微调脚本，用于对挑选出的模型进行 lm_head 微调

# 如果任何命令执行时返回非零退出码（即发生错误），则立即退出脚本
set -e

echo "====================================================="
echo "🚀 开始执行一键式 lm_head 微调任务..."
echo "====================================================="

# 使用一个数组来清晰地管理所有待微调的模型路径
# 这样未来您可以非常方便地添加或删除模型
MODELS_TO_TUNE=(
    "hybrid_models/grafted_mode-projection_k1_lam0.70_seed42.pt"
    "hybrid_models/convex_gamma0.1.pt"
    "hybrid_models/grafted_advanced_lam0.00_seed42.pt"
    "hybrid_models/map_v2_lowrank_r5_lam0.01_seed42_train_10.pt"
    "hybrid_models/subspace_ridge_r4_lam0.01_seed42.pt"
    "hybrid_models/grafted_advanced_lam0.30_seed42.pt"
)

echo
echo "将对以下 ${#MODELS_TO_TUNE[@]} 个模型进行微调："
# 循环打印出每个模型的名字，让用户在开始前可以确认
for model in "${MODELS_TO_TUNE[@]}"; do
    echo "  - $model"
done
echo # 打印一个空行，用于格式美化

# 执行核心的 Python 微调脚本
# "${MODELS_TO_TUNE[@]}" 会被 Bash 正确地展开为所有模型路径，作为参数列表传递给 Python 脚本
python finetune_lm_head.py --model_paths "${MODELS_TO_TUNE[@]}"

echo
echo "====================================================="
echo "✅ 所有微调任务已成功完成！"
echo "修复后的模型已保存在 'finetuned_models/' 目录下。"
echo "现在您可以使用 eval_hybrid.py 进行最终评估。"
echo "====================================================="