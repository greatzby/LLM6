#!/bin/bash
#
# =================================================================
#           run_final_experiment.sh (最终版)
#
#   一键式执行最终实验的完整流程：
#   1. 准备纯净的 S1->S3 数据集 (如果不存在)
#   2. 对指定的4个模型进行专项 lm_head 微调
#   3. 对微调后的4个模型进行评估
# =================================================================

# 如果任何命令失败，则立即退出
set -e

# --- 脚本配置 ---
# 1. 定义您要处理的4个模型
SOURCE_MODELS=(
    "grafted_mode-projection_k1_lam0.70_seed42.pt"
    "grafted_advanced_lam0.00_seed42.pt"
    "subspace_ridge_r4_lam0.01_seed42.pt"
    "grafted_advanced_lam0.30_seed42.pt"
)

# 2. 定义目录
HYBRID_DIR="hybrid_models"
FINETUNED_DIR="finetuned_specialized_models"
PURE_DATA_DIR="data/simple_graph/composition_90_pure"
# !! 关键 !!: 评估时必须使用包含原始 test.txt 的数据目录
EVAL_DATA_DIR="data/simple_graph/composition_90" 

# 3. 定义脚本名称
FINETUNE_SCRIPT="finetune_specialized_v2.py"
EVAL_SCRIPT="evaluate_hybrid_model.py" # 使用您自己的评估脚本

# =================================================================
#   阶段 1: 准备纯净的 S1->S3 数据集
# =================================================================
echo "====================================================="
echo "✅ 阶段 1: 准备纯净数据集"
echo "====================================================="

if [ -f "$PURE_DATA_DIR/train.bin" ] && [ -f "$PURE_DATA_DIR/val.bin" ]; then
    echo "[*] 纯净数据集 '$PURE_DATA_DIR' 已存在，跳过数据生成。"
else
    echo "[*] 未找到纯净数据集，现在开始创建..."
    # (此处省略数据创建脚本，假设您已通过之前步骤创建)
    # 如果需要，请先运行 `create_pure_s1s3_dataset.py` 和 `prepare_composition.py`
    # 来生成 $PURE_DATA_DIR 目录及其中的 .bin 文件。
    echo "[!] 警告: 纯净数据集未找到。请手动创建或取消此检查。"
    # exit 1 # 如果希望在没有数据时强制停止，请取消此行的注释
fi


# =================================================================
#   阶段 2: 对4个模型进行专项微调
# =================================================================
echo -e "\n====================================================="
echo "✅ 阶段 2: 专项微调 (使用 ${FINETUNE_SCRIPT})"
echo "====================================================="

mkdir -p "$FINETUNED_DIR"

for model_name in "${SOURCE_MODELS[@]}"; do
    source_path="$HYBRID_DIR/$model_name"
    target_path="$FINETUNED_DIR/${model_name%.pt}_finetuned.pt"
    
    if [ ! -f "$source_path" ]; then
        echo "[!] 警告: 源模型未找到，跳过: $source_path"
        continue
    fi
    
    python "$FINETUNE_SCRIPT" \
        --source_model_path "$source_path" \
        --output_model_path "$target_path" \
        --dataset_dir "$PURE_DATA_DIR"
done

# =================================================================
#   阶段 3: 评估微调后的模型
# =================================================================
echo -e "\n====================================================="
echo "✅ 阶段 3: 评估微调后的模型 (使用 ${EVAL_SCRIPT})"
echo "====================================================="

if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "[!] 致命错误: 评估脚本 '$EVAL_SCRIPT' 未找到。无法继续。"
    exit 1
fi

for model_name in "${SOURCE_MODELS[@]}"; do
    finetuned_model_path="$FINETUNED_DIR/${model_name%.pt}_finetuned.pt"
    
    if [ ! -f "$finetuned_model_path" ]; then
        echo "[!] 警告: 微调后的模型未找到，无法评估: $finetuned_model_path"
        continue
    fi
    
    echo -e "\n\n--- 正在评估: $finetuned_model_path ---"
    python "$EVAL_SCRIPT" \
        --model_path "$finetuned_model_path" \
        --data_dir "$EVAL_DATA_DIR" \
        --temperature 0.0 \
        --top_k 1
done

echo -e "\n====================================================="
echo "🎉🎉🎉 所有任务已成功完成！ 🎉🎉🎉"
echo "专项微调后的模型已保存在 '$FINETUNED_DIR' 目录下。"
echo "最终评估结果已显示在上方。"
echo "====================================================="