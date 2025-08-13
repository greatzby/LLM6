#!/bin/bash
#
# =================================================================
#           run_final_experiment.sh (v2.1 - 最终修复版)
#
#   一键式执行最终实验的完整流程。
#   v2.1 修复: 将 PURE_DATA_DIR 的路径修正为与数据生成脚本完全一致。
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

# <<< 核心修复：将此处的目录名与您成功创建的目录名保持一致 >>>
PURE_DATA_DIR="data/simple_graph/composition_pure_90" 

# 评估时必须使用包含原始 test.txt 的数据目录
EVAL_DATA_DIR="data/simple_graph/composition_90" 

# 3. 定义脚本名称
FINETUNE_SCRIPT="finetune_specialized_v2.py"
EVAL_SCRIPT="evaluate_hybrid_model.py"

# =================================================================
#   阶段 1: 检查纯净的 S1->S3 数据集
# =================================================================
echo "====================================================="
echo "✅ 阶段 1: 检查纯净数据集"
echo "====================================================="

if [ -d "$PURE_DATA_DIR" ] && [ -f "$PURE_DATA_DIR/train.bin" ]; then
    echo "[*] 纯净数据集 '$PURE_DATA_DIR' 已找到，准备进行微调。"
else
    echo "[!] 致命错误: 在 '$PURE_DATA_DIR' 中未找到纯净数据集。"
    echo "[!] 请先成功运行 'run_prepare_pure_data.sh' 脚本。"
    exit 1
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