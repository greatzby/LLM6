#!/bin/bash
#
# =================================================================
#           run_prepare_pure_data.sh (v1.1 - 已修复路径问题)
#
#   一键式脚本，用于生成并预处理纯 S1->S3 组合数据集。
#   v1.1 修复: 正确调用 prepare_composition.py，避免路径拼接错误。
# =================================================================

# 如果任何命令失败，则立即退出
set -e

# --- 配置 ---
# (修复) 我们将基础名称和最终目录名分开定义
BASE_NAME="composition_pure"
NODE_COUNT=90
FULL_EXP_NAME="${BASE_NAME}_${NODE_COUNT}" # 这将是 "composition_pure_90"

PURE_DATA_DIR="data/simple_graph/${FULL_EXP_NAME}"
PREPARE_SCRIPT="data/simple_graph/prepare_composition.py"

echo "====================================================="
echo "🚀 开始执行纯净数据集创建与预处理流程 (v1.1)..."
echo "====================================================="

# --- 步骤 1: 生成纯 S1->S3 路径的 .txt 文件 ---
echo -e "\n[1/2] 正在生成纯 S1->S3 .txt 文件..."
# (修复) 我们告诉创建脚本使用新的、正确的目录名
python create_pure_s1s3_dataset.py --experiment_name "$FULL_EXP_NAME"

# --- 步骤 2: 将 .txt 文件转换为 .bin 和 meta.pkl ---
echo -e "\n[2/2] 正在将 .txt 转换为 .bin 格式..."

# 为了让 prepare_composition.py 能处理我们的文件，我们临时重命名它们
# 以匹配脚本所期望的 "train_10.txt" 和 "test.txt" 格式。
mv "$PURE_DATA_DIR/train_pure.txt" "$PURE_DATA_DIR/train_10.txt"
mv "$PURE_DATA_DIR/val_pure.txt" "$PURE_DATA_DIR/test.txt"

# (核心修复)
# 现在我们给 prepare_composition.py 传递它所期望的“基础名称”，
# 让它自己去拼接 `_90`，这样它就能找到正确的目录了。
echo "[*] 调用 prepare_composition.py, experiment_name='${BASE_NAME}', total_nodes='${NODE_COUNT}'"
python "$PREPARE_SCRIPT" \
    --experiment_name "$BASE_NAME" \
    --total_nodes "$NODE_COUNT" \
    --train_paths_per_pair 10

# 将生成的文件重命名为标准的 train.bin
# 注意：您的 prepare_composition.py 会在正确的目录直接生成 train_10.bin 和 val.bin
mv "$PURE_DATA_DIR/train_10.bin" "$PURE_DATA_DIR/train.bin"

# 清理临时的 .txt 文件
rm "$PURE_DATA_DIR/train_10.txt"
rm "$PURE_DATA_DIR/test.txt"

echo
echo "====================================================="
echo "✅ 所有任务已成功完成！"
echo "纯净的、可用于微调的数据集已准备就绪于:"
echo "   ---> $PURE_DATA_DIR"
echo "现在您可以运行最终的实验脚本了。"
echo "====================================================="