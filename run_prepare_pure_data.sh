#!/bin/bash
#
# =================================================================
#           run_prepare_pure_data.sh
#
#   一键式脚本，用于生成并预处理纯 S1->S3 组合数据集。
# =================================================================

# 如果任何命令失败，则立即退出
set -e

# --- 配置 ---
PURE_DATA_DIR="data/simple_graph/composition_90_pure"
PREPARE_SCRIPT="data/simple_graph/prepare_composition.py"

echo "====================================================="
echo "🚀 开始执行纯净数据集创建与预处理流程..."
echo "====================================================="

# --- 步骤 1: 生成纯 S1->S3 路径的 .txt 文件 ---
echo -e "\n[1/2] 正在生成纯 S1->S3 .txt 文件..."
python create_pure_s1s3_dataset.py --experiment_name "composition_90_pure"

# --- 步骤 2: 将 .txt 文件转换为 .bin 和 meta.pkl ---
echo -e "\n[2/2] 正在将 .txt 转换为 .bin 格式..."

# 为了让 prepare_composition.py 能处理我们的文件，我们临时重命名它们
# 以匹配脚本所期望的 "train_10.txt" 和 "test.txt" 格式。
mv "$PURE_DATA_DIR/train_pure.txt" "$PURE_DATA_DIR/train_10.txt"
mv "$PURE_DATA_DIR/val_pure.txt" "$PURE_DATA_DIR/test.txt"

# 调用您已有的预处理脚本，让它在我们新目录上工作
# 我们需要告诉它正确的目录名
BASE_DIR_ARG=$(basename "$PURE_DATA_DIR")
python "$PREPARE_SCRIPT" \
    --experiment_name "$BASE_DIR_ARG" \
    --total_nodes 90 \
    --train_paths_per_pair 10 # 这个参数现在只用于构成文件名，无实际意义

# 将生成的文件重命名为标准的 train.bin 和 val.bin
mv "$PURE_DATA_DIR/train_10.bin" "$PURE_DATA_DIR/train.bin"
# 注意：您的 prepare_composition.py 将验证集输出为 val.bin，所以这里不需要移动
# mv "$PURE_DATA_DIR/test.bin" "$PURE_DATA_DIR/val.bin" # 如果您的脚本输出 test.bin 则需要这行

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