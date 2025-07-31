#!/bin/bash

# 设置脚本在遇到任何错误时立即退出
set -e

# 打印开始信息
echo "================================================="
echo "🚀 开始执行实验套件，共4个任务..."
echo "================================================="

# --- 任务 1/4 ---
echo ""
echo "--- [1/4] 正在运行: 0% mix, 92 embed, seed 456 ---"
python trainfour.py \
    --data_dir data/simple_graph/composition_90 \
    --n_embd 92 \
    --mixing_ratio 0 \
    --seed 456

# --- 任务 2/4 ---
echo ""
echo "--- [2/4] 正在运行: 20% mix, 92 embed, seed 42 ---"
python trainfour.py \
    --data_dir data/simple_graph/composition_90_mixed_20 \
    --n_embd 92 \
    --mixing_ratio 20 \
    --seed 42

# --- 任务 3/4 ---
echo ""
echo "--- [3/4] 正在运行: 20% mix, 92 embed, seed 123 ---"
python trainfour.py \
    --data_dir data/simple_graph/composition_90_mixed_20 \
    --n_embd 92 \
    --mixing_ratio 20 \
    --seed 123

# --- 任务 4/4 ---
echo ""
echo "--- [4/4] 正在运行: 20% mix, 92 embed, seed 456 ---"
python trainfour.py \
    --data_dir data/simple_graph/composition_90_mixed_20 \
    --n_embd 92 \
    --mixing_ratio 20 \
    --seed 456

# 打印结束信息
echo ""
echo "================================================="
echo "🎉🎉🎉 全部4个实验均已成功执行完毕！ 🎉🎉🎉"
echo "================================================="