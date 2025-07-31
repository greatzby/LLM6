#!/bin/bash

# 设置脚本在遇到任何错误时立即退出
set -e

# 打印开始信息
echo "================================================="
echo "🚀 开始执行最终实验套件，共6个任务..."
echo "================================================="

# --- 组 1: 嵌入维度 n_embd = 92 ---
echo ""
echo "--- [1/6] 正在运行: 0% mix, 92 embed, seed 456 ---"
python train_final_suite.py \
    --data_dir data/simple_graph/composition_90 \
    --n_embd 92 \
    --mixing_ratio 0 \
    --seed 456

echo ""
echo "--- [2/6] 正在运行: 20% mix, 92 embed, seed 42 ---"
python train_final_suite.py \
    --data_dir data/simple_graph/composition_90_mixed_20 \
    --n_embd 92 \
    --mixing_ratio 20 \
    --seed 42

echo ""
echo "--- [3/6] 正在运行: 20% mix, 92 embed, seed 123 ---"
python train_final_suite.py \
    --data_dir data/simple_graph/composition_90_mixed_20 \
    --n_embd 92 \
    --mixing_ratio 20 \
    --seed 123

# --- 组 2: 嵌入维度 n_embd = 120 ---
echo ""
echo "--- [4/6] 正在运行: 0% mix, 120 embed, seed 456 ---"
python train_final_suite.py \
    --data_dir data/simple_graph/composition_90 \
    --n_embd 120 \
    --mixing_ratio 0 \
    --seed 456

echo ""
echo "--- [5/6] 正在运行: 20% mix, 120 embed, seed 42 ---"
python train_final_suite.py \
    --data_dir data/simple_graph/composition_90_mixed_20 \
    --n_embd 120 \
    --mixing_ratio 20 \
    --seed 42

echo ""
echo "--- [6/6] 正在运行: 20% mix, 120 embed, seed 123 ---"
python train_final_suite.py \
    --data_dir data/simple_graph/composition_90_mixed_20 \
    --n_embd 120 \
    --mixing_ratio 20 \
    --seed 123

# 打印结束信息
echo ""
echo "================================================="
echo "🎉🎉🎉 全部6个实验均已成功执行完毕！ 🎉🎉🎉"
echo "================================================="