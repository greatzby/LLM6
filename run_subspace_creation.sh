#!/bin/bash
set -e

# --- 配置区 ---
# 请务必在此处核对您的模型路径！
HOST_MODEL="out_d92/composition_mix0_seed42_20250801_054758/ckpt_mix0_seed42_iter50000.pt"
DONOR_MODEL="out_d92/composition_mix20_seed42_20250801_064928/ckpt_mix20_seed42_iter50000.pt"

# --- 关键修正：在这里明确指定完整的数据文件路径 ---
DATA_PATH="data/simple_graph/composition_90/train_10.bin"

OUTPUT_DIR="hybrid_models"
SEED=42

# 围绕我们已知的甜点 k=5 进行搜索
RANKS=(4 5 6 7 8) 
# 测试几个数量级的正则化强度
LAMBDAS=(0.01 0.1 1.0) 

echo "======================================================="
echo "🚀 开始通过子空间岭回归批量生成混合模型..."
echo "======================================================="

# 检查数据文件是否存在
if [ ! -f "$DATA_PATH" ]; then
    echo "错误: 数据文件未找到于 '$DATA_PATH'"
    echo "请在脚本中设置正确的 DATA_PATH 变量。"
    exit 1
fi

for r in "${RANKS[@]}"; do
  for l in "${LAMBDAS[@]}"; do
    echo ""
    echo "--- 正在生成模型: Rank(r)=$r, Lambda(λ)=$l ---"
    
    # --- 关键修正：使用 --data_path 传递完整路径 ---
    python create_subspace_grafted_model.py \
      --host_model_path "$HOST_MODEL" \
      --donor_model_path "$DONOR_MODEL" \
      --data_path "$DATA_PATH" \
      --output_dir "$OUTPUT_DIR" \
      --rank "$r" \
      --lam "$l" \
      --seed "$SEED"
      
    echo "  [✔] 模型生成完毕: subspace_ridge_r${r}_lam${l}_seed${SEED}.pt"
  done
done

echo ""
echo "🎉🎉🎉 所有模型已生成完毕！下一步请运行评估脚本。🎉🎉🎉"