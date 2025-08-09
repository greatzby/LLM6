#!/bin/bash

# --- 配置 ---
HOST_MODEL="out_d92/composition_mix0_seed42_20250801_054758/ckpt_mix0_seed42_iter50000.pt"
DONOR_MODEL="out_d92/composition_mix20_seed42_20250801_054758/ckpt_mix20_seed42_iter50000.pt"
DATA_DIR="data/simple_graph"
OUTPUT_DIR="hybrid_models"
SEED=42

# --- 网格搜索参数 ---
# 围绕我们已知的甜点 k=5 进行搜索
RANKS=(4 5 6 7 8) 
# 测试几个数量级的正则化强度
LAMBDAS=(0.01 0.1 1.0) 

echo "======================================================="
echo "🚀 开始通过子空间岭回归批量生成混合模型..."
echo "======================================================="

for r in "${RANKS[@]}"; do
  for l in "${LAMBDAS[@]}"; do
    echo ""
    echo "--- 正在生成模型: Rank(r)=$r, Lambda(λ)=$l ---"
    
    python create_subspace_grafted_model.py \
      --host_model_path "$HOST_MODEL" \
      --donor_model_path "$DONOR_MODEL" \
      --data_dir "$DATA_DIR" \
      --output_dir "$OUTPUT_DIR" \
      --rank "$r" \
      --lam "$l" \
      --seed "$SEED"
      
    echo "  [✔] 模型生成完毕: subspace_ridge_r${r}_lam${l}_seed${SEED}.pt"
  done
done

echo ""
echo "🎉🎉🎉 所有模型已生成完毕！下一步请运行评估脚本。🎉🎉🎉"