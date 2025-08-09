#!/bin/bash
set -e

HOST_MODEL="out_d92/composition_mix0_seed42_20250801_054758/ckpt_mix0_seed42_iter50000.pt"
DONOR_MODEL="out_d92/composition_mix20_seed42_20250801_064928/ckpt_mix20_seed42_iter50000.pt"
DATA_PATH="data/simple_graph/composition_90/train_10.bin"
OUTPUT_DIR="hybrid_models_mapping_v2"
SEED=42

# 推荐的超参数网格搜索
RANKS=(4 5 6 7 8) 
LAMBDAS=(0.01 0.1 1.0 3.0 10.0) 

echo "======================================================="
echo "🚀 开始通过『恒等增量』隐状态映射生成混合模型..."
echo "======================================================="

for r in "${RANKS[@]}"; do
  for l in "${LAMBDAS[@]}"; do
    echo ""
    echo "--- 正在生成模型: Rank(r)=$r, Lambda(λ)=$l ---"
    
    python create_mapping_v2_lowrank.py \
      --host_model_path "$HOST_MODEL" \
      --donor_model_path "$DONOR_MODEL" \
      --data_path "$DATA_PATH" \
      --output_dir "$OUTPUT_DIR" \
      --rank "$r" \
      --lam "$l" \
      --seed "$SEED"
  done
done

echo ""
echo "🎉🎉🎉 所有模型已生成完毕！请更新并运行评估脚本。🎉🎉🎉"