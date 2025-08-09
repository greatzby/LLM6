#!/bin/bash
set -e

# ==============================================================================
#                  自动化诊断与修正执行脚本
# ==============================================================================

echo "脚本开始：将按顺序执行健全性检查、新模型生成和提示评估。"

# --- 步骤 0: 定义通用变量 ---
HOST_MODEL="out_d92/composition_mix0_seed42_20250801_054758/ckpt_mix0_seed42_iter50000.pt"
DONOR_MODEL="out_d92/composition_mix20_seed42_20250801_064928/ckpt_mix20_seed42_iter50000.pt"
DATA_PATH="data/simple_graph/composition_90/train_10.bin"

# ==============================================================================
#                  阶段一：健全性检查 (Sanity Checks)
# ==============================================================================
echo ""
echo "======================================================="
echo "阶段一：开始健全性检查..."
echo "======================================================="

# --- 1.A: 生成凸组合基线模型 (用于验证评估流程) ---
echo ""
echo "--> 正在生成凸组合基线模型 (gamma=0.1, 0.2)..."
python create_convex_baseline.py \
  --host_model_path "$HOST_MODEL" \
  --donor_model_path "$DONOR_MODEL" \
  --gamma 0.1

python create_convex_baseline.py \
  --host_model_path "$HOST_MODEL" \
  --donor_model_path "$DONOR_MODEL" \
  --gamma 0.2
echo "--> 凸组合模型已生成于 hybrid_models_convex/ 目录。"

# --- 1.B: 对失败的旧方法进行诊断 (不保存模型，只看输出) ---
echo ""
echo "--> 正在对旧的子空间岭回归方法进行诊断 (r=5, lam=0.1)..."
python create_subspace_grafted_model_with_diagnostics.py \
  --host_model_path "$HOST_MODEL" \
  --donor_model_path "$DONOR_MODEL" \
  --data_path "$DATA_PATH" \
  --rank 5 \
  --lam 0.1
echo "--> 诊断信息已打印到上方终端。"


# ==============================================================================
#                  阶段二：生成新的『恒等增量』映射模型
# ==============================================================================
echo ""
echo "======================================================="
echo "阶段二：开始批量生成新的 v2 映射模型..."
echo "======================================================="

# 确保 run_mapping_v2.sh 是可执行的
chmod +x run_mapping_v2.sh

# 运行脚本
./run_mapping_v2.sh

echo "--> 新的 v2 映射模型已生成于 hybrid_models_mapping_v2/ 目录。"


# ==============================================================================
#                  阶段三：评估提示
# ==============================================================================
echo ""
echo "======================================================="
echo "🎉 所有模型生成完毕！"
echo "======================================================="
echo ""
echo "下一步行动："
echo "1. 请检查您的评估脚本 (例如 run_final_benchmark_v2.2.sh)。"
echo "2. 确保它会评估以下目录中的模型："
echo "   - hybrid_models (旧方法，用于对比)"
echo "   - hybrid_models_convex (健全性检查)"
echo "   - hybrid_models_mapping_v2 (新方法的结果)"
echo "3. 然后手动运行评估脚本："
echo ""
echo "   ./run_final_benchmark_v2.2.sh"
echo ""
echo "======================================================="