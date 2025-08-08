#!/bin/bash

# ==============================================================================
#           最终冲刺：自动化实验脚本 (v2.0 - 已根据最新洞察修正)
#
#  核心修正：
#  1. 确认 Procrustes 在 projection 模式下无效，所有实验默认添加 --no_procrustes。
#  2. 聚焦于对 alpha 在 1.30 峰值附近的精细化扫描。
#  3. 将多 seed 验证作为可选的、注释掉的步骤。
# ==============================================================================

# 如果任何命令失败，脚本将立即退出
set -e

LOG_DIR="experiment_logs_sprint"
mkdir -p $LOG_DIR
echo "所有最终冲刺实验结果将保存在 '$LOG_DIR' 目录中。"
echo "=============================================================================="

# --- 固定参数 ---
K_VAL=2
LAM_VAL=0.5
MODE="projection"
BASE_SEED=42

# ------------------------------------------------------------------------------
# 阶段一: 在 alpha 峰值附近进行精细化扫描 (seed=42)
# ------------------------------------------------------------------------------
echo "🚀 [阶段 1/2] 开始精细化扫描 alpha (k=2, λ=0.5, no_procrustes)..."
for alpha in 1.26 1.28 1.30 1.32 1.34; do
    MODEL_FILENAME="grafted_mode-${MODE}_k${K_VAL}_lam${LAM_VAL}_alpha${alpha}_seed${BASE_SEED}_NO_PROCRUSTES.pt"
    LOG_FILE="$LOG_DIR/results_k${K_VAL}_lam${LAM_VAL}_alpha${alpha}_seed${BASE_SEED}_NO_PROCRUSTES.log"
    echo "  [*] 正在测试 alpha = $alpha ..."
    
    # 执行嫁接，注意我们现在总是添加 --no_procrustes
    python graft_advanced.py \
        --k $K_VAL \
        --lam $LAM_VAL \
        --mode $MODE \
        --alpha $alpha \
        --seed $BASE_SEED \
        --no_procrustes \
        --output_file "hybrid_models/${MODEL_FILENAME}"
    
    # 执行评估
    python evaluate_hybrid_model.py \
        --model_path "hybrid_models/${MODEL_FILENAME}" \
        --data_dir data/simple_graph/composition_90 | tee $LOG_FILE
    
    echo "  [✔] 完成。结果已保存到 $LOG_FILE"
    echo "------------------------------------------------------------"
done

# ------------------------------------------------------------------------------
# 阶段二: 稳健性验证 (多 seed 复核 - 可选)
# 如果您有其他种子的 ckpt，请取消下面的注释并运行
# ------------------------------------------------------------------------------
echo "🚀 [阶段 2/2] 稳健性验证（可选）。"
echo "    如果需要，请编辑此脚本，取消下面部分的注释。"

# --- 取消下面的注释以在其他种子(例如41, 43)上验证最佳alpha ---
#
# BEST_ALPHA=1.30 # 假设 1.30 是第一阶段的最佳值，请根据实际结果修改
# for seed in 41 43; do
#     echo "  [*] 正在使用 seed=$seed 验证最佳 alpha=$BEST_ALPHA ..."
#     MODEL_FILENAME="grafted_mode-${MODE}_k${K_VAL}_lam${LAM_VAL}_alpha${BEST_ALPHA}_seed${seed}_NO_PROCRUSTES.pt"
#     LOG_FILE="$LOG_DIR/results_k${K_VAL}_lam${LAM_VAL}_alpha${BEST_ALPHA}_seed${seed}_NO_PROCRUSTES.log"
# 
#     python graft_advanced.py \
#         --k $K_VAL \
#         --lam $LAM_VAL \
#         --mode $MODE \
#         --alpha $BEST_ALPHA \
#         --seed $seed \
#         --no_procrustes \
#         --output_file "hybrid_models/${MODEL_FILENAME}"
# 
#     python evaluate_hybrid_model.py \
#         --model_path "hybrid_models/${MODEL_FILENAME}" \
#         --data_dir data/simple_graph/composition_90 | tee $LOG_FILE
# 
#     echo "  [✔] seed=$seed 的验证完成。结果已保存到 $LOG_FILE"
#     echo "------------------------------------------------------------"
# done


echo "🎉🎉🎉 所有高优先级实验已全部执行完毕！ 🎉🎉🎉"
echo "请检查 '$LOG_DIR' 目录中的日志文件，以确定最终的冠军配置！"