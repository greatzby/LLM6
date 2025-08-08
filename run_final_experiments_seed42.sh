#!/bin/bash

# ==============================================================================
#           精准嫁接最终决战：自动化实验脚本 (v1.1 - 已修正)
#
#  重要修正：已移除所有关于其他seed的测试，完全专注于 seed=42。
#
#  本脚本将系统性地执行以下实验：
#  1. 精细化搜索: 围绕已发现的甜点区 (projection, k=2) 微调 alpha 和 lambda。
#  2. 关键A/B测试: 验证 Procrustes 对齐在 projection 模式下是否真的有益。
# ==============================================================================

# 如果任何命令失败，脚本将立即退出
set -e

# 创建一个目录来存放所有的实验结果日志
LOG_DIR="experiment_logs"
mkdir -p $LOG_DIR
echo "所有实验结果将保存在 '$LOG_DIR' 目录中。"
echo "=============================================================================="

# --- 固定参数 ---
K_VAL=2
MODE="projection"
BASE_LAM=0.5
BASE_ALPHA=1.25
SEED=42

# ------------------------------------------------------------------------------
# 阶段一 (A): 围绕冠军配置精细化搜索 alpha
# 固定 k=2, lambda=0.5, seed=42，探索最佳的 alpha 值
# ------------------------------------------------------------------------------
echo "🚀 [阶段 1/3] 开始精细化搜索 alpha (k=2, lambda=0.5, seed=42)..."
for alpha in 1.10 1.15 1.20 1.30 1.35; do
    # 注意：为避免文件名冲突，我们使用更详细的命名
    MODEL_FILENAME="grafted_mode-${MODE}_k${K_VAL}_lam${BASE_LAM}_alpha${alpha}_seed${SEED}.pt"
    LOG_FILE="$LOG_DIR/results_k${K_VAL}_lam${BASE_LAM}_alpha${alpha}_seed${SEED}.log"
    echo "  [*] 正在测试 alpha = $alpha ..."
    
    python graft_advanced.py --k $K_VAL --lam $BASE_LAM --mode $MODE --alpha $alpha --seed $SEED --output_file "hybrid_models/${MODEL_FILENAME}"
    
    # 执行评估，并将结果同时输出到屏幕和日志文件
    python evaluate_hybrid_model.py --model_path "hybrid_models/${MODEL_FILENAME}" --data_dir data/simple_graph/composition_90 | tee $LOG_FILE
    
    echo "  [✔] 完成。结果已保存到 $LOG_FILE"
    echo "------------------------------------------------------------"
done

# ------------------------------------------------------------------------------
# 阶段一 (B): 围绕冠军配置精细化搜索 lambda
# 固定 k=2, alpha=1.25, seed=42，探索最佳的 lambda 值
# ------------------------------------------------------------------------------
echo "🚀 [阶段 2/3] 开始精细化搜索 lambda (k=2, alpha=1.25, seed=42)..."
for lam in 0.4 0.45 0.55 0.6; do
    MODEL_FILENAME="grafted_mode-${MODE}_k${K_VAL}_lam${lam}_alpha${BASE_ALPHA}_seed${SEED}.pt"
    LOG_FILE="$LOG_DIR/results_k${K_VAL}_lam${lam}_alpha${BASE_ALPHA}_seed${SEED}.log"
    echo "  [*] 正在测试 lambda = $lam ..."
    
    python graft_advanced.py --k $K_VAL --lam $lam --mode $MODE --alpha $BASE_ALPHA --seed $SEED --output_file "hybrid_models/${MODEL_FILENAME}"
    
    python evaluate_hybrid_model.py --model_path "hybrid_models/${MODEL_FILENAME}" --data_dir data/simple_graph/composition_90 | tee $LOG_FILE
    
    echo "  [✔] 完成。结果已保存到 $LOG_FILE"
    echo "------------------------------------------------------------"
done

# ------------------------------------------------------------------------------
# 阶段二: 关键 A/B 测试 - Procrustes 对齐是否有益？
# 使用冠军配置 (seed=42)，但关闭 Procrustes 对齐
# ------------------------------------------------------------------------------
echo "🚀 [阶段 3/3] 开始关键 A/B 测试 (Procrustes 对齐, seed=42)..."
MODEL_FILENAME="grafted_mode-${MODE}_k${K_VAL}_lam${BASE_LAM}_alpha${BASE_ALPHA}_seed${SEED}_NO_PROCRUSTES.pt"
LOG_FILE="$LOG_DIR/results_k${K_VAL}_lam${BASE_LAM}_alpha${BASE_ALPHA}_seed${SEED}_NO_PROCRUSTES.log"
echo "  [*] 正在测试冠军配置，但关闭 Procrustes 对齐..."

python graft_advanced.py --k $K_VAL --lam $BASE_LAM --mode $MODE --alpha $BASE_ALPHA --no_procrustes --seed $SEED --output_file "hybrid_models/${MODEL_FILENAME}"

python evaluate_hybrid_model.py --model_path "hybrid_models/${MODEL_FILENAME}" --data_dir data/simple_graph/composition_90 | tee $LOG_FILE

echo "  [✔] 完成。结果已保存到 $LOG_FILE"
echo "------------------------------------------------------------"


echo "🎉🎉🎉 所有 seed=42 的核心实验已全部执行完毕！ 🎉🎉🎉"
echo "请检查 '$LOG_DIR' 目录中的日志文件以分析结果。"