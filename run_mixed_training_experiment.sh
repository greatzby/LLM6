#!/bin/bash
# run_mixed_training_experiment.sh

echo "=========================================="
echo "混合训练实验流程"
echo "=========================================="

# 1. 生成实验报告
echo -e "\n[Step 1] 生成当前实验报告..."
python generate_experiment_report.py

# 2. 创建混合数据集（5% S1->S3）
echo -e "\n[Step 2] 创建5%混合数据集..."
python create_mixed_composition_dataset.py \
    --original_dir data/simple_graph/composition_90 \
    --s1_s3_ratio 0.05

# 3. 准备混合数据的二进制文件
echo -e "\n[Step 3] 准备混合数据..."
cd data/simple_graph
python prepare_composition.py \
    --experiment_name composition_mixed_5 \
    --total_nodes 90 \
    --train_paths_per_pair 10
cd ../..

# 4. 训练混合模型
echo -e "\n[Step 4] 开始混合训练..."
python train_composition_fixed_final.py \
    --data_dir data/simple_graph/composition_90_mixed_5 \
    --max_iters 30000 \
    --test_interval 1000 \
    --checkpoint_interval 5000

# 5. 创建10%混合数据集
echo -e "\n[Step 5] 创建10%混合数据集..."
python create_mixed_composition_dataset.py \
    --original_dir data/simple_graph/composition_90 \
    --s1_s3_ratio 0.10

# 6. 准备10%混合数据
echo -e "\n[Step 6] 准备10%混合数据..."
cd data/simple_graph
python prepare_composition.py \
    --experiment_name composition_mixed_10 \
    --total_nodes 90 \
    --train_paths_per_pair 10
cd ../..

# 7. 训练10%混合模型
echo -e "\n[Step 7] 开始10%混合训练..."
python train_composition_fixed_final.py \
    --data_dir data/simple_graph/composition_90_mixed_10 \
    --max_iters 30000 \
    --test_interval 1000 \
    --checkpoint_interval 5000

echo -e "\n=========================================="
echo "实验完成！"
echo "=========================================="