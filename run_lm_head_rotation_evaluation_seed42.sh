#!/bin/bash
echo '🚀 开始批量评估 lm_head 旋转对齐嫁接模型的剂量效应... 🚀'
echo '
--- 正在评估: hybrid_lm_head_rotated_k0_seed42.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models_lm_head/hybrid_lm_head_rotated_k0_seed42.pt --data_dir data/simple_graph/composition_90
echo '
--- 正在评估: hybrid_lm_head_rotated_k1_seed42.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models_lm_head/hybrid_lm_head_rotated_k1_seed42.pt --data_dir data/simple_graph/composition_90
echo '
--- 正在评估: hybrid_lm_head_rotated_k2_seed42.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models_lm_head/hybrid_lm_head_rotated_k2_seed42.pt --data_dir data/simple_graph/composition_90
echo '
--- 正在评估: hybrid_lm_head_rotated_k3_seed42.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models_lm_head/hybrid_lm_head_rotated_k3_seed42.pt --data_dir data/simple_graph/composition_90
echo '
--- 正在评估: hybrid_lm_head_rotated_k4_seed42.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models_lm_head/hybrid_lm_head_rotated_k4_seed42.pt --data_dir data/simple_graph/composition_90
echo '
--- 正在评估: hybrid_lm_head_rotated_k5_seed42.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models_lm_head/hybrid_lm_head_rotated_k5_seed42.pt --data_dir data/simple_graph/composition_90
echo '
--- 正在评估: hybrid_lm_head_rotated_k6_seed42.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models_lm_head/hybrid_lm_head_rotated_k6_seed42.pt --data_dir data/simple_graph/composition_90
echo '
--- 正在评估: hybrid_lm_head_rotated_k7_seed42.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models_lm_head/hybrid_lm_head_rotated_k7_seed42.pt --data_dir data/simple_graph/composition_90
echo '
--- 正在评估: hybrid_lm_head_rotated_k8_seed42.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models_lm_head/hybrid_lm_head_rotated_k8_seed42.pt --data_dir data/simple_graph/composition_90
echo '
--- 正在评估: hybrid_lm_head_rotated_k9_seed42.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models_lm_head/hybrid_lm_head_rotated_k9_seed42.pt --data_dir data/simple_graph/composition_90
echo '
--- 正在评估: hybrid_lm_head_rotated_k10_seed42.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models_lm_head/hybrid_lm_head_rotated_k10_seed42.pt --data_dir data/simple_graph/composition_90
