#!/bin/bash
echo '🚀 开始批量评估 lm_head 旋转对齐嫁接模型(V2)的剂量效应... 🚀'
echo '
--- 正在评估: hybrid_lm_head_rotated_v2_k0_seed42.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models_lm_head_v2/hybrid_lm_head_rotated_v2_k0_seed42.pt --data_dir data/simple_graph/composition_90
echo '
--- 正在评估: hybrid_lm_head_rotated_v2_k1_seed42.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models_lm_head_v2/hybrid_lm_head_rotated_v2_k1_seed42.pt --data_dir data/simple_graph/composition_90
echo '
--- 正在评估: hybrid_lm_head_rotated_v2_k2_seed42.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models_lm_head_v2/hybrid_lm_head_rotated_v2_k2_seed42.pt --data_dir data/simple_graph/composition_90
echo '
--- 正在评估: hybrid_lm_head_rotated_v2_k3_seed42.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models_lm_head_v2/hybrid_lm_head_rotated_v2_k3_seed42.pt --data_dir data/simple_graph/composition_90
echo '
--- 正在评估: hybrid_lm_head_rotated_v2_k4_seed42.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models_lm_head_v2/hybrid_lm_head_rotated_v2_k4_seed42.pt --data_dir data/simple_graph/composition_90
echo '
--- 正在评估: hybrid_lm_head_rotated_v2_k5_seed42.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models_lm_head_v2/hybrid_lm_head_rotated_v2_k5_seed42.pt --data_dir data/simple_graph/composition_90
echo '
--- 正在评估: hybrid_lm_head_rotated_v2_k6_seed42.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models_lm_head_v2/hybrid_lm_head_rotated_v2_k6_seed42.pt --data_dir data/simple_graph/composition_90
echo '
--- 正在评估: hybrid_lm_head_rotated_v2_k7_seed42.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models_lm_head_v2/hybrid_lm_head_rotated_v2_k7_seed42.pt --data_dir data/simple_graph/composition_90
echo '
--- 正在评估: hybrid_lm_head_rotated_v2_k8_seed42.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models_lm_head_v2/hybrid_lm_head_rotated_v2_k8_seed42.pt --data_dir data/simple_graph/composition_90
echo '
--- 正在评估: hybrid_lm_head_rotated_v2_k9_seed42.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models_lm_head_v2/hybrid_lm_head_rotated_v2_k9_seed42.pt --data_dir data/simple_graph/composition_90
echo '
--- 正在评估: hybrid_lm_head_rotated_v2_k10_seed42.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models_lm_head_v2/hybrid_lm_head_rotated_v2_k10_seed42.pt --data_dir data/simple_graph/composition_90
