#!/bin/bash
echo '🚀 开始批量评估 h.0 旋转对齐嫁接模型(Final)的剂量效应... 🚀'
echo '
--- 正在评估: hybrid_h0_rotated_k0.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models_h0_final_seed42/hybrid_h0_rotated_k0.pt --data_dir data/simple_graph/composition_90
echo '
--- 正在评估: hybrid_h0_rotated_k1.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models_h0_final_seed42/hybrid_h0_rotated_k1.pt --data_dir data/simple_graph/composition_90
echo '
--- 正在评估: hybrid_h0_rotated_k2.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models_h0_final_seed42/hybrid_h0_rotated_k2.pt --data_dir data/simple_graph/composition_90
echo '
--- 正在评估: hybrid_h0_rotated_k3.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models_h0_final_seed42/hybrid_h0_rotated_k3.pt --data_dir data/simple_graph/composition_90
echo '
--- 正在评估: hybrid_h0_rotated_k4.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models_h0_final_seed42/hybrid_h0_rotated_k4.pt --data_dir data/simple_graph/composition_90
echo '
--- 正在评估: hybrid_h0_rotated_k5.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models_h0_final_seed42/hybrid_h0_rotated_k5.pt --data_dir data/simple_graph/composition_90
echo '
--- 正在评估: hybrid_h0_rotated_k6.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models_h0_final_seed42/hybrid_h0_rotated_k6.pt --data_dir data/simple_graph/composition_90
echo '
--- 正在评估: hybrid_h0_rotated_k7.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models_h0_final_seed42/hybrid_h0_rotated_k7.pt --data_dir data/simple_graph/composition_90
echo '
--- 正在评估: hybrid_h0_rotated_k8.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models_h0_final_seed42/hybrid_h0_rotated_k8.pt --data_dir data/simple_graph/composition_90
echo '
--- 正在评估: hybrid_h0_rotated_k9.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models_h0_final_seed42/hybrid_h0_rotated_k9.pt --data_dir data/simple_graph/composition_90
echo '
--- 正在评估: hybrid_h0_rotated_k10.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models_h0_final_seed42/hybrid_h0_rotated_k10.pt --data_dir data/simple_graph/composition_90
