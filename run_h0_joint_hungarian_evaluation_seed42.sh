#!/bin/bash
echo '🚀 开始批量评估h.0联合匈牙利对齐移植模型... 🚀'
echo '
--- 正在评估: hybrid_h0_joint_hungarian_k0.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models_h0_joint_hungarian_seed42/hybrid_h0_joint_hungarian_k0.pt --data_dir data/simple_graph/composition_90
echo '
--- 正在评估: hybrid_h0_joint_hungarian_k1.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models_h0_joint_hungarian_seed42/hybrid_h0_joint_hungarian_k1.pt --data_dir data/simple_graph/composition_90
echo '
--- 正在评估: hybrid_h0_joint_hungarian_k2.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models_h0_joint_hungarian_seed42/hybrid_h0_joint_hungarian_k2.pt --data_dir data/simple_graph/composition_90
echo '
--- 正在评估: hybrid_h0_joint_hungarian_k3.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models_h0_joint_hungarian_seed42/hybrid_h0_joint_hungarian_k3.pt --data_dir data/simple_graph/composition_90
echo '
--- 正在评估: hybrid_h0_joint_hungarian_k5.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models_h0_joint_hungarian_seed42/hybrid_h0_joint_hungarian_k5.pt --data_dir data/simple_graph/composition_90
echo '
--- 正在评估: hybrid_h0_joint_hungarian_k8.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models_h0_joint_hungarian_seed42/hybrid_h0_joint_hungarian_k8.pt --data_dir data/simple_graph/composition_90
echo '
--- 正在评估: hybrid_h0_joint_hungarian_k10.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models_h0_joint_hungarian_seed42/hybrid_h0_joint_hungarian_k10.pt --data_dir data/simple_graph/composition_90
echo '
--- 正在评估: hybrid_h0_joint_hungarian_k15.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models_h0_joint_hungarian_seed42/hybrid_h0_joint_hungarian_k15.pt --data_dir data/simple_graph/composition_90
echo '
--- 正在评估: hybrid_h0_joint_hungarian_k20.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models_h0_joint_hungarian_seed42/hybrid_h0_joint_hungarian_k20.pt --data_dir data/simple_graph/composition_90
echo '
--- 正在评估: hybrid_h0_joint_hungarian_k30.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models_h0_joint_hungarian_seed42/hybrid_h0_joint_hungarian_k30.pt --data_dir data/simple_graph/composition_90
echo '
--- 正在评估: hybrid_h0_joint_hungarian_k40.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models_h0_joint_hungarian_seed42/hybrid_h0_joint_hungarian_k40.pt --data_dir data/simple_graph/composition_90
