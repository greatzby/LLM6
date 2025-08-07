#!/bin/bash
echo '🚀 开始批量评估 h.0 权重插值模型... 🚀'
echo '
--- 正在评估: hybrid_interp_alpha0.00_seed42.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models/hybrid_interp_alpha0.00_seed42.pt --data_dir data/simple_graph/composition_90
echo '
--- 正在评估: hybrid_interp_alpha0.20_seed42.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models/hybrid_interp_alpha0.20_seed42.pt --data_dir data/simple_graph/composition_90
echo '
--- 正在评估: hybrid_interp_alpha0.40_seed42.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models/hybrid_interp_alpha0.40_seed42.pt --data_dir data/simple_graph/composition_90
echo '
--- 正在评估: hybrid_interp_alpha0.60_seed42.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models/hybrid_interp_alpha0.60_seed42.pt --data_dir data/simple_graph/composition_90
echo '
--- 正在评估: hybrid_interp_alpha0.80_seed42.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models/hybrid_interp_alpha0.80_seed42.pt --data_dir data/simple_graph/composition_90
echo '
--- 正在评估: hybrid_interp_alpha0.90_seed42.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models/hybrid_interp_alpha0.90_seed42.pt --data_dir data/simple_graph/composition_90
echo '
--- 正在评估: hybrid_interp_alpha0.95_seed42.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models/hybrid_interp_alpha0.95_seed42.pt --data_dir data/simple_graph/composition_90
echo '
--- 正在评估: hybrid_interp_alpha0.98_seed42.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models/hybrid_interp_alpha0.98_seed42.pt --data_dir data/simple_graph/composition_90
echo '
--- 正在评估: hybrid_interp_alpha1.00_seed42.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models/hybrid_interp_alpha1.00_seed42.pt --data_dir data/simple_graph/composition_90
