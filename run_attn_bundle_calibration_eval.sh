#!/bin/bash
echo '🚀 开始批量评估 attn Bundle 的校准模型 (路线图 A+ v3)... 🚀'

echo '
--- 正在评估: hybrid_bundle_attn_calibrated_g0.2_seed42.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models/hybrid_bundle_attn_calibrated_g0.2_seed42.pt --data_dir data/simple_graph/composition_90 --temperature 0

echo '
--- 正在评估: hybrid_bundle_attn_calibrated_g0.3_seed42.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models/hybrid_bundle_attn_calibrated_g0.3_seed42.pt --data_dir data/simple_graph/composition_90 --temperature 0

echo '
--- 正在评估: hybrid_bundle_attn_calibrated_g0.5_seed42.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models/hybrid_bundle_attn_calibrated_g0.5_seed42.pt --data_dir data/simple_graph/composition_90 --temperature 0

echo '
--- 正在评估: hybrid_bundle_attn_calibrated_g1.0_seed42.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models/hybrid_bundle_attn_calibrated_g1.0_seed42.pt --data_dir data/simple_graph/composition_90 --temperature 0
