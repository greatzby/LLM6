#!/bin/bash
echo '🚀 开始批量评估 h.0 功能捆绑包移植模型... 🚀'
echo '将使用确定性的贪心解码进行评估，确保结果稳定。'

echo '
--- 正在评估: hybrid_bundle_attn_bundle_transplant_seed42.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models/hybrid_bundle_attn_bundle_transplant_seed42.pt --data_dir data/simple_graph/composition_90 --temperature 0

echo '
--- 正在评估: hybrid_bundle_mlp_bundle_transplant_seed42.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models/hybrid_bundle_mlp_bundle_transplant_seed42.pt --data_dir data/simple_graph/composition_90 --temperature 0
