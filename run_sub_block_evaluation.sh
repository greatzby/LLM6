#!/bin/bash
echo '🚀 开始批量评估 h.0 内部组件移植模型... 🚀'
echo '
--- 正在评估: hybrid_sub_block_base_transplant_only_seed42.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models/hybrid_sub_block_base_transplant_only_seed42.pt --data_dir data/simple_graph/composition_90
echo '
--- 正在评估: hybrid_sub_block_plus_only_attention_seed42.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models/hybrid_sub_block_plus_only_attention_seed42.pt --data_dir data/simple_graph/composition_90
echo '
--- 正在评估: hybrid_sub_block_plus_only_ffn_seed42.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models/hybrid_sub_block_plus_only_ffn_seed42.pt --data_dir data/simple_graph/composition_90
echo '
--- 正在评估: hybrid_sub_block_plus_full_h0_seed42.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models/hybrid_sub_block_plus_full_h0_seed42.pt --data_dir data/simple_graph/composition_90
