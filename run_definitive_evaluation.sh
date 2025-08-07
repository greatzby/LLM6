#!/bin/bash
echo '=================================================='
echo '🚀 开始批量评估最终的、决定性的移植模型... 🚀'
echo '=================================================='

echo '
--- 正在评估: hybrid_definitive_step1_io_only_seed42.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models/hybrid_definitive_step1_io_only_seed42.pt --data_dir data/simple_graph/composition_90
echo '
--- 正在评估: hybrid_definitive_step2_plus_pos_emb_seed42.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models/hybrid_definitive_step2_plus_pos_emb_seed42.pt --data_dir data/simple_graph/composition_90
echo '
--- 正在评估: hybrid_definitive_step3_plus_final_ln_seed42.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models/hybrid_definitive_step3_plus_final_ln_seed42.pt --data_dir data/simple_graph/composition_90
echo '
--- 正在评估: hybrid_definitive_step4_full_transplant_seed42.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models/hybrid_definitive_step4_full_transplant_seed42.pt --data_dir data/simple_graph/composition_90
