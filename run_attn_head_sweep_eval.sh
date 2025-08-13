#!/bin/bash
echo '🚀 开始批量评估 attn Bundle 的头部校准模型 (rank 0-10)... 🚀'

echo '
--- 正在评估 Rank=0: hybrid_bundle_attn_head_calibrated_rank0_seed42.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models/hybrid_bundle_attn_head_calibrated_rank0_seed42.pt --data_dir data/simple_graph/composition_90 --temperature 0

echo '
--- 正在评估 Rank=1: hybrid_bundle_attn_head_calibrated_rank1_seed42.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models/hybrid_bundle_attn_head_calibrated_rank1_seed42.pt --data_dir data/simple_graph/composition_90 --temperature 0

echo '
--- 正在评估 Rank=2: hybrid_bundle_attn_head_calibrated_rank2_seed42.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models/hybrid_bundle_attn_head_calibrated_rank2_seed42.pt --data_dir data/simple_graph/composition_90 --temperature 0

echo '
--- 正在评估 Rank=3: hybrid_bundle_attn_head_calibrated_rank3_seed42.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models/hybrid_bundle_attn_head_calibrated_rank3_seed42.pt --data_dir data/simple_graph/composition_90 --temperature 0

echo '
--- 正在评估 Rank=4: hybrid_bundle_attn_head_calibrated_rank4_seed42.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models/hybrid_bundle_attn_head_calibrated_rank4_seed42.pt --data_dir data/simple_graph/composition_90 --temperature 0

echo '
--- 正在评估 Rank=5: hybrid_bundle_attn_head_calibrated_rank5_seed42.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models/hybrid_bundle_attn_head_calibrated_rank5_seed42.pt --data_dir data/simple_graph/composition_90 --temperature 0

echo '
--- 正在评估 Rank=6: hybrid_bundle_attn_head_calibrated_rank6_seed42.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models/hybrid_bundle_attn_head_calibrated_rank6_seed42.pt --data_dir data/simple_graph/composition_90 --temperature 0

echo '
--- 正在评估 Rank=7: hybrid_bundle_attn_head_calibrated_rank7_seed42.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models/hybrid_bundle_attn_head_calibrated_rank7_seed42.pt --data_dir data/simple_graph/composition_90 --temperature 0

echo '
--- 正在评估 Rank=8: hybrid_bundle_attn_head_calibrated_rank8_seed42.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models/hybrid_bundle_attn_head_calibrated_rank8_seed42.pt --data_dir data/simple_graph/composition_90 --temperature 0

echo '
--- 正在评估 Rank=9: hybrid_bundle_attn_head_calibrated_rank9_seed42.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models/hybrid_bundle_attn_head_calibrated_rank9_seed42.pt --data_dir data/simple_graph/composition_90 --temperature 0

echo '
--- 正在评估 Rank=10: hybrid_bundle_attn_head_calibrated_rank10_seed42.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models/hybrid_bundle_attn_head_calibrated_rank10_seed42.pt --data_dir data/simple_graph/composition_90 --temperature 0
