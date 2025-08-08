#!/bin/bash
echo '🚀 开始评估h.0 Attention模块移植模型... 🚀'
python evaluate_hybrid_model.py --model_path hybrid_models_h0_attn_transplant_seed42/hybrid_h0_attn_transplant.pt --data_dir data/simple_graph/composition_90
