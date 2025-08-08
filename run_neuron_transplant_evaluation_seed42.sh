#!/bin/bash
echo '🚀 开始批量评估神经元移植模型 (v2)... 🚀'
echo '
--- 正在评估: hybrid_neuron_k0.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models_neuron_transplant_seed42/hybrid_neuron_k0.pt --data_dir data/simple_graph/composition_90
echo '
--- 正在评估: hybrid_neuron_k1.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models_neuron_transplant_seed42/hybrid_neuron_k1.pt --data_dir data/simple_graph/composition_90
echo '
--- 正在评估: hybrid_neuron_k2.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models_neuron_transplant_seed42/hybrid_neuron_k2.pt --data_dir data/simple_graph/composition_90
echo '
--- 正在评估: hybrid_neuron_k5.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models_neuron_transplant_seed42/hybrid_neuron_k5.pt --data_dir data/simple_graph/composition_90
echo '
--- 正在评估: hybrid_neuron_k10.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models_neuron_transplant_seed42/hybrid_neuron_k10.pt --data_dir data/simple_graph/composition_90
echo '
--- 正在评估: hybrid_neuron_k20.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models_neuron_transplant_seed42/hybrid_neuron_k20.pt --data_dir data/simple_graph/composition_90
echo '
--- 正在评估: hybrid_neuron_k40.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models_neuron_transplant_seed42/hybrid_neuron_k40.pt --data_dir data/simple_graph/composition_90
echo '
--- 正在评估: hybrid_neuron_k80.pt ---'
python evaluate_hybrid_model.py --model_path hybrid_models_neuron_transplant_seed42/hybrid_neuron_k80.pt --data_dir data/simple_graph/composition_90
