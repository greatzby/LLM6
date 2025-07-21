import matplotlib.pyplot as plt
import numpy as np

# ==============================================================================
# 1. 数据准备 (Data Preparation)
#    数据从你提供的日志中手动提取并硬编码于此。
#    结构: data[k_value][metric] = [3k, 10k, 20k, 30k, 40k, 50k]
# ==============================================================================
data = {
    '10': {
        'V-Similarity': [0.8115, 0.6947, 0.6284, 0.5913, 0.5646, 0.5376],
        'U-Similarity': [0.8169, 0.7552, 0.7113, 0.6888, 0.6680, 0.6420],
    },
    '15': {
        'V-Similarity': [0.8503, 0.7385, 0.6737, 0.6237, 0.5911, 0.5665],
        'U-Similarity': [0.8496, 0.7713, 0.7312, 0.6963, 0.6774, 0.6622],
    },
    '20': {
        'V-Similarity': [0.8579, 0.7629, 0.7153, 0.6759, 0.6389, 0.6101],
        'U-Similarity': [0.8530, 0.7889, 0.7599, 0.7333, 0.7114, 0.6996],
    },
    '30': {
        'V-Similarity': [0.8838, 0.8117, 0.7630, 0.7386, 0.7139, 0.6966],
        'U-Similarity': [0.8777, 0.8212, 0.7901, 0.7764, 0.7647, 0.7578],
    },
    '45': {
        'V-Similarity': [0.9124, 0.8696, 0.8290, 0.7999, 0.7776, 0.7590],
        'U-Similarity': [0.9059, 0.8689, 0.8427, 0.8309, 0.8242, 0.8192],
    },
    '60': {
        'V-Similarity': [0.9274, 0.9009, 0.8693, 0.8462, 0.8328, 0.8195],
        'U-Similarity': [0.9244, 0.9025, 0.8812, 0.8723, 0.8698, 0.8648],
    },
    '65': {
        'V-Similarity': [0.9338, 0.9034, 0.8801, 0.8642, 0.8511, 0.8374],
        'U-Similarity': [0.9353, 0.9094, 0.8927, 0.8847, 0.8804, 0.8770],
    },
    '70': {
        'V-Similarity': [0.9366, 0.9091, 0.8885, 0.8771, 0.8653, 0.8520],
        'U-Similarity': [0.9418, 0.9203, 0.9061, 0.9009, 0.8984, 0.8935],
    }
}

iterations = [3, 10, 20, 30, 40, 50]
k_to_plot = ['10', '30', '70'] # 选择代表性的k值：核心、中部、整体

# ==============================================================================
# 2. 绘图 (Plotting)
# ==============================================================================
plt.style.use('seaborn-v0_8-whitegrid') # 使用一个美观的绘图风格
fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=False) # 创建一个包含两个子图的画布

# 定义颜色映射，确保同一k值在两个子图中颜色一致
colors = {'10': 'royalblue', '30': 'darkorange', '70': 'forestgreen'}

# --- 左面板: 相似度演化 (Similarity Evolution) ---
ax = axes[0]
for k in k_to_plot:
    # 绘制V-Similarity (实线)
    ax.plot(iterations, data[k]['V-Similarity'], marker='o', linestyle='-', 
            color=colors[k], label=f'V-Sim (k={k})', linewidth=2.5)
    # 绘制U-Similarity (虚线)
    ax.plot(iterations, data[k]['U-Similarity'], marker='x', linestyle='--', 
            color=colors[k], label=f'U-Sim (k={k})', linewidth=2)

ax.set_title('Similarity Evolution: Core vs. Peripheral Subspaces', fontsize=16, fontweight='bold')
ax.set_xlabel('Training Iteration (in thousands)', fontsize=12)
ax.set_ylabel('Subspace Similarity', fontsize=12)
ax.legend(fontsize=11)
ax.set_ylim(0.5, 1.0) # 设置Y轴范围以便更好地比较

# --- 右面板: 差距演化 (Gap Evolution) ---
ax = axes[1]
for k in k_to_plot:
    v_sim = np.array(data[k]['V-Similarity'])
    u_sim = np.array(data[k]['U-Similarity'])
    gap = u_sim - v_sim
    ax.plot(iterations, gap, marker='s', linestyle='-', 
            color=colors[k], label=f'Gap (k={k})', linewidth=2.5)

ax.axhline(0, color='black', linestyle=':', linewidth=1.5, label='No Difference')
ax.set_title('Evolution of the V-U Similarity Gap', fontsize=16, fontweight='bold')
ax.set_xlabel('Training Iteration (in thousands)', fontsize=12)
ax.set_ylabel('Similarity Gap (U-Sim minus V-Sim)', fontsize=12)
ax.legend(fontsize=11)
ax.set_ylim(bottom=-0.01) # Y轴从-0.01开始，让'No Difference'线更清晰

# --- 全局设置与保存 ---
fig.suptitle('Divergence Dynamics Across Subspace Hierarchies', fontsize=20, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96]) # 调整布局以适应总标题
plt.savefig('divergence_dynamics_k_comparison.png', dpi=300, bbox_inches='tight')
print("绘图完成！图片已保存为 'divergence_dynamics_k_comparison.png'")
plt.show()