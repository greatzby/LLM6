import matplotlib.pyplot as plt
import numpy as np

# --- 从你的日志中手动提取50k步的最终数据 ---
# (为了方便，我直接帮你从上面的日志里整理好了)
k_values = [10, 15, 20, 30, 45, 60, 65, 70]

# 50k步时V和U的相似度
v_sim_50k = [0.5376, 0.5665, 0.6101, 0.6966, 0.7590, 0.8195, 0.8374, 0.8520]
u_sim_50k = [0.6420, 0.6622, 0.6996, 0.7578, 0.8192, 0.8648, 0.8770, 0.8935]

# 50k步时V和U的覆盖率
v_cov_50k = [0.3572, 0.3987, 0.4466, 0.5533, 0.6466, 0.7320, 0.7587, 0.7801]
u_cov_50k = [0.5205, 0.5321, 0.5721, 0.6479, 0.7363, 0.8029, 0.8222, 0.8448]

# --- 计算差距 ---
similarity_gap = np.array(u_sim_50k) - np.array(v_sim_50k)
coverage_gap = np.array(u_cov_50k) - np.array(v_cov_50k)

# --- 绘图 ---
fig, ax1 = plt.subplots(figsize=(12, 7))

# 设置标题和标签
fig.suptitle('Robustness Check: Divergence Gap vs. Truncation Threshold k', fontsize=16)
ax1.set_xlabel('Truncation Threshold (k)', fontsize=12)
ax1.set_ylabel('Similarity Gap (U_sim - V_sim)', color='tab:blue', fontsize=12)

# 绘制相似度差距
ax1.plot(k_values, similarity_gap, 'o-', color='tab:blue', label='Similarity Gap (U - V)')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.grid(True, linestyle='--', alpha=0.6)

# 创建第二个Y轴共享X轴
ax2 = ax1.twinx()
ax2.set_ylabel('Coverage Gap (U_cov - V_cov)', color='tab:green', fontsize=12)

# 绘制覆盖率差距
ax2.plot(k_values, coverage_gap, 's--', color='tab:green', label='Coverage Gap (U - V)')
ax2.tick_params(axis='y', labelcolor='tab:green')

# 添加一个水平线在y=0，作为参考
ax1.axhline(0, color='red', linestyle=':', linewidth=2, label='No Gap (y=0)')

# 合并图例
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='best')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("divergence_gap_vs_k.png", dpi=300)
plt.show()

print("Meta-analysis plot 'divergence_gap_vs_k.png' saved successfully.")