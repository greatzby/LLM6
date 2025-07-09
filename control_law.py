#!/usr/bin/env python3
# control_law.py
"""
画一个二维箭头示意图：
  坐标同相图 (ρ, d)
  • 起点给两条典型轨迹 (单 task / 混 task)
  • 用箭头表示“增加分散力”或“减少对齐力”的调控方向
输出: report/control_law.png
"""

import argparse, os, matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("-o","--out",default="report/control_law.png")
args = parser.parse_args()

rho_star, d_star = 0.20, 65
fig, ax = plt.subplots(figsize=(5,4))
ax.axvline(rho_star, ls="--", c="k"); ax.axhline(d_star, ls="--", c="k")

# 标注区域
ax.text(0.05, 70, "safe", color="g", fontsize=10)
ax.text(0.28, 50, "collapse", color="r", fontsize=10)

# 两个示例点
ax.scatter([0.30, 0.28], [63, 70], c=["r","g"], s=60)
ax.text(0.31, 63, "单任务\n(ρ↑ d↓)", fontsize=8)
ax.text(0.29, 70, "混任务\n(d↑)", fontsize=8)

# 调控箭头
ax.arrow(0.30, 63, -0.12, +8, width=0.002, head_width=0.02,
         length_includes_head=True, color="b")
ax.text(0.18, 71, "↑ 分散力\n(跨阶段梯度 / dropout)", fontsize=7)

ax.arrow(0.30, 63, -0.05, 0, width=0.002, head_width=0.02,
         length_includes_head=True, color="b")
ax.text(0.24, 61, "↓ 对齐力\n(对比损 / 正交正则)", fontsize=7)

ax.set_xlim(0.10,0.40); ax.set_ylim(55,75)
ax.set_xlabel("ρ  (WRC★)"); ax.set_ylabel("d  (ER)")
ax.set_title("Control law: shift toward safe zone")
plt.tight_layout(); os.makedirs("report",exist_ok=True)
plt.savefig(args.out, dpi=300)
print("✓ control law figure saved to", args.out)