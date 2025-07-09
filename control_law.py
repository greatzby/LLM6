#!/usr/bin/env python3
# control_law_ascii.py
import matplotlib, os, argparse, matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'SimHei'           # 没中文字体可注释掉
matplotlib.rcParams['axes.unicode_minus'] = False

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--out", default="report/control_law.png")
args = parser.parse_args()

rho_star, d_star = 0.20, 65
fig, ax = plt.subplots(figsize=(5, 4))
ax.axvline(rho_star, ls="--", c="k")
ax.axhline(d_star,  ls="--", c="k")

ax.text(0.12, 72, "安全区", color="g")
ax.text(0.28, 58, "坍缩区", color="r")

# 示例点
ax.scatter([0.30, 0.25], [63, 70], c=["r", "g"], s=70)
ax.text(0.31, 63, "单任务", fontsize=9)
ax.text(0.26, 70, "混任务", fontsize=9)

# 箭头
ax.arrow(0.30, 63, -0.08, 7, width=0.002, head_width=0.02,
         length_includes_head=True, color="b")
ax.text(0.18, 71, "↑ 分散力\n(跨阶段梯度 / dropout)", fontsize=8)

ax.arrow(0.30, 63, -0.05, 0, width=0.002, head_width=0.02,
         length_includes_head=True, color="b")
ax.text(0.24, 61, "↓ 对齐力\n(正交正则 / 对比损)", fontsize=8)

ax.set_xlim(0.10, 0.40)
ax.set_ylim(55, 75)
ax.set_xlabel("rho (WRC★)")
ax.set_ylabel("d (ER)")
ax.set_title("控制律：把轨迹推回安全区")
plt.tight_layout()
os.makedirs("report", exist_ok=True)
plt.savefig(args.out, dpi=300)
print("✓ control law saved to", args.out)