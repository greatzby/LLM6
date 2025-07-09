#!/usr/bin/env python3
# control_law_en.py
import matplotlib, os, argparse, matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--out", default="report/control_law_en.png")
args = parser.parse_args()

rho_star, d_star = 0.20, 65
fig, ax = plt.subplots(figsize=(5, 4))
ax.axvline(rho_star, ls="--", c="k")
ax.axhline(d_star, ls="--", c="k")

# region labels
ax.text(0.12, 70.5, "safe",  color="green", fontsize=10)
ax.text(0.26, 58.5, "collapse", color="red",   fontsize=10)

# example points
ax.scatter([0.30, 0.25], [63, 70], c=["red", "green"], s=70)
ax.text(0.305, 63, "single-task", fontsize=9)
ax.text(0.255, 70, "mixed-task",  fontsize=9)

# arrows (blue)
ax.arrow(0.30, 63, -0.08, 7, width=0.002, head_width=0.02,
         length_includes_head=True, color="blue")
ax.text(0.18, 71, "↑ dispersion\n(dropout / hetero-grad)", fontsize=8)

ax.arrow(0.30, 63, -0.05, 0, width=0.002, head_width=0.02,
         length_includes_head=True, color="blue")
ax.text(0.24, 61, "↓ alignment\n(orthogonal reg / contrastive)", fontsize=8)

ax.set_xlim(0.10, 0.40)
ax.set_ylim(55, 75)
ax.set_xlabel("rho (WRC★)")
ax.set_ylabel("d (ER)")
ax.set_title("Control law: shift trajectory toward safe zone")
plt.tight_layout()
os.makedirs("report", exist_ok=True)
plt.savefig(args.out, dpi=300)
print("✓ saved:", args.out)