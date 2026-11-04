"""Hypothesis sketch: predicted probe accuracy by layer. Qualitative shapes only, no data."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

out = sys.argv[1]
plt.rcParams.update({"font.family": "Inter", "font.size": 12, "axes.edgecolor": "#5b6370",
                     "axes.labelcolor": "#1b2127", "xtick.color": "#5b6670", "ytick.color": "#5b6670"})
teal, red, grey, amber = "#0f6b6e", "#c2432d", "#8a949d", "#a8641a"
x = np.linspace(0, 1, 200)
sig = lambda x, c, k: 1 / (1 + np.exp(-(x - c) * k))

easy = 0.55 + 0.42 * sig(x, 0.15, 30)                         # train structures
hard = 0.38 + 0.52 * sig(x, 0.45, 14) - 0.07 * sig(x, 0.88, 30)  # distractor structures
ctrl = 0.58 + 0.06 * sig(x, 0.3, 10) - 0.04 * sig(x, 0.8, 12)   # control task

fig, ax = plt.subplots(figsize=(10.4, 5.0), dpi=200)
ax.axhline(0.5, color=grey, lw=1.2, ls=(0, (4, 4)))
ax.plot(x, easy, color=grey, lw=2.6)
ax.plot(x, hard, color=teal, lw=3.2)
ax.plot(x, ctrl, color=amber, lw=2.2, ls=(0, (6, 3)))
ax.axvspan(0.35, 0.75, color=teal, alpha=0.06, lw=0)

ax.text(0.98, 0.985, "Train structures (simple sentences)", color="#5b6670", ha="right", va="bottom", fontsize=12)
ax.text(0.585, 0.885, "Held-out distractor structures", color=teal, fontsize=12.5, fontweight="semibold", ha="right")
ax.text(0.99, 0.575, "Control task (random labels)", color=amber, fontsize=12, ha="right", va="top")
ax.text(0.99, 0.485, "Keyword rule on minimal pairs", color="#5b6670", fontsize=11.5, ha="right", va="top")
ax.text(0.02, 0.245, "H2: early layers follow the family word,\nso distractor accuracy falls below chance", color=teal, fontsize=11, va="bottom")
ax.text(0.58, 0.41, "H1, H3: attribution becomes\ndecodable mid-network", color=teal, fontsize=11, ha="center")
xs = 0.66; yh = 0.38 + 0.52 / (1 + np.exp(-(xs - 0.45) * 14)) - 0.07 / (1 + np.exp(-(xs - 0.88) * 30)); yc = 0.58 + 0.06 / (1 + np.exp(-(xs - 0.3) * 10)) - 0.04 / (1 + np.exp(-(xs - 0.8) * 12))
ax.annotate("", xy=(xs, yh - 0.01), xytext=(xs, yc + 0.01),
            arrowprops=dict(arrowstyle="<->", color=teal, lw=1.4))
ax.text(xs + 0.012, (yh + yc) / 2, "selectivity", color=teal, fontsize=10.5, va="center")

ax.set_xlim(0, 1); ax.set_ylim(0.22, 1.03)
ax.set_xticks([0, 0.25, 0.55, 1.0], ["embeddings", "early layers", "middle layers", "final layer"])
ax.set_yticks([0.5, 1.0], ["chance", "ceiling"])
ax.set_ylabel("Probe accuracy")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.suptitle("Predicted layer profile (a hypothesis sketch, not data)", x=0.065, ha="left", fontsize=16,
             fontweight="semibold", color="#1b2127", y=0.975)
fig.text(0.065, 0.895, "Shapes are illustrative. Figure 1 of the final report replaces this with measured curves.",
         fontsize=12, color="#5b6670")
fig.subplots_adjust(left=0.1, right=0.94, bottom=0.1, top=0.84)
fig.savefig(out, facecolor="white")
