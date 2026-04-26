"""
Generate publication figures for NCAA volleyball ablation study (UPDATED).

Outputs:
    SEM8/Grad Project/figures/fig_ablation.pdf
    SEM8/Grad Project/figures/table_results.tex
"""

import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})

# ✅ UPDATED PATH
OUT = "SEM8/Grad Project/figures"
os.makedirs(OUT, exist_ok=True)


# ════════════════════════════════════════════════════════════
# FIGURE — Ablation (Logistic Regression, YOUR RESULTS)
# ════════════════════════════════════════════════════════════

groups = [
    ("Box score\nonly",      6,  0.6001, 0.6471),
    ("Elo only",             3,  0.7712, 0.8567),
    ("Box + Elo",            9,  0.7717, 0.8563),
    ("+ Context",           12,  0.7712, 0.8564),
    ("Full model",          12,  0.7717, 0.8563),
]

labels = [g[0] for g in groups]
ns     = [g[1] for g in groups]
accs   = [g[2] for g in groups]
aucs   = [g[3] for g in groups]

x = np.arange(len(labels))
w = 0.32

fig, ax = plt.subplots(figsize=(7, 3.8))
b1 = ax.bar(x - w/2, accs, w, label="Accuracy", zorder=3)
b2 = ax.bar(x + w/2, aucs, w, label="ROC AUC",  zorder=3)

ax.set_ylim(0.55, 0.90)
ax.set_ylabel("Score")
ax.set_xticks(x)
ax.set_xticklabels([f"{l}\n({n})" for l, n in zip(labels, ns)], fontsize=8)
ax.legend(loc="upper left", framealpha=0.9)

ax.set_title("Feature ablation study (Logistic Regression, walk-forward)")

for bars in [b1, b2]:
    for bar in bars:
        val = round(bar.get_height(), 3)
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.003,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=7.5
        )

fig.tight_layout()
fig.savefig(f"{OUT}/fig_ablation.pdf")
fig.savefig(f"{OUT}/fig_ablation.png")
plt.close(fig)

print("Saved updated ablation figure.")


# ════════════════════════════════════════════════════════════
# TABLE — Model comparison (YOUR RESULTS)
# ════════════════════════════════════════════════════════════

rows = [
    # label,            #feat,  LR_acc,  LR_auc,  XGB_acc, XGB_auc
    ("Box score only",     6,   0.6001,  0.6471,  0.5794,  0.6097),
    ("Elo only",           3,   0.7712,  0.8567,  0.7708,  0.8528),
    ("Box + Elo",          9,   0.7717,  0.8563,  0.7591,  0.8409),
    ("+ Context",         12,   0.7712,  0.8564,  0.7573,  0.8436),
    ("Full model",        12,   0.7717,  0.8563,  0.7591,  0.8409),
]

tex = r"""\begin{table}[htbp]
\centering
\caption{Walk-forward model comparison across feature configurations.
}
\label{tab:results}
\begin{tabular}{lc rr rr}
\toprule
 & & \multicolumn{2}{c}{Logistic Regression} & \multicolumn{2}{c}{XGBoost} \\
\cmidrule(lr){3-4} \cmidrule(lr){5-6}
Feature set & \#\,Feat. & Acc. & AUC & Acc. & AUC \\
\midrule
"""

# Find best values
lr_accs = [r[2] for r in rows]
lr_aucs = [r[3] for r in rows]
xg_accs = [r[4] for r in rows]
xg_aucs = [r[5] for r in rows]

best_lr_acc = max(lr_accs)
best_lr_auc = max(lr_aucs)
best_xg_acc = max(xg_accs)
best_xg_auc = max(xg_aucs)

for label, n, lr_a, lr_u, xg_a, xg_u in rows:

    lr_a_s = f"{lr_a:.4f}"
    lr_u_s = f"{lr_u:.4f}"
    if lr_a == best_lr_acc:
        lr_a_s = r"\textbf{" + lr_a_s + "}"
    if lr_u == best_lr_auc:
        lr_u_s = r"\textbf{" + lr_u_s + "}"

    xg_a_s = f"{xg_a:.4f}"
    xg_u_s = f"{xg_u:.4f}"
    if xg_a == best_xg_acc:
        xg_a_s = r"\textbf{" + xg_a_s + "}"
    if xg_u == best_xg_auc:
        xg_u_s = r"\textbf{" + xg_u_s + "}"

    tex += f"{label} & {n} & {lr_a_s} & {lr_u_s} & {xg_a_s} & {xg_u_s} \\\\\n"

tex += r"""\bottomrule
\end{tabular}
\end{table}
"""

with open(f"{OUT}/table_results.tex", "w") as f:
    f.write(tex)

print("Saved updated LaTeX table.")