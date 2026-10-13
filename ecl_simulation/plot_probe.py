"""Gap-sensitivity probe: ECL(G2), ECL(G20) and their ratio for superoxide-only parameter variants (steady state, E_col = 0.4 V, W = 2 um).
Values from probe_gap.py output (13 Oct run)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

V = [  # name, ECL(G2), ECL(G20)  [photons m^-2 s^-1]
    ("baseline", 1.954e15, 3.572e14),
    ("k$_{ECL}$ = 10$^8$", 1.342e15, 2.288e14),
    ("k$_{ECL}$ = 10$^7$", 8.503e14, 1.206e14),
    ("over-reduction ↓\n(k$_0$ = 10$^{-8}$)", 2.058e15, 3.663e14),
    ("over-reduction ↑\n(k$_0$ = 10$^{-6}$)", 1.553e15, 3.073e14),
    ("sluggish collector\nsink (k$_0$ = 10$^{-9}$)", 2.012e15, 3.641e14),
    ("dismutation ×30", 1.659e15, 2.134e14),
]
names = [v[0] for v in V]; g2 = np.array([v[1] for v in V]); g20 = np.array([v[2] for v in V])
ratio = g2 / g20
fig, ax = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw=dict(width_ratios=[1.15, 1]))
x = np.arange(len(V)); w = 0.38
ax[0].bar(x - w / 2, g2 / 1e12, w, color="tab:blue", label="W2G2")
ax[0].bar(x + w / 2, g20 / 1e12, w, color="tab:orange", label="W2G20")
ax[0].set_xticks(x, names, rotation=35, ha="right", fontsize=8); ax[0].set_ylabel("ECL / 10$^{12}$ photons m$^{-2}$ s$^{-1}$")
ax[0].legend(frameon=False); ax[0].set_title("Absolute ECL", fontsize=10)
cols = ["0.5"] + ["tab:red"] * (len(V) - 1)
ax[1].bar(x, ratio, color=cols)
ax[1].axhline(22.5 / 10.0, color="k", ls="--"); ax[1].text(len(V) - 0.5, 22.5 / 10 + 0.15, "experiment (2.25)", ha="right", fontsize=9)
for i, r in enumerate(ratio): ax[1].text(i, r + 0.1, f"{r:.1f}", ha="center", fontsize=8)
ax[1].set_xticks(x, names, rotation=35, ha="right", fontsize=8); ax[1].set_ylabel("ECL(W2G2) / ECL(W2G20)")
ax[1].set_title("Gap sensitivity: no single constant reaches the experiment", fontsize=10)
for a in ax: a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
fig.tight_layout(); fig.savefig("ecl_gap_probe.png", dpi=150)
