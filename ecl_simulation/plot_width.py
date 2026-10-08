import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = json.load(open("width_scan.json"))
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for Ec, mk in [(0.4, "o-"), (0.7, "s--")]:
    r = sorted([x for x in R if x["Ec"] == Ec], key=lambda x: x["w"])
    w = np.array([x["w"] for x in r]) * 1e6
    ax[0].plot(w, [x["ecl_foot"] / 1e12 for x in r], mk, label=f"E$_{{col}}$ = {Ec} V")
    ax[1].plot(w, [x["P_per_collector_area"] / 1e12 for x in r], mk, label=f"{Ec} V")
    ax[2].plot(w, [x["P_unit_cell"] / 1e9 for x in r], mk, label=f"{Ec} V")
for a, t, y in zip(ax, ["Signal for a fixed chip area", "Per unit collector area", "Per strip pair (per unit length)"],
                   ["ECL / 10$^{12}$ photons m$^{-2}$ s$^{-1}$ (footprint)", "ECL / 10$^{12}$ photons m$^{-2}$ s$^{-1}$ (collector area)",
                    "ECL / 10$^{9}$ photons m$^{-1}$ s$^{-1}$"]):
    a.set_xscale("log", base=2); a.set_xticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
    a.set_xlabel("electrode width w (gen = col) / µm, gap = 2 µm"); a.set_ylabel(y, fontsize=8); a.set_title(t, fontsize=10)
    a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
ax[0].legend(frameon=False); ax[0].set_ylim(bottom=0); ax[1].set_yscale("log")
ax[2].set_ylim(bottom=0)
ax[2].annotate("w = 16 µm not fully\nsteady (drift 10$^{-3}$)", xy=(16, R[4]["P_unit_cell"] / 1e9), xytext=(5, 3), fontsize=8, arrowprops=dict(arrowstyle="->"))
fig.suptitle("Effect of electrode width on ECL (2-D model, steady state, E$_{gen}$ = −0.8 V)", y=1.0)
fig.tight_layout(); fig.savefig("ecl_width_dependence.png", dpi=150)
