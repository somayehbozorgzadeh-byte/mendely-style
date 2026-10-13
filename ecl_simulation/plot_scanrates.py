import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from run_scanrates import RATES

cm = plt.cm.viridis(np.linspace(0.05, 0.9, len(RATES)))
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
rows = []
for V, c in zip(RATES, cm):
    d = np.load(f"scan_{V}.npz"); E, ecl, ic = d["E"], d["ecl"] / 1e12, d["ic"] / 10
    f = d["t"] <= d["tf"]
    ax[0].plot(E[f], ecl[f], color=c, label=f"{V:g} V/s")
    ax[0].plot(E[~f], ecl[~f], "--", color=c, alpha=.6)
    ax[1].plot(E[f], ecl[f] / ecl[f].max(), color=c, label=f"{V:g} V/s")
    ax[2].plot(E[f], ic[f], color=c, label=f"{V:g} V/s")
    ax[2].plot(E[~f], ic[~f], "--", color=c, alpha=.6)
    rows.append((V, E[f][ecl[f].argmax()], ecl[f].max(), ecl[~f].max()))
ax[0].set(xlabel="E$_{collector}$ / V vs Ag/AgCl", ylabel="ECL / 10$^{12}$ photons m$^{-2}$ s$^{-1}$ (footprint)",
          title="ECL–potential (solid: forward, dashed: reverse)")
ax[1].set(xlabel="E$_{collector}$ / V vs Ag/AgCl", ylabel="ECL / ECL$_{max}$ (forward)", title="Normalised, forward sweep")
ax[2].set(xlabel="E$_{collector}$ / V vs Ag/AgCl", ylabel="collector current / mA cm$^{-2}$", title="Collector current")
for a in ax:
    a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
ax[0].legend(frameon=False, title="scan rate")
fig.suptitle("w = 2 µm, gap = 2 µm interdigitated device, E$_{gen}$ = −0.8 V", y=1.0)
fig.tight_layout(); fig.savefig("ecl_scanrates_2d.png", dpi=150)
print("V/s, Epeak(fwd), ECLpeak fwd, ECLpeak rev  [photons m-2 s-1]")
for r in rows: print("%5g  %.2f V  %.3e  %.3e  (rev/fwd %.2f)" % (*r, r[3] / r[2]))
