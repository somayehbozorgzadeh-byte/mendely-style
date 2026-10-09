import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

A = json.load(open("verify_analytic.json")); G = json.load(open("verify_grid.json"))
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
c = A["cottrell"]; t = np.array(c["t"])
ax[0].loglog(t * 1e3, np.array(c["i_exact"]) / 10, "k-", label="Cottrell (exact)")
ax[0].loglog(t * 1e3, np.array(c["i_num"]) / 10, "o", color="tab:red", ms=4, label="this code")
ax[0].set_xlabel("time / ms"); ax[0].set_ylabel("current / mA cm$^{-2}$")
ax[0].set_title(f"1) Potential step: max error {100*c['max_rel_err']:.2f}%", fontsize=10); ax[0].legend(frameon=False)
lab = [f"W{r['w']}G{r['g']}" for r in A["aoki"]]; x = np.arange(len(lab))
ax[1].bar(x - 0.2, [r["i_aoki"] * 1e6 for r in A["aoki"]], 0.4, color="0.7", edgecolor="k", label="Aoki 1988 (analytical)")
ax[1].bar(x + 0.2, [r["i_num"] * 1e6 for r in A["aoki"]], 0.4, color="tab:red", label="this code")
for i, r in enumerate(A["aoki"]): ax[1].text(i + 0.2, r["i_num"] * 1e6 * 1.02, f"{100*r['rel_err']:+.1f}%", ha="center", fontsize=8)
ax[1].set_xticks(x, lab); ax[1].set_ylabel("redox-cycling current / µA m$^{-1}$ (per unit cell)")
ax[1].set_title("2) Interdigitated-array redox cycling", fontsize=10); ax[1].legend(frameon=False, fontsize=8)
order = ["coarse", "base (used)", "fine"]
get = lambda gr, g, rt=1e-5: next(r for r in G if r["grid"] == gr and r["g"] == g and r["rtol"] == rt)
ratio = [get(gr, 2)["ecl"] / get(gr, 20)["ecl"] for gr in order]
cells = [get(gr, 2)["n"] for gr in order]
ax[2].plot(cells, ratio, "o-", color="tab:red", label="ECL(W2G2) / ECL(W2G20)")
ax[2].plot(get("base (used)", 2)["n"], get("base (used)", 2, 1e-7)["ecl"] / get("base (used)", 20, 1e-7)["ecl"], "k*", ms=12, label="base grid, 100× tighter tolerance")
ax[2].set_xscale("log"); ax[2].set_xlabel("number of grid cells"); ax[2].set_ylabel("ECL ratio G2/G20")
ax[2].set_ylim(ratio[1] * 0.97, ratio[1] * 1.03)
for n_, r_, gr in zip(cells, ratio, order): ax[2].annotate(gr, (n_, r_), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
ax[2].set_title("3) Mesh & solver convergence (±3% window)", fontsize=10); ax[2].legend(frameon=False, fontsize=8, loc="lower right")
for a in ax: a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
fig.suptitle("Verification: does the code solve its equations correctly?", fontsize=11)
fig.tight_layout(); fig.savefig("ecl_verification.png", dpi=140)
for gr in order:
    print(f"{gr:12s} cells {get(gr,2)['n']:5d}  ECL G2 {get(gr,2)['ecl']:.4e}  G20 {get(gr,20)['ecl']:.4e}  ratio {get(gr,2)['ecl']/get(gr,20)['ecl']:.4f}")
