import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from redox_gc_array import FCMEOH, F

R = {(r["geo"], r["mode"]): r for r in json.load(open("redox_fcmeoh.json"))}
fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
summary = {}
for geo, col, lab in [("ida", "tab:blue", "flat interdigitated"), ("spiral", "tab:red", "entwined spiral (rings)")]:
    gc, sg = R[(geo, "GC")], R[(geo, "single")]
    A = gc["gen_area"]; Eg = np.array(gc["Eg"]); ig = np.array(gc["i_gen"]); ic = np.array(gc["i_col"]); i1 = np.array(sg["i_gen"])
    ax[0].plot(Eg, ig * 1e9, color=col, label=f"{lab}: generator")
    ax[0].plot(Eg, ic * 1e9, color=col, ls="--", label=f"{lab}: collector")
    ax[1].plot(Eg, ig / A / 10, color=col, label=f"{lab}, GC mode")
    ax[1].plot(Eg, i1 / A / 10, color=col, ls=":", label=f"{lab}, collector off")
    k = int(np.argmin(np.abs(Eg - 0.5)))
    summary[geo] = dict(i_gen=ig[k], i_col=ic[k], i_single=i1[k], CE=-ic[k] / ig[k], AF=ig[k] / i1[k], j=ig[k] / A, A=A)
ax[0].set_xlabel("E$_{generator}$ / V vs Ag/AgCl"); ax[0].set_ylabel("current / nA"); ax[0].legend(frameon=False, fontsize=8)
ax[0].set_title("Generator-collector CV (1 mM FcMeOH, 0.1 V/s, collector 0 V)", fontsize=10)
ax[1].set_xlabel("E$_{generator}$ / V vs Ag/AgCl"); ax[1].set_ylabel("generator current density / mA cm$^{-2}$")
ax[1].set_title("Per generator area: redox cycling vs. single electrode", fontsize=10); ax[1].legend(frameon=False, fontsize=8)
names = ["collection\nefficiency", "amplification\nfactor", "j$_{gen}$ at 0.5 V\n/ mA cm$^{-2}$"]
x = np.arange(3)
for i, (geo, col) in enumerate([("ida", "tab:blue"), ("spiral", "tab:red")]):
    s = summary[geo]
    v = [s["CE"], s["AF"], s["j"] / 10]
    b = ax[2].bar(x + (i - 0.5) * 0.38, v, 0.38, color=col, label="flat interdigitated" if geo == "ida" else "entwined spiral")
    for xx, vv in zip(x + (i - 0.5) * 0.38, v): ax[2].text(xx, vv * 1.02, f"{vv:.2f}", ha="center", fontsize=8)
ax[2].set_xticks(x, names); ax[2].legend(frameon=False, fontsize=8); ax[2].set_title("Figures of merit at E$_{gen}$ = 0.5 V (forward scan)", fontsize=10)
for a in ax: a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
fig.suptitle("W2G2, 25 generator/collector pairs: flat interdigitated array (200 µm fingers) vs entwined spiral", fontsize=11)
fig.tight_layout(); fig.savefig("redox_fcmeoh_ida_vs_spiral.png", dpi=140)

p = FCMEOH; x_ = 1.0
aoki_per_finger = F * p["c_bulk"] * p["D_R"] * (0.637 * np.log(2.55 * (1 + x_)) - 0.19 / (1 + x_) ** 2)
aoki_ida = 25 * aoki_per_finger * 200e-6
for geo in ("ida", "spiral"):
    s = summary[geo]
    print(f"{geo:7s} gen area {s['A']*1e12:.0f} um2 | i_gen {s['i_gen']*1e9:7.2f} nA  i_col {s['i_col']*1e9:7.2f} nA  "
          f"single {s['i_single']*1e9:6.2f} nA | CE {s['CE']:.3f}  AF {s['AF']:.2f}  j_gen {s['j']/10:.3f} mA/cm2")
print(f"Aoki (infinite IDA, 25 fingers x 200 um, D_R): {aoki_ida*1e9:.2f} nA  vs finite-array model {summary['ida']['i_gen']*1e9:.2f} nA")
