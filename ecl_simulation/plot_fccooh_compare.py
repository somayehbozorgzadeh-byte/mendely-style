"""Measured (Fig. S4, read by eye) vs simulated FcCOOH redox cycling on the seven spiral devices (1 mm x 1 mm, gold)."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = {(2, 2): (3.45, -3.30), (2, 5): (1.60, -1.45), (2, 10): (0.67, -0.50), (2, 20): (0.40, -0.25),
       (5, 10): (0.85, -0.65), (10, 10): (0.90, -0.68), (20, 10): (0.75, -0.55)}          # plateau i_GE, i_CE / uA
S = {(r["w"], r["g"]): r for r in json.load(open("fccooh_series.json"))}
devs = [(2, 2), (2, 5), (2, 10), (2, 20), (5, 10), (10, 10), (20, 10)]
lab = [f"W{w}G{g}" for w, g in devs]
mg = np.array([S[d]["i_gen"][-1] * 1e6 for d in devs]); mc = np.array([S[d]["i_col"][-1] * 1e6 for d in devs])
eg = np.array([EXP[d][0] for d in devs]); ec = np.array([EXP[d][1] for d in devs])
scale = np.exp(np.mean(np.log(eg / mg)))          # single overall factor (active area / concentration / D)

fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
x = np.arange(len(devs)); wd = 0.27
ax[0].bar(x - wd, eg, wd, color="0.75", edgecolor="k", label="measured")
ax[0].bar(x, mg, wd, color="tab:red", alpha=.5, label="model (no fitting)")
ax[0].bar(x + wd, mg * scale, wd, color="tab:red", label=f"model × {scale:.2f} (one common factor)")
ax[0].set_yscale("log"); ax[0].set_xticks(x, lab, rotation=30); ax[0].set_ylabel("generator plateau current / µA")
ax[0].set_title("Generator current, all devices", fontsize=10); ax[0].legend(frameon=False, fontsize=8)
ax[1].bar(x - wd / 2, -ec / eg, wd, color="0.75", edgecolor="k", label="measured")
ax[1].bar(x + wd / 2, -mc / mg, wd, color="tab:red", label="model (no fitting)")
ax[1].set_xticks(x, lab, rotation=30); ax[1].set_ylabel("collection efficiency  |i$_{CE}$| / i$_{GE}$"); ax[1].set_ylim(0, 1.05)
ax[1].set_title("Collection efficiency (independent of area, c, D)", fontsize=10); ax[1].legend(frameon=False, fontsize=8)
for (w, g), c in zip(devs, plt.cm.viridis(np.linspace(0, .9, len(devs)))):
    r = S[(w, g)]; E = np.array(r["Eg"])
    ax[2].plot(E - 0.32, np.array(r["i_gen"]) * 1e6 * scale, color=c, label=f"W{w}G{g}")
    ax[2].plot(E - 0.32, np.array(r["i_col"]) * 1e6 * scale, color=c, ls="--")
ax[2].set_xlabel("E$_{gen}$ − E$^{0'}$ / V"); ax[2].set_ylabel("current / µA (model × common factor)")
ax[2].set_title("Simulated CVs (solid: generator, dashed: collector)", fontsize=10); ax[2].legend(frameon=False, fontsize=7, ncol=2)
for a in ax: a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
fig.suptitle("1 mM ferrocenecarboxylic acid, 0.1 V/s, gold spirals in 1 mm × 1 mm: measured (Fig. S4) vs model", fontsize=11)
fig.tight_layout(); fig.savefig("fccooh_model_vs_experiment.png", dpi=140)
print(f"common factor (measured/model) = {scale:.3f}")
print("device  turns  i_GE meas  model  model*f  | CE meas  model")
for d, a, b, c_, e in zip(devs, eg, mg, mc, ec):
    print(f"W{d[0]}G{d[1]:<2}  {S[d]['turns']:4d}  {a:6.2f}  {b:6.2f}  {b*scale:6.2f}   | {-e/a:.2f}   {-c_/b:.2f}")
