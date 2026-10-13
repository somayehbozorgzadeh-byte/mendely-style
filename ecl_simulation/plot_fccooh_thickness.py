"""Measured (Fig. S4) vs model with flat and 100 nm thick electrodes, FcCOOH, all spiral devices."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = {(2, 2): (3.45, -3.30), (2, 5): (1.60, -1.45), (2, 10): (0.67, -0.50), (2, 20): (0.40, -0.25),
       (5, 10): (0.85, -0.65), (10, 10): (0.90, -0.68), (20, 10): (0.75, -0.55)}
devs = [(2, 2), (2, 5), (2, 10), (2, 20), (5, 10), (10, 10), (20, 10)]
lab = [f"W{w}G{g}" for w, g in devs]
eg = np.array([EXP[d][0] for d in devs]); ec = np.array([EXP[d][1] for d in devs])
M = {}
for fn, name in (("fccooh_series.json", "flat"), ("fccooh_series_t100nm.json", "100 nm thick")):
    S = {(r["w"], r["g"]): r for r in json.load(open(fn))}
    mg = np.array([S[d]["i_gen"][-1] * 1e6 for d in devs]); mc = np.array([S[d]["i_col"][-1] * 1e6 for d in devs])
    f = np.exp(np.mean(np.log(eg / mg)))
    M[name] = dict(ig=mg, ic=mc, f=f)

fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
x = np.arange(len(devs)); wd = 0.27
ax[0].bar(x - wd, eg, wd, color="0.75", edgecolor="k", label="measured")
for k, (name, col) in enumerate([("flat", "tab:red"), ("100 nm thick", "tab:orange")]):
    ax[0].bar(x + k * wd, M[name]["ig"] * M[name]["f"], wd, color=col, label=f"model, {name} (× {M[name]['f']:.2f})")
ax[0].set_yscale("log"); ax[0].set_xticks(x, lab, rotation=30); ax[0].set_ylabel("generator plateau current / µA")
ax[0].set_title("Generator current (one common factor per model)", fontsize=10); ax[0].legend(frameon=False, fontsize=8)
ax[1].bar(x - wd, -ec / eg, wd, color="0.75", edgecolor="k", label="measured")
for k, (name, col) in enumerate([("flat", "tab:red"), ("100 nm thick", "tab:orange")]):
    ax[1].bar(x + k * wd, -M[name]["ic"] / M[name]["ig"], wd, color=col, label=f"model, {name}")
ax[1].set_xticks(x, lab, rotation=30); ax[1].set_ylim(0, 1.05); ax[1].set_ylabel("collection efficiency |i$_{CE}$|/i$_{GE}$")
ax[1].set_title("Collection efficiency (no fitting)", fontsize=10); ax[1].legend(frameon=False, fontsize=8, loc="lower left")
ws = [2, 5, 10, 20]; wdev = [(w, 10) for w in ws]; iw = [devs.index(d) for d in wdev]
ax[2].plot(ws, eg[iw] / eg[iw[0]], "ko-", label="measured")
for name, col in (("flat", "tab:red"), ("100 nm thick", "tab:orange")):
    ax[2].plot(ws, M[name]["ig"][iw] / M[name]["ig"][iw[0]], "o-", color=col, label=f"model, {name}")
ax[2].plot(ws, eg[iw] / (eg[iw[0]] * 1.15), "k--", alpha=.5, label="measured, if W2G10 were 15% higher")
ax[2].set_xscale("log", base=2); ax[2].set_xticks(ws, [str(w) for w in ws]); ax[2].set_xlabel("electrode width / µm (gap 10 µm)")
ax[2].set_ylabel("generator current / W2G10"); ax[2].set_title("Width series, relative to W2G10", fontsize=10)
ax[2].legend(frameon=False, fontsize=8)
for a in ax: a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
fig.suptitle("1 mM ferrocenecarboxylic acid, 0.1 V/s, gold two-arm spirals in 1 mm × 1 mm: measured (Fig. S4) vs model", fontsize=11)
fig.tight_layout(); fig.savefig("fccooh_thickness_compare.png", dpi=140)
print("ok")
