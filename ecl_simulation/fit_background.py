"""Fit exp = a*model + b (gap-independent background b) on the gap series, validate on the width series.
Model = peak ECL of the 0->0.8 V sweep at 0.1 V/s (exp_sweep.json), everything normalised to W2G10."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = {(r["w"], r["g"]): r for r in json.load(open("exp_sweep.json"))}
ref = R[(2, 10)]["peak"]
mod = lambda d: R[d]["peak"] / ref
gap_dev = [(2, 2), (2, 5), (2, 10), (2, 20)]
w_dev = [(2, 10), (5, 10), (10, 10), (20, 10)]
exp_gap = np.array([22.5, 19.3, 14.4, 10.0]) / 14.4
exp_w = np.array([14.4, 15.8, 16.9, 14.8]) / 14.4
x = np.array([mod(d) for d in gap_dev])
A = np.c_[x, np.ones_like(x)]
# constrain a + b = 1 at the reference (normalisation): exp = a*(m-1) + 1 -> fit a only; b = 1 - a
a = np.sum((x - 1) * (exp_gap - 1)) / np.sum((x - 1) ** 2)
b = 1 - a
pred = lambda d: a * mod(d) + b
print(f"fit on gap series: signal = {a:.2f} * model + {b:.2f}  -> gap-independent part = {100*b:.0f}% of the W2G10 signal")
print("gap series  exp   model  model+bkg")
for d, e in zip(gap_dev, exp_gap): print("  W%dG%-2d   %.2f   %.2f   %.2f" % (*d, e, mod(d), pred(d)))
print("width series (validation, no refit)")
for d, e in zip(w_dev, exp_w): print("  W%dG%-2d   %.2f   %.2f   %.2f" % (*d, e, mod(d), pred(d)))
rms = lambda dev, e: np.sqrt(np.mean([(pred(d) - v) ** 2 for d, v in zip(dev, e)]))
print("rms error: gap %.3f, width %.3f   (model alone: gap %.3f, width %.3f)" % (
    rms(gap_dev, exp_gap), rms(w_dev, exp_w),
    np.sqrt(np.mean([(mod(d) - v) ** 2 for d, v in zip(gap_dev, exp_gap)])), np.sqrt(np.mean([(mod(d) - v) ** 2 for d, v in zip(w_dev, exp_w)]))))

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for k, (dev, e, ttl, lab) in enumerate([(gap_dev, exp_gap, "Gap series (fit)", [f"G{d[1]}" for d in gap_dev]),
                                        (w_dev, exp_w, "Width series (validation)", [f"W{d[0]}" for d in w_dev])]):
    i = np.arange(4); wd = 0.27
    ax[k].bar(i - wd, e, wd, color="0.75", edgecolor="k", label="experiment")
    ax[k].bar(i, [mod(d) for d in dev], wd, color="tab:red", label="model (superoxide ECL only)")
    ax[k].bar(i + wd, [pred(d) for d in dev], wd, color="tab:green", label=f"model + background ({100*b:.0f}% fixed)")
    ax[k].set_xticks(i, lab); ax[k].set_title(ttl); ax[k].set_ylabel("signal / signal(W2G10)")
    ax[k].spines["top"].set_visible(False); ax[k].spines["right"].set_visible(False)
ax[0].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig("ecl_fit_background.png", dpi=150)
