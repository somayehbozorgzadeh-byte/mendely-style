"""Experiment vs model for both protocols: (a) gen -0.8 V, collector at 0 V during pre-hold;
(b) gen -0.7 V for 5 s with collector disconnected, then collector sweep 0->0.8 V at 0.1 V/s. Normalised to W2G10."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

A = {(r["w"], r["g"]): r for r in json.load(open("exp_sweep.json"))}
B = {(r["w"], r["g"]): r for r in json.load(open("exp_sweep_eg07.json"))}
gap_dev = [(2, 2), (2, 5), (2, 10), (2, 20)]; w_dev = [(2, 10), (5, 10), (10, 10), (20, 10)]
exp = {(2, 2): 22.5, (2, 5): 19.3, (2, 10): 14.4, (2, 20): 10.0, (5, 10): 15.8, (10, 10): 16.9, (20, 10): 14.8}
err = {(2, 2): 1.4, (2, 5): 1.4, (2, 10): 1.5, (2, 20): 0.6, (5, 10): 1.8, (10, 10): 1.9, (20, 10): 1.2}
n = lambda R, d, k="peak": R[d][k] / R[(2, 10)][k]
# background fit (signal = a*model + 1 - a) on gap series, protocol B
x = np.array([n(B, d) for d in gap_dev]); y = np.array([exp[d] / 14.4 for d in gap_dev])
a = np.sum((x - 1) * (y - 1)) / np.sum((x - 1) ** 2)
fit = lambda d: a * n(B, d) + 1 - a

fig, ax = plt.subplots(1, 3, figsize=(16, 4.4), gridspec_kw=dict(width_ratios=[1, 1, 1.1]))
for k, (devs, ttl, lab) in enumerate([(gap_dev, "Gap series (W = 2 µm)", [f"G{d[1]}" for d in gap_dev]),
                                      (w_dev, "Width series (G = 10 µm)", [f"W{d[0]}" for d in w_dev])]):
    i = np.arange(4); wd = 0.2
    ax[k].bar(i - 1.5 * wd, [exp[d] / 14.4 for d in devs], wd, yerr=[err[d] / 14.4 for d in devs], capsize=3, color="0.75", edgecolor="k", label="experiment")
    ax[k].bar(i - 0.5 * wd, [n(A, d) for d in devs], wd, color="tab:red", alpha=.55, label="model: gen −0.8 V, collector at 0 V in pre-hold")
    ax[k].bar(i + 0.5 * wd, [n(B, d) for d in devs], wd, color="tab:red", label="model: gen −0.7 V, 5 s, collector off (your protocol)")
    ax[k].bar(i + 1.5 * wd, [fit(d) for d in devs], wd, color="tab:green", label=f"your protocol + {100*(1-a):.0f}% gap-independent background")
    ax[k].set_xticks(i, lab); ax[k].set_title(ttl); ax[k].set_ylabel("ECL signal / signal(W2G10)")
    ax[k].spines["top"].set_visible(False); ax[k].spines["right"].set_visible(False)
ax[0].legend(frameon=False, fontsize=7.5)
cm = plt.cm.viridis(np.linspace(0, 0.9, 7))
for (d, r), c in zip(sorted(B.items()), cm):
    ax[2].plot(r["E"], np.array(r["ecl"]) / 1e12, color=c, label=f"W{d[0]}G{d[1]}")
ax[2].set_xlabel("E$_{collector}$ / V vs Ag/AgCl"); ax[2].set_ylabel("ECL / 10$^{12}$ photons m$^{-2}$ s$^{-1}$")
ax[2].set_title("Simulated sweeps, your protocol"); ax[2].legend(frameon=False, fontsize=8, ncol=2)
ax[2].spines["top"].set_visible(False); ax[2].spines["right"].set_visible(False)
fig.tight_layout(); fig.savefig("ecl_protocol_compare.png", dpi=150)
print("device  exp   model(-0.8)  model(-0.7, OC 5s)  +bkg   Epk(-0.7)")
for d in gap_dev + w_dev[1:]:
    print("W%dG%-2d  %.2f   %.2f        %.2f              %.2f   %.2f" % (*d, exp[d] / 14.4, n(A, d), n(B, d), fit(d), B[d]["Epk"]))
print("background fraction %.0f%%; rms gap %.3f width %.3f" % (100 * (1 - a),
      np.sqrt(np.mean([(fit(d) - exp[d] / 14.4) ** 2 for d in gap_dev])), np.sqrt(np.mean([(fit(d) - exp[d] / 14.4) ** 2 for d in w_dev]))))
