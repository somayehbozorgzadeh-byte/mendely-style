import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = {(r["w"], r["g"]): r for r in json.load(open("exp_sweep.json"))}
exp_gap = {2: 22.5, 5: 19.3, 10: 14.4, 20: 10.0}          # W2 G*  (read from Fig. C, x10^6 a.u.)
exp_w = {2: 14.4, 5: 15.8, 10: 16.9, 20: 14.8}            # W* G10 (read from Fig. D)
err_g = {2: 1.4, 5: 1.4, 10: 1.5, 20: 0.6}
err_w = {2: 1.3, 5: 1.8, 10: 1.9, 20: 1.2}

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw=dict(width_ratios=[1, 1, 1.15]))
for a, series, key, lab, exp, err in [(ax[0], [2, 5, 10, 20], "g", "gap G / µm (W = 2 µm)", exp_gap, err_g),
                                      (ax[1], [2, 5, 10, 20], "w", "width W / µm (G = 10 µm)", exp_w, err_w)]:
    ref_exp = exp[10 if key == "g" else 2]
    dev = lambda s: (2, s) if key == "g" else (s, 10)
    ref_model = R[(2, 10)]["peak"]
    x = np.arange(4); wd = 0.27
    a.bar(x - wd, [exp[s] / ref_exp for s in series], wd, yerr=[err[s] / ref_exp for s in series], color="0.75", edgecolor="k", label="experiment", capsize=3)
    a.bar(x, [R[dev(s)]["peak"] / ref_model for s in series], wd, color="tab:red", label="model: peak ECL")
    ref_i = R[(2, 10)]["integral"]
    a.bar(x + wd, [R[dev(s)]["integral"] / ref_i for s in series], wd, color="tab:blue", label="model: ECL integrated over sweep")
    a.set_xticks(x, [str(s) for s in series]); a.set_xlabel(lab); a.set_ylabel("signal / signal(W2 G10)")
    a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
ax[0].set_title("Gap series"); ax[1].set_title("Width series"); ax[0].legend(frameon=False, fontsize=8)
cm = plt.cm.viridis(np.linspace(0, 0.9, 7))
for (k, r), c in zip(sorted(R.items()), cm):
    ax[2].plot(r["E"], np.array(r["ecl"]) / 1e12, color=c, label=f"W{k[0]}G{k[1]}")
ax[2].set_xlabel("E$_{collector}$ / V vs Ag/AgCl"); ax[2].set_ylabel("ECL / 10$^{12}$ photons m$^{-2}$ s$^{-1}$ (footprint)")
ax[2].legend(frameon=False, fontsize=8, ncol=2); ax[2].set_title("Simulated sweeps (0.1 V/s)")
ax[2].spines["top"].set_visible(False); ax[2].spines["right"].set_visible(False)
fig.suptitle("Experiment (bars read from Fig. 5C/D) vs 2-D model, normalised to W2G10", y=1.0)
fig.tight_layout(); fig.savefig("ecl_model_vs_experiment.png", dpi=150)
print("device   exp(norm)  model peak(norm)  model integral(norm)  Epk")
for dev, e in [((2, 2), 22.5 / 14.4), ((2, 5), 19.3 / 14.4), ((2, 10), 1.0), ((2, 20), 10 / 14.4), ((5, 10), 15.8 / 14.4), ((10, 10), 16.9 / 14.4), ((20, 10), 14.8 / 14.4)]:
    r = R[dev]
    print("W%dG%-2d    %.2f       %.2f              %.2f                 %.2f V" % (*dev, e, r["peak"] / R[(2, 10)]["peak"], r["integral"] / R[(2, 10)]["integral"], r["Epk"]))
