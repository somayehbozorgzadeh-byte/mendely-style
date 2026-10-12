"""Full-spiral ECL for all devices vs the measured ECL bars (Fig. 5C/D, read by eye)."""
import os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NCELL = int(os.environ.get("NCELL", 3))
EXP = {(2, 2): 22.5, (2, 5): 19.3, (2, 10): 14.4, (2, 20): 10.0, (5, 10): 15.8, (10, 10): 16.9, (20, 10): 14.8}
ERR = {(2, 2): 1.4, (2, 5): 1.4, (2, 10): 1.5, (2, 20): 0.6, (5, 10): 1.8, (10, 10): 1.9, (20, 10): 1.2}
devs = [d for d in [(2, 2), (2, 5), (2, 10), (2, 20), (5, 10), (10, 10), (20, 10)] if os.path.exists(f"ecl_series_W{d[0]}G{d[1]}_n{NCELL}.npz")]
D = {d: dict(np.load(f"ecl_series_W{d[0]}G{d[1]}_n{NCELL}.npz")) for d in devs}
summ = {}
for d, r in D.items():
    sw = ~np.isnan(r["E"]); ph = r["photons"]
    summ[d] = dict(peak=ph[sw].max(), Epk=r["E"][sw][ph[sw].argmax()], cath=ph[25], integ=np.trapezoid(ph[sw], r["t"][sw]))

fig, ax = plt.subplots(1, 3, figsize=(17, 4.6), gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
ref = (2, 10) if (2, 10) in D else devs[0]
x = np.arange(len(devs)); wd = 0.27
ax[0].bar(x - wd, [EXP[d] / EXP[ref] for d in devs], wd, yerr=[ERR[d] / EXP[ref] for d in devs], capsize=3, color="0.75", edgecolor="k", label="measured")
ax[0].bar(x, [summ[d]["peak"] / summ[ref]["peak"] for d in devs], wd, color="tab:red", label="model: sweep peak")
ax[0].bar(x + wd, [summ[d]["integ"] / summ[ref]["integ"] for d in devs], wd, color="tab:orange", label="model: integrated over sweep")
ax[0].set_xticks(x, [f"W{d[0]}G{d[1]}" for d in devs], rotation=30); ax[0].set_ylabel(f"ECL / W{ref[0]}G{ref[1]}")
ax[0].set_title("Whole-device ECL, 1 mm × 1 mm spirals (no refitting)", fontsize=10); ax[0].legend(frameon=False, fontsize=8)
cm = plt.cm.viridis(np.linspace(0, .9, len(devs)))
for d, c in zip(devs, cm):
    r = D[d]; ax[1].plot(r["t"], r["photons"] / 1e9, color=c, label=f"W{d[0]}G{d[1]}")
ax[1].axvspan(0, 5, color="tab:blue", alpha=.07); ax[1].set_xlabel("time / s"); ax[1].set_ylabel("ECL / 10$^9$ photons s$^{-1}$ (whole device)")
ax[1].set_title("ECL during pre-hold (shaded) and sweep", fontsize=10); ax[1].legend(frameon=False, fontsize=7, ncol=2)
ax[2].bar(x, [summ[d]["cath"] / summ[d]["peak"] for d in devs], color="tab:purple")
ax[2].set_xticks(x, [f"W{d[0]}G{d[1]}" for d in devs], rotation=30); ax[2].set_ylim(0, 1.05)
ax[2].set_ylabel("pre-hold ECL / sweep peak"); ax[2].set_title("Predicted brightness of the cathodic (pre-hold) glow", fontsize=10)
for a in ax: a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
fig.suptitle("ECL, gold W/G spirals in 1 mm × 1 mm: gen −0.7 V, 5 s pre-hold (collector off), collector 0→0.8 V at 0.1 V/s", fontsize=11)
fig.tight_layout(); fig.savefig(f"ecl_series_vs_experiment_n{NCELL}.png", dpi=140)
print("device turns  meas  peak  integ  Epk  cath/peak  photons/s(peak)")
for d in devs:
    s = summ[d]
    print(f"W{d[0]}G{d[1]:<2} {int(D[d]['turns']):4d}  {EXP[d]/EXP[ref]:.2f}  {s['peak']/summ[ref]['peak']:.2f}  {s['integ']/summ[ref]['integ']:.2f}  {s['Epk']:.2f}  {s['cath']/s['peak']:.2f}  {s['peak']:.3e}")
