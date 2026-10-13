"""Measured peak ECL (Fig. 5C/D) vs model sweep-peak ECL for all 7 full-spiral devices; one common scale factor."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

devs = [(2, 2), (2, 5), (2, 10), (2, 20), (5, 10), (10, 10), (20, 10)]
e = np.array([22.5, 19.3, 14.4, 10.0, 15.8, 16.9, 14.8]); er = np.array([1.4, 1.4, 1.5, 0.6, 1.8, 1.9, 1.2])
m = []
for w, g in devs:
    d = np.load(f"ecl_series_W{w}G{g}_n3.npz"); sw = ~np.isnan(d["E"]); m.append(d["photons"][sw].max())
m = np.array(m)
f = np.exp(np.mean(np.log(e / m))); mf = m * f; z = (mf - e) / er
fig, ax = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw=dict(width_ratios=[1.6, 1]))
x = np.arange(7); wd = 0.38
ax[0].bar(x - wd / 2, e, wd, yerr=er, capsize=3, color="0.75", edgecolor="k", label="measured peak ECL")
ax[0].bar(x + wd / 2, mf, wd, color="tab:red", label="model peak ECL × one common factor")
for i, zz in enumerate(z): ax[0].text(i + wd / 2, mf[i] + 0.4, f"{zz:+.1f}σ", ha="center", fontsize=8)
ax[0].axvline(3.5, color="0.5", ls=":", lw=.8)
ax[0].text(1.5, 25.5, "gap series (W = 2 µm)", ha="center", fontsize=9); ax[0].text(5, 25.5, "width series (G = 10 µm)", ha="center", fontsize=9)
ax[0].set_xticks(x, [f"W{w}G{g}" for w, g in devs]); ax[0].set_ylabel("peak ECL / 10$^6$ a.u."); ax[0].set_ylim(0, 27)
ax[0].legend(frameon=False, fontsize=8, loc="upper right", bbox_to_anchor=(1, 0.9)); ax[0].set_title(f"χ²/N = {np.mean(z**2):.2f} over 7 devices (one scale factor, no other fitting)", fontsize=10)
ax[1].errorbar(e, mf, xerr=er, fmt="o", color="tab:red", capsize=3)
for (w, g), a, b in zip(devs, e, mf): ax[1].annotate(f"W{w}G{g}", (a, b), textcoords="offset points", xytext=(5, -10), fontsize=7)
lim = [8, 26]; ax[1].plot(lim, lim, "k--", lw=.8); ax[1].set_xlim(lim); ax[1].set_ylim(lim)
ax[1].set_xlabel("measured peak ECL / 10$^6$ a.u."); ax[1].set_ylabel("model / 10$^6$ a.u."); ax[1].set_title("Parity plot", fontsize=10)
for a in ax: a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
fig.suptitle("Peak ECL per CV: gold spirals in 1 mm × 1 mm, gen −0.7 V, 5 s pre-hold, collector 0→0.8 V at 0.1 V/s", fontsize=11)
fig.tight_layout(); fig.savefig("ecl_peak_model_vs_experiment.png", dpi=140)
print("factor", f, "chi2/N", np.mean(z ** 2))
