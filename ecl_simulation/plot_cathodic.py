"""Cathodic (pre-hold) + anodic-collector ECL, real protocol; comparison with the experimental gap/width series."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = json.load(open("cathodic_runs.json"))
V = json.load(open("cathodic_validation.json")) if os.path.exists("cathodic_validation.json") else []
K = sorted({r["k"] for r in R})
get = lambda w, g, k: next((r for r in R + V if r["w"] == w and r["g"] == g and r["k"] == k), None)
exp = {(2, 2): 22.5, (2, 5): 19.3, (2, 10): 14.4, (2, 20): 10.0, (5, 10): 15.8, (10, 10): 16.9, (20, 10): 14.8}
err = {(2, 2): 1.4, (2, 5): 1.4, (2, 10): 1.5, (2, 20): 0.6, (5, 10): 1.8, (10, 10): 1.9, (20, 10): 1.2}

fig = plt.figure(figsize=(16, 8.4))
gs = fig.add_gridspec(2, 3)
# traces for each k
for j, k in enumerate(K):
    a = fig.add_subplot(gs[0, j])
    for (w, g), c in zip([(2, 2), (2, 10), (2, 20)], ["tab:purple", "tab:green", "goldenrod"]):
        r = get(w, g, k)
        a.plot(r["t"], np.array(r["ecl"]) / 1e12, color=c, label=f"W{w}G{g}")
    a.axvspan(0, 5, color="tab:blue", alpha=.07); a.text(2.5, a.get_ylim()[1] * 0.92, "pre-hold\ngen −0.7 V, collector off\n(cathodic ECL)", ha="center", va="top", fontsize=8)
    a.text(9, a.get_ylim()[1] * 0.92, "collector sweep\n0 → 0.8 V, 0.1 V/s", ha="center", va="top", fontsize=8)
    for E in (0.2, 0.4, 0.6, 0.8):
        a.axvline(5 + E / 0.1, color="0.85", lw=0.6, zorder=0)
        a.text(5 + E / 0.1, a.get_ylim()[1] * 1.01, f"{E} V", ha="center", va="bottom", fontsize=7, color="0.4")
    a.set_xlabel("time / s"); a.set_ylabel("ECL / 10$^{12}$ photons m$^{-2}$ s$^{-1}$")
    a.set_title(f"k(LH$^-$ + O$_2^{{\\bullet-}}$) = {k:.0e} M$^{{-1}}$s$^{{-1}}$", fontsize=10, pad=24)
    a.legend(frameon=False, fontsize=8, loc="center left")
    a.spines["right"].set_visible(False)
# gap series comparison
a = fig.add_subplot(gs[1, 0:2])
devs = [(2, 2), (2, 5), (2, 10), (2, 20), (5, 10), (10, 10), (20, 10)]
x = np.arange(len(devs)); wd = 0.8 / (len(K) + 1)
a.bar(x - 0.4 + wd / 2, [exp[d] / 14.4 for d in devs], wd, yerr=[err[d] / 14.4 for d in devs], capsize=3, color="0.75", edgecolor="k", label="experiment")
cols = plt.cm.Reds(np.linspace(0.35, 0.9, len(K)))
for i, (k, c) in enumerate(zip(K, cols)):
    ref = get(2, 10, k)["peak"]
    vals = [get(*d, k)["peak"] / ref if get(*d, k) else np.nan for d in devs]
    a.bar(x - 0.4 + wd * (i + 1.5), vals, wd, color=c, label=f"model, k = {k:.0e}")
a.axvline(3.5, color="0.5", lw=0.8, ls=":")
a.text(1.5, a.get_ylim()[1] * 0.97, "gap series (W = 2 µm)", ha="center", va="top"); a.text(5, a.get_ylim()[1] * 0.97, "width series (G = 10 µm)", ha="center", va="top")
a.set_xticks(x, [f"W{d[0]}G{d[1]}" for d in devs]); a.set_ylabel("peak ECL in sweep / W2G10")
a.legend(frameon=False, fontsize=8, ncol=4, loc="upper right", bbox_to_anchor=(1, 0.9))
a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
a.set_title("Peak ECL during the collector sweep (includes the cathodic contribution), normalised to W2G10", fontsize=10)
# cathodic share
a = fig.add_subplot(gs[1, 2])
for k, c in zip(K, cols):
    gg = [2, 10, 20]
    a.plot(gg, [get(2, g, k)["cath_end"] / get(2, g, k)["peak"] for g in gg], "o-", color=c, label=f"k = {k:.0e}")
a.set_xlabel("gap / µm (W = 2 µm)"); a.set_ylabel("cathodic ECL (end of pre-hold) / sweep peak")
a.set_title("How bright the pre-hold (cathodic) ECL is", fontsize=10); a.legend(frameon=False, fontsize=8); a.set_ylim(0, 1.05)
a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
fig.tight_layout(); fig.savefig("ecl_cathodic_protocol.png", dpi=140)
print("device  exp   " + "  ".join(f"k={k:.0e}" for k in K))
for d in devs:
    print("W%dG%-2d  %.2f  " % (*d, exp[d] / 14.4) + "  ".join("%7s" % ("%.2f" % (get(*d, k)["peak"] / get(2, 10, k)["peak"]) if get(*d, k) else "-") for k in K))
