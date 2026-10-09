"""Final comparison: cathodic + collector ECL with the same 300 um bulk boundary for all devices, k scan."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KS = [3e3, 5e3, 1e4]
R = {k: {(r["w"], r["g"]): r for r in json.load(open(f"cathodic_sameH_k{k:.0e}.json")) if r["H"] == 300} for k in KS}
exp = {(2, 2): 22.5, (2, 5): 19.3, (2, 10): 14.4, (2, 20): 10.0, (5, 10): 15.8, (10, 10): 16.9, (20, 10): 14.8}
err = {(2, 2): 1.4, (2, 5): 1.4, (2, 10): 1.5, (2, 20): 0.6, (5, 10): 1.8, (10, 10): 1.9, (20, 10): 1.2}
devs = [(2, 2), (2, 5), (2, 10), (2, 20), (5, 10), (10, 10), (20, 10)]
n = lambda k, d: R[k][d]["peak"] / R[k][(2, 10)]["peak"]

fig = plt.figure(figsize=(16, 8))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])
a = fig.add_subplot(gs[0, :])
x = np.arange(len(devs)); wd = 0.2
a.bar(x - 1.5 * wd, [exp[d] / 14.4 for d in devs], wd, yerr=[err[d] / 14.4 for d in devs], capsize=3, color="0.75", edgecolor="k", label="experiment")
for i, (k, c) in enumerate(zip(KS, plt.cm.Reds([0.4, 0.65, 0.9]))):
    a.bar(x + (i - 0.5) * wd, [n(k, d) for d in devs], wd, color=c, label=f"model, k = {k:.0e} M$^{{-1}}$s$^{{-1}}$")
a.axvline(3.5, color="0.5", ls=":", lw=.8)
a.set_xticks(x, [f"W{d[0]}G{d[1]}" for d in devs]); a.set_ylabel("peak ECL in sweep / W2G10")
a.set_title("Experiment vs model (cathodic + collector ECL; same 300 µm bulk boundary for all devices; gen −0.7 V, 5 s pre-hold, 0→0.8 V at 0.1 V/s)", fontsize=10)
a.text(1.5, 1.95, "gap series (W = 2 µm)", ha="center"); a.text(5, 1.95, "width series (G = 10 µm)", ha="center")
a.legend(frameon=False, fontsize=8, ncol=4, loc="upper right"); a.set_ylim(0, 2.1)
a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
for j, (grp, ttl) in enumerate([([(2, 2), (2, 10), (2, 20)], "gap series"), ([(2, 10), (10, 10), (20, 10)], "width series")]):
    a = fig.add_subplot(gs[1, j])
    for d, c in zip(grp, ["tab:purple", "tab:green", "goldenrod"]):
        r = R[5e3][d]
        a.plot(r["t"], np.array(r["ecl"]) / 1e12, color=c, label=f"W{d[0]}G{d[1]}")
    a.axvspan(0, 5, color="tab:blue", alpha=.07)
    a.text(2.5, a.get_ylim()[1] * 0.95, "pre-hold: gen −0.7 V,\ncollector off (cathodic ECL)", ha="center", va="top", fontsize=8)
    a.text(9, a.get_ylim()[1] * 0.95, "collector sweep 0→0.8 V\n(peak at ≈0.4 V, t ≈ 9 s)", ha="center", va="top", fontsize=8)
    a.set_xlabel("time / s"); a.set_ylabel("ECL / 10$^{12}$ photons m$^{-2}$ s$^{-1}$"); a.set_title(f"ECL traces, k = 5e3 ({ttl})", fontsize=10)
    a.legend(frameon=False, fontsize=8, loc="lower right"); a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
a = fig.add_subplot(gs[1, 2])
for k, c in zip(KS, plt.cm.Reds([0.4, 0.65, 0.9])):
    a.plot([f"W{d[0]}G{d[1]}" for d in devs], [R[k][d]["cath_end"] / R[k][d]["peak"] for d in devs], "o-", color=c, label=f"k = {k:.0e}")
a.set_ylim(0, 1.05); a.set_ylabel("pre-hold (cathodic) ECL / sweep peak"); a.tick_params(axis="x", rotation=45, labelsize=8)
a.set_title("Predicted brightness of the pre-hold glow", fontsize=10); a.legend(frameon=False, fontsize=8)
a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
fig.tight_layout(); fig.savefig("ecl_cathodic_sameH.png", dpi=140)
print("device  exp    " + "   ".join(f"k={k:.0e}" for k in KS) + "   cath/peak(k=5e3)")
for d in devs:
    print("W%dG%-2d  %.2f   " % (*d, exp[d] / 14.4) + "   ".join("%7.2f" % n(k, d) for k in KS) + "     %.2f" % (R[5e3][d]["cath_end"] / R[5e3][d]["peak"]))
