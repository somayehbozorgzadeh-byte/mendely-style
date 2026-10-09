"""Where the two ROS zones meet: O2.- zone (generator) vs L.- zone (collector) -> ECL zone."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator as RGI
import ecl_gc_simulation as m

d = np.load("flux_state_0.4.npz")
c, xc, zc = d["c"], d["xc"], d["zc"]
p = m.PARAMS
ph = m.solution_rates(c, p)[1] * m.NA
SO, RAD = c[m.iSO], c[m.iRad]
xr = np.linspace(xc[0], xc[-1], 400); zr = np.linspace(zc[0], 4e-6, 250)
P = np.stack(np.meshgrid(xr, zr, indexing="ij"), -1)
f = lambda a: RGI((xc, zc), a)(P)
so, rad, em = f(SO), f(RAD), f(ph)
nso, nrad = so / so.max(), rad / rad.max()

fig = plt.figure(figsize=(14, 7.5))
gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1])
a = fig.add_subplot(gs[0, :2])
rgb = np.ones(nso.shape + (3,))
rgb[..., 0] = 1 - 0.9 * nso          # blue zone : O2.-  (red channel reduced -> blue/cyan)
rgb[..., 1] = 1 - 0.9 * nso - 0.9 * nrad + 0.8 * nso * nrad
rgb[..., 2] = 1 - 0.9 * nrad         # red zone : L.-
rgb = np.clip(rgb, 0, 1)
# simple additive-subtractive composite: blue = O2.-, red = L.-, dark/purple = both
comp = np.ones_like(rgb)
comp[..., 0] = 1 - nso * 0.85
comp[..., 1] = 1 - nso * 0.55 - nrad * 0.85
comp[..., 2] = 1 - nrad * 0.85
comp = np.clip(comp, 0, 1)
a.imshow(np.transpose(comp, (1, 0, 2)), origin="lower", extent=[0, xr[-1] * 1e6, 0, zr[-1] * 1e6], aspect="equal")
a.contour(xr * 1e6, zr * 1e6, nso.T, levels=[0.1, 0.3], colors="tab:blue", linewidths=[0.8, 1.6])
a.contour(xr * 1e6, zr * 1e6, nrad.T, levels=[0.1, 0.3], colors="tab:red", linewidths=[0.8, 1.6])
a.contour(xr * 1e6, zr * 1e6, (em / em.max()).T, levels=[0.5], colors="k", linewidths=2, linestyles="--")
a.plot([0, 1], [0, 0], color="tab:blue", lw=8, solid_capstyle="butt", clip_on=False)
a.plot([3, 4], [0, 0], color="tab:red", lw=8, solid_capstyle="butt", clip_on=False)
a.text(0.5, -0.45, "GENERATOR\nO$_2$ + e$^-$ → O$_2^{\\bullet-}$", ha="center", va="top", color="tab:blue", fontsize=9)
a.text(3.5, -0.45, "COLLECTOR\nLH$^-$ → L$^{\\bullet-}$ + e$^-$", ha="center", va="top", color="tab:red", fontsize=9)
a.text(0.35, 2.6, "O$_2^{\\bullet-}$ zone", color="tab:blue", fontsize=12, weight="bold")
a.text(2.55, 2.6, "L$^{\\bullet-}$ zone", color="tab:red", fontsize=12, weight="bold")
a.annotate("ECL zone\n(L$^{\\bullet-}$ + O$_2^{\\bullet-}$ → 3-APA* → hν)", xy=(3.0, 0.45), xytext=(1.5, 1.5),
           arrowprops=dict(arrowstyle="->", lw=1.5), fontsize=10, ha="center")
a.set_xlabel("x / µm"); a.set_ylabel("z / µm"); a.set_ylim(0, 4)
a.set_title("Normalised concentrations (blue = O$_2^{\\bullet-}$, red = L$^{\\bullet-}$); contours: 10% / 30% of max; "
            "black dashed = 50% of peak emission", fontsize=9)

b = fig.add_subplot(gs[0, 2])
pc = b.pcolormesh(xr * 1e6, zr * 1e6, (em / 1e21).T, cmap="magma", shading="auto")
fig.colorbar(pc, ax=b, label="emission / 10$^{21}$ photons m$^{-3}$ s$^{-1}$")
b.set_aspect("equal"); b.set_xlabel("x / µm"); b.set_ylim(0, 4); b.set_title("Light generation rate", fontsize=10)
b.plot([0, 1], [0, 0], color="tab:blue", lw=8, solid_capstyle="butt", clip_on=False)
b.plot([3, 4], [0, 0], color="tab:red", lw=8, solid_capstyle="butt", clip_on=False)

for k, zq in enumerate([0.1e-6, 0.4e-6, 1.2e-6]):
    ax = fig.add_subplot(gs[1, k])
    j = np.argmin(abs(zr - zq))
    ax.plot(xr * 1e6, nso[:, j], color="tab:blue", label="O$_2^{\\bullet-}$ (norm.)")
    ax.plot(xr * 1e6, nrad[:, j], color="tab:red", label="L$^{\\bullet-}$ (norm.)")
    ax.fill_between(xr * 1e6, 0, np.minimum(nso[:, j], nrad[:, j]), color="gold", alpha=.6, label="overlap")
    ax2 = ax.twinx(); ax2.plot(xr * 1e6, em[:, j] / 1e21, "k--", label="emission"); ax2.set_ylabel("emission / 10$^{21}$", fontsize=8)
    ax.axvspan(0, 1, color="tab:blue", alpha=.08); ax.axvspan(3, 4, color="tab:red", alpha=.08)
    ax.set_xlabel("x / µm"); ax.set_title(f"cut at z = {zr[j]*1e6:.2f} µm", fontsize=10); ax.set_ylim(0, 1.05)
    if k == 0: ax.set_ylabel("normalised conc."); ax.legend(fontsize=7, loc="upper center", frameon=False)
    ax.spines["top"].set_visible(False)
fig.suptitle("Where the two ROS zones meet: w = 2 µm, gap = 2 µm, E$_{gen}$ = −0.8 V, E$_{col}$ = +0.4 V (steady state)", y=0.995)
fig.tight_layout(); fig.savefig("ecl_zones_meeting.png", dpi=150)
# location of emission centroid
A = np.outer(np.gradient(xr), np.gradient(zr))
print("emission centroid x = %.2f um, z = %.2f um" % (np.sum(em * A * xr[:, None]) / np.sum(em * A) * 1e6, np.sum(em * A * zr[None, :]) / np.sum(em * A) * 1e6))
