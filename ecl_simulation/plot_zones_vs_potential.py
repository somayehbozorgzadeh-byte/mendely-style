"""How the O2.- zone (generator) and L.- zone (collector) meet as the collector potential rises (w2 g2)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator as RGI
import ecl_gc_simulation as m

POTS = [0.2, 0.3, 0.4, 0.5, 0.8]
p = m.PARAMS
xr = np.linspace(0, 4e-6, 300); zr = np.linspace(0, 2.5e-6, 150)
data = {}
for E in POTS:
    d = np.load(f"flux_state_{E}.npz"); c, xc, zc = d["c"], d["xc"], d["zc"]
    P = np.stack(np.meshgrid(np.clip(xr, xc[0], xc[-1]), np.clip(zr, zc[0], zc[-1]), indexing="ij"), -1)
    f = lambda a: RGI((xc, zc), a)(P)
    data[E] = dict(so=f(c[m.iSO]), rad=f(c[m.iRad]), em=f(m.solution_rates(c, p)[1] * m.NA))
so_max = max(v["so"].max() for v in data.values())
rad_max = max(v["rad"].max() for v in data.values())
em_max = max(v["em"].max() for v in data.values())

fig, ax = plt.subplots(3, len(POTS), figsize=(16, 7.6), sharex=True, sharey=True)
for j, E in enumerate(POTS):
    v = data[E]
    nso, nrad = v["so"] / so_max, v["rad"] / rad_max            # common scale across potentials
    comp = np.ones(nso.shape + (3,))
    comp[..., 0] = 1 - 0.85 * nso
    comp[..., 1] = 1 - 0.55 * nso - 0.85 * nrad
    comp[..., 2] = 1 - 0.85 * nrad
    comp = np.clip(comp, 0, 1)
    ext = [0, 4, 0, 2.5]
    ax[0, j].imshow(np.transpose(comp, (1, 0, 2)), origin="lower", extent=ext, aspect="equal")
    ax[0, j].contour(xr * 1e6, zr * 1e6, (v["em"] / em_max).T, levels=[0.5], colors="k", linestyles="--", linewidths=1.5)
    ax[0, j].set_title(f"E$_{{col}}$ = {E} V", fontsize=11)
    ax[1, j].imshow(np.transpose(v["rad"] / rad_max, (1, 0)), origin="lower", extent=ext, aspect="equal", cmap="Reds", vmin=0, vmax=1)
    ax[1, j].contour(xr * 1e6, zr * 1e6, (v["so"] / so_max).T, levels=[0.1, 0.25], colors="tab:blue", linewidths=[0.8, 1.5])
    pcm = ax[2, j].imshow(np.transpose(v["em"] / 1e21, (1, 0)), origin="lower", extent=ext, aspect="equal", cmap="magma", vmin=0, vmax=em_max / 1e21)
    for r in range(3):
        ax[r, j].plot([0, 1], [0, 0], color="tab:blue", lw=6, solid_capstyle="butt", clip_on=False)
        ax[r, j].plot([3, 4], [0, 0], color="tab:red", lw=6, solid_capstyle="butt", clip_on=False)
ax[0, 0].set_ylabel("overlay\nblue O$_2^{\\bullet-}$ / red L$^{\\bullet-}$\n(dashed: 50% emission)\nz / µm")
ax[1, 0].set_ylabel("L$^{\\bullet-}$ zone (red)\nwith O$_2^{\\bullet-}$ contours (blue: 10%, 25%)\nz / µm")
ax[2, 0].set_ylabel("ECL emission\nz / µm")
for a in ax[2]: a.set_xlabel("x / µm")
fig.colorbar(pcm, ax=ax[2, :], label="emission / 10$^{21}$ photons m$^{-3}$ s$^{-1}$", fraction=0.02, pad=0.01)
fig.suptitle("W2 G2 device, E$_{gen}$ = −0.8 V: how the O$_2^{\\bullet-}$ and L$^{\\bullet-}$ zones meet as the collector potential increases "
             "(common colour scales; generator at x = 0–1 µm, collector at x = 3–4 µm)", fontsize=10)
fig.savefig("ecl_zones_vs_potential.png", dpi=150, bbox_inches="tight")

print("E_col  L.- layer thickness(10% of max, above collector centre, um)   O2.- at collector surface (rel)   peak emission(rel)")
for E in POTS:
    v = data[E]; i = np.argmin(abs(xr - 3.8e-6))
    prof = v["rad"][i] / v["rad"][i].max()
    th = zr[np.where(prof > 0.1)[0].max()] * 1e6
    print("%.1f   %.2f    %.3f    %.2f" % (E, th, v["so"][i, 0] / so_max, v["em"].max() / em_max))
