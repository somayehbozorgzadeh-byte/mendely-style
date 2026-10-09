"""COMSOL-style maps of cathodic ECL during the pre-hold vs ECL at the 0.4 V sweep peak (k_lum_so = 1e4)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator as RGI

DEVS = [(2, 2), (2, 10), (2, 20)]
STATES = [(3, "end of 5 s pre-hold\n(collector off: cathodic ECL)"), (4, "sweep, E$_{col}$ = 0.4 V\n(cathodic + collector ECL)")]
SO, RAD = 0, 1


def prep(w, g, k, zmax_um=8):
    d = np.load(f"cath_field_W{w}G{g}.npz"); xc, zc = d["xc"], d["zc"]
    L = (w + g) * 1e-6
    xr = np.linspace(0, L, 220); zr = np.linspace(zc[0], zmax_um * 1e-6, 140)
    P = np.stack(np.meshgrid(xr, zr, indexing="ij"), -1)
    f = lambda a: RGI((xc, zc), a, bounds_error=False, fill_value=None)(P)
    mir = lambda a: np.concatenate([a[::-1], a[1:]], 0)
    X = np.concatenate([-xr[::-1], xr[1:]]) * 1e6
    return dict(X=X, Z=zr * 1e6, so=mir(f(d["c"][SO][..., k])) * 1e3, rad=mir(f(d["c"][RAD][..., k])) * 1e6,
                em=mir(f(d["em"][..., k])) / 1e21, L=L * 1e6, w=w)


P = {(d, k): prep(*d, k) for d in DEVS for k, _ in STATES}
so_max = max(v["so"].max() for v in P.values()); rad_max = max(v["rad"].max() for v in P.values()); em_max = max(v["em"].max() for v in P.values())
fig, ax = plt.subplots(3, 6, figsize=(20, 8.4))
col = 0
for d in DEVS:
    for k, lab in STATES:
        v = P[(d, k)]; ext = [v["X"][0], v["X"][-1], 0, v["Z"][-1]]
        ims = []
        for r, (key, vmax, cmap) in enumerate([("so", so_max, "turbo"), ("rad", rad_max, "turbo"), ("em", em_max, "turbo")]):
            a = ax[r, col]
            ims.append(a.imshow(v[key].T, origin="lower", extent=ext, aspect="auto", cmap=cmap, vmin=0, vmax=vmax))
            L, w = v["L"], v["w"]
            a.plot([-w / 2, w / 2], [0, 0], color="w", lw=6, solid_capstyle="butt", clip_on=False)
            for xcen in (-L, L):
                a.plot([xcen - w / 2, xcen + w / 2], [0, 0], color="m", lw=6, solid_capstyle="butt", clip_on=False)
            if r == 2: a.set_xlabel("x / µm")
            if col == 0: a.set_ylabel("z / µm")
        ax[0, col].set_title(f"W{d[0]}G{d[1]}\n{lab}", fontsize=9)
        col += 1
for r, lab in enumerate(["O$_2^{\\bullet-}$ / µM", "L$^{\\bullet-}$ / nM", "ECL emission / 10$^{21}$ photons m$^{-3}$ s$^{-1}$"]):
    cb = fig.colorbar(ims[r] if False else ax[r, -1].images[0], ax=ax[r, :], fraction=0.012, pad=0.01); cb.set_label(lab, fontsize=8)
fig.suptitle("Cathodic ECL during the pre-hold vs. ECL at the sweep peak (k(LH$^-$+O$_2^{\\bullet-}$) = 10$^4$ M$^{-1}$s$^{-1}$, gen −0.7 V). "
             "White bar = generator, magenta bars = collector (mirrored). Common colour scales.", fontsize=10)
fig.savefig("ecl_cathodic_fields.png", dpi=130, bbox_inches="tight")

# emission height profiles (x-integrated) — where is the light, how high above the chip?
fig, ax = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
for a, d in zip(ax, DEVS):
    for k, lab in STATES:
        v = P[(d, k)]
        a.plot(v["em"].mean(0), v["Z"], label=lab.replace("\n", " "))
    a.set_title(f"W{d[0]}G{d[1]}"); a.set_xlabel("x-averaged emission / 10$^{21}$ m$^{-3}$ s$^{-1}$"); a.set_ylim(0, 8)
    a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
ax[0].set_ylabel("height above chip z / µm"); ax[0].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig("ecl_cathodic_height.png", dpi=140)
for d in DEVS:
    for k, lab in STATES:
        v = P[(d, k)]; em = v["em"]; Z = v["Z"]
        zc = (em.mean(0) * Z).sum() / em.mean(0).sum()
        print(f"W{d[0]}G{d[1]:<2} {lab.splitlines()[0]:<22} max O2.- {v['so'].max():6.1f} uM, max L.- {v['rad'].max():7.1f} nM, emission centroid height {zc:.2f} um")
