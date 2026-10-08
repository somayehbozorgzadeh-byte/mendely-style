"""COMSOL-style 2-D maps: O2.- concentration + flux streamlines, L.- layer, and ECL emission, for the gap series and the width series."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator as RGI
import ecl_gc_simulation as m

p = m.PARAMS


def load(w, g):
    d = np.load(f"field_W{w}G{g}.npz")
    return d["c"], d["xc"], d["zc"]


def prep(w, g, nx=260, nz=140):
    """Mirror the symmetry cell about x=0 (generator centre) and resample on a regular grid."""
    c, xc, zc = load(w, g)
    L = (w + g) * 1e-6
    zmax = min(L, 0.9 * zc[-1])
    xr = np.linspace(0, L, nx); zr = np.linspace(zc[0], zmax, nz)
    P = np.stack(np.meshgrid(xr, zr, indexing="ij"), -1)
    f = lambda a: RGI((xc, zc), a, bounds_error=False, fill_value=None)(P)
    so = f(c[m.iSO])
    gx, gz = np.gradient(c[m.iSO], xc, zc, edge_order=2)
    jx, jz = f(-p["D"][m.iSO] * gx), f(-p["D"][m.iSO] * gz)
    rad = f(c[m.iRad]); em = f(m.solution_rates(c, p)[1] * m.NA)
    # mirror about x = 0
    mir = lambda a, s=1: np.concatenate([s * a[::-1], a[1:]], axis=0)
    X = np.concatenate([-xr[::-1], xr[1:]])
    return dict(X=X * 1e6, Z=zr * 1e6, so=mir(so) * 1e3, jx=mir(jx, -1), jz=mir(jz), rad=mir(rad) * 1e6, em=mir(em) / 1e21, w=w, g=g, L=L * 1e6)


def figure(devs, title, fname, labels):
    D = [prep(*d) for d in devs]
    so_max = max(d["so"].max() for d in D); em_max = max(d["em"].max() for d in D)
    fig, ax = plt.subplots(3, 4, figsize=(17, 9.6))
    for j, d in enumerate(D):
        ext = [d["X"][0], d["X"][-1], 0, d["Z"][-1]]
        a = ax[0, j]
        im0 = a.imshow(d["so"].T, origin="lower", extent=ext, aspect="equal", cmap="turbo", vmin=0, vmax=so_max)
        spd = np.hypot(d["jx"], d["jz"]).T
        a.streamplot(d["X"], d["Z"], d["jx"].T, d["jz"].T, color="w", density=1.3, linewidth=0.5 + 1.5 * spd / spd.max(), arrowsize=0.8)
        b = ax[1, j]
        im1 = b.imshow(d["rad"].T, origin="lower", extent=ext, aspect="equal", cmap="turbo", vmin=0, vmax=max(x["rad"].max() for x in D))
        c_ = ax[2, j]
        im2 = c_.imshow(d["em"].T, origin="lower", extent=ext, aspect="equal", cmap="turbo", vmin=0, vmax=em_max)
        for r, a_ in enumerate((a, b, c_)):
            L, w = d["L"], d["w"]
            a_.plot([-w / 2, w / 2], [0, 0], color="k", lw=6, solid_capstyle="butt", clip_on=False)               # generator
            a_.plot([-L - w / 2, -L + w / 2], [0, 0], color="k", lw=6, solid_capstyle="butt", clip_on=False)      # collector (mirror)
            a_.plot([L - w / 2, L + w / 2], [0, 0], color="k", lw=6, solid_capstyle="butt", clip_on=False)        # collector
            a_.text(0, -0.04 * d["Z"][-1], "G", ha="center", va="top", fontsize=8)
            a_.text(L - w * 0.0, -0.04 * d["Z"][-1], "C", ha="center", va="top", fontsize=8, color="tab:red")
            a_.text(-L, -0.04 * d["Z"][-1], "C", ha="center", va="top", fontsize=8, color="tab:red")
            if r == 2: a_.set_xlabel("x / µm")
        a.set_title(labels[j], fontsize=12, weight="bold")
    for k, (im, lab) in enumerate([(im0, "O$_2^{\\bullet-}$ / µM  (white lines: diffusive flux)"), (im1, "L$^{\\bullet-}$ / nM"),
                                  (im2, "ECL emission / 10$^{21}$ photons m$^{-3}$ s$^{-1}$")]):
        cb = fig.colorbar(im, ax=ax[k, :], fraction=0.015, pad=0.01); cb.set_label(lab, fontsize=9)
        ax[k, 0].set_ylabel("z / µm")
    fig.suptitle(title, fontsize=12)
    fig.savefig(fname, dpi=130, bbox_inches="tight")


figure([(2, 2), (2, 5), (2, 10), (2, 20)],
       "Gap series (W = 2 µm), E$_{gen}$ = −0.8 V, E$_{col}$ = 0.4 V steady state. G = generator, C = collector (mirrored). "
       "Common colour scales, axes in µm", "ecl_fields_gap_series.png", ["W2 G2", "W2 G5", "W2 G10", "W2 G20"])
figure([(2, 10), (5, 10), (10, 10), (20, 10)],
       "Width series (G = 10 µm), E$_{gen}$ = −0.8 V, E$_{col}$ = 0.4 V steady state. Common colour scales, axes in µm",
       "ecl_fields_width_series.png", ["W2 G10", "W5 G10", "W10 G10", "W20 G10"])
print("ok")
