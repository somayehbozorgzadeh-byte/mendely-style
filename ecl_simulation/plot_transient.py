import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator as RGI

GAPS = [2, 5, 10, 20]
cols = dict(zip(GAPS, plt.cm.viridis([0.0, 0.3, 0.6, 0.9])))
D = {g: np.load(f"transient_G{g}.npz") for g in GAPS}
ZP = 0.3e-6                                            # probe height above the electrodes
SO, RAD = 0, 1


def probe(d, k, x):
    xc, zc = d["xc"], d["zc"]
    return np.array([RGI((xc, zc), d["C"][k][..., i])([[x, ZP]])[0] for i in range(len(d["t"]))])


def t_at(t, y, frac):
    y = np.asarray(y); yf = y[-1]
    i = np.where(y >= frac * yf)[0]
    return t[i[0]] if len(i) else np.nan


fig, ax = plt.subplots(2, 2, figsize=(12, 8.2))
rows = []
for g in GAPS:
    d = D[g]; t = d["t"][1:]; w = float(d["w"]) * 1e-6; gg = g * 1e-6
    xmid, xcol = w / 2 + gg / 2, w / 2 + gg - 0.05e-6
    so_mid, rad_mid = probe(d, SO, xmid)[1:], probe(d, RAD, xmid)[1:]
    so_col = probe(d, SO, xcol)[1:]
    ecl = d["ecl"][1:]
    ax[0, 0].semilogx(t * 1e3, ecl / ecl[-1], color=cols[g], label=f"G{g}")
    ax[0, 1].semilogx(t * 1e3, so_mid / so_mid[-1], color=cols[g], label=f"G{g}")
    ax[1, 0].semilogx(t * 1e3, so_col / so_col[-1], color=cols[g], label=f"G{g}")
    # superoxide front: distance from the generator edge where O2.- falls to 10% of its value at the generator edge (z = 0.3 um)
    xs = np.linspace(w / 2, w / 2 + gg, 200)
    fr = []
    for i in range(1, len(d["t"])):
        prof = RGI((d["xc"], d["zc"]), d["C"][SO][..., i])(np.c_[xs, np.full_like(xs, ZP)])
        j = np.where(prof >= 0.1 * prof.max())[0]
        fr.append((xs[j.max()] - w / 2) * 1e6 if prof.max() > 1e-9 else 0)
    ax[1, 1].loglog(t * 1e3, np.maximum(fr, 1e-3), color=cols[g], label=f"G{g}")
    rows.append((g, t_at(t, so_mid, .5) * 1e3, t_at(t, so_col, .5) * 1e3, t_at(t, rad_mid, .5) * 1e3 if rad_mid[-1] > 0 else np.nan,
                 t_at(t, ecl, .1) * 1e3, t_at(t, ecl, .5) * 1e3, t_at(t, ecl, .9) * 1e3))
tt = np.logspace(-2, 3.4, 50)
ax[1, 1].loglog(tt, np.sqrt(2 * 1.8e-9 * tt * 1e-3) * 1e6 * 1.0, "k:", label="√(2D$_{O_2^{•-}}$t)")
ax[0, 0].set_title("ECL(t)/ECL(steady)"); ax[0, 1].set_title("O$_2^{\\bullet-}$ at mid-gap (z = 0.3 µm), normalised")
ax[1, 0].set_title("O$_2^{\\bullet-}$ arriving at the collector edge, normalised"); ax[1, 1].set_title("O$_2^{\\bullet-}$ front: distance from generator edge")
for a in ax.ravel():
    a.set_xlabel("time after potential step / ms"); a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
    a.legend(frameon=False, fontsize=8)
ax[1, 1].set_ylabel("µm"); ax[0, 0].axhline(0.5, color="0.7", lw=.7); ax[0, 1].axhline(0.5, color="0.7", lw=.7)
fig.suptitle("When the generator's superoxide crosses the gap (W = 2 µm, potentials stepped at t = 0: E$_{gen}$ = −0.8 V, E$_{col}$ = +0.4 V)", fontsize=11)
fig.tight_layout(); fig.savefig("ecl_transient_arrival.png", dpi=150)
print("gap  t50 O2.- mid-gap   t50 O2.- at collector   t50 L.- mid-gap   ECL t10   t50   t90   [ms]")
for r in rows: print("G%-2d  %8.2f  %12.2f  %14s  %8.2f %8.2f %8.2f" % (r[0], r[1], r[2], "n/a" if np.isnan(r[3]) else "%.2f" % r[3], r[4], r[5], r[6]))

# snapshots for G10
d = D[10]; xc, zc = d["xc"], d["zc"]; t = d["t"]
times_ms = [0.3, 3, 30, 300, 3000]
idx = [int(np.argmin(abs(t - tm * 1e-3))) for tm in times_ms]
et = d["emission_t"]; eidx = [int(np.argmin(abs(et - t[i]))) for i in idx]
xr = np.linspace(xc[0], xc[-1], 300); zr = np.linspace(zc[0], 6e-6, 140)
P = np.stack(np.meshgrid(xr, zr, indexing="ij"), -1)
f = lambda a: RGI((xc, zc), a)(P)
fig, ax = plt.subplots(3, len(idx), figsize=(16, 7.4), sharex=True, sharey=True)
smax = max(f(d["C"][SO][..., i]).max() for i in idx) * 1e3; rmax = max(f(d["C"][RAD][..., i]).max() for i in idx) * 1e6
emax = max(f(d["emission"][e]).max() for e in eidx) / 1e21
for j, (i, e) in enumerate(zip(idx, eidx)):
    ext = [0, xr[-1] * 1e6, 0, 6]
    ax[0, j].imshow(f(d["C"][SO][..., i]).T * 1e3, origin="lower", extent=ext, aspect="equal", cmap="turbo", vmin=0, vmax=smax)
    ax[1, j].imshow(f(d["C"][RAD][..., i]).T * 1e6, origin="lower", extent=ext, aspect="equal", cmap="turbo", vmin=0, vmax=rmax)
    ax[2, j].imshow(f(d["emission"][e]).T / 1e21, origin="lower", extent=ext, aspect="equal", cmap="turbo", vmin=0, vmax=emax)
    ax[0, j].set_title(f"t = {t[i]*1e3:g} ms", fontsize=11, weight="bold")
    for r in range(3):
        ax[r, j].plot([0, 1], [0, 0], color="k", lw=6, solid_capstyle="butt", clip_on=False)
        ax[r, j].plot([11, 12], [0, 0], color="k", lw=6, solid_capstyle="butt", clip_on=False)
    ax[2, j].set_xlabel("x / µm")
for r, lab in enumerate(["O$_2^{\\bullet-}$ / µM", "L$^{\\bullet-}$ / nM", "emission / 10$^{21}$ photons m$^{-3}$ s$^{-1}$"]):
    ax[r, 0].set_ylabel("z / µm")
    ax[r, -1].text(1.02, 0.5, lab, transform=ax[r, -1].transAxes, rotation=90, va="center", fontsize=9)
fig.suptitle("W2 G10: the two zones after the potential step (generator at x = 0–1 µm, collector at x = 11–12 µm; common colour scales)", fontsize=11)
fig.tight_layout(); fig.savefig("ecl_transient_snapshots_G10.png", dpi=140)
