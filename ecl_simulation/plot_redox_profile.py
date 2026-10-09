import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator as RGI


def load(name):
    d = np.load(f"redox_profile_{name}.npz")
    nx, nz = int(d["nx"]), int(d["nz"])
    c = d["c"].reshape(2, nx, nz)
    return dict(R=c[0], O=c[1], xc=d["xc"], zc=d["zc"], segs=d["segs"])


def field(D, key, x0, x1, z1, nx=400, nz=160):
    xr = np.linspace(x0, x1, nx); zr = np.linspace(D["zc"][0], z1, nz)
    P = np.stack(np.meshgrid(xr, zr, indexing="ij"), -1)
    return xr, zr, RGI((D["xc"], D["zc"]), D[key], bounds_error=False, fill_value=None)(P)


def electrodes(a, D, x0, x1, y=0):
    for s in D["segs"]:
        if s[2] and s[1] > x0 and s[0] < x1:
            a.plot([max(s[0], x0) * 1e6, min(s[1], x1) * 1e6], [y, y], lw=6, solid_capstyle="butt", clip_on=False,
                   color="tab:blue" if s[2] == 1 else "tab:red")


ida, spi, single = load("ida_GC"), load("spiral_GC"), load("ida_single")
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.05])
# row 1-2: maps (IDA middle of array, IDA array edge, spiral middle) for FcMeOH and FcMeOH+
views = [(ida, 96e-6, 120e-6, "flat interdigitated – middle of array"),
         (ida, 180e-6, 215e-6, "flat interdigitated – outer edge of array"),
         (spi, 104e-6, 128e-6, "entwined spiral – middle (r ≈ 100–130 µm)")]
for j, (D, x0, x1, ttl) in enumerate(views):
    for r, (key, lab, cm) in enumerate([("R", "FcMeOH (reduced) / mM", "viridis"), ("O", "FcMeOH$^+$ (oxidised) / mM", "magma")]):
        a = fig.add_subplot(gs[r, j])
        xr, zr, v = field(D, key, x0, x1, 12e-6)
        im = a.imshow(v.T, origin="lower", extent=[x0 * 1e6, x1 * 1e6, 0, zr[-1] * 1e6], aspect="auto", cmap=cm, vmin=0, vmax=1)
        electrodes(a, D, x0, x1)
        if j == 2: fig.colorbar(im, ax=a, label=lab, fraction=0.05)
        if j == 0: a.set_ylabel("z / µm")
        if r == 0: a.set_title(ttl, fontsize=10)
        a.set_xlabel("x / µm" if D is ida else "r / µm")
# row 3: line profile across a few electrodes at z = 0.2 um, GC vs single-electrode mode
a = fig.add_subplot(gs[2, 0:2])
x0, x1 = 96e-6, 120e-6
for D, ls, lab in [(ida, "-", "generator-collector mode"), (single, ":", "collector off (generator only)")]:
    xr = np.linspace(x0, x1, 600)
    for key, col, nm in [("R", "tab:green", "FcMeOH"), ("O", "tab:purple", "FcMeOH$^+$")]:
        v = RGI((D["xc"], D["zc"]), D[key])(np.c_[xr, np.full_like(xr, 0.2e-6)])
        a.plot(xr * 1e6, v, ls=ls, color=col, label=f"{nm}, {lab}")
for s in ida["segs"]:
    if s[2] and s[1] > x0 and s[0] < x1:
        a.axvspan(max(s[0], x0) * 1e6, min(s[1], x1) * 1e6, color="tab:blue" if s[2] == 1 else "tab:red", alpha=0.12)
a.set_xlabel("x / µm  (blue band: generator, red band: collector)"); a.set_ylabel("concentration at z = 0.2 µm / mM")
a.set_title("Across the electrodes, 0.2 µm above the surface (flat interdigitated, middle of array)", fontsize=10)
a.legend(frameon=False, fontsize=8, ncol=2, loc="center right"); a.set_ylim(-0.02, 1.05)
a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
a = fig.add_subplot(gs[2, 2])
xm = 0.5 * (ida["segs"][48][0] + ida["segs"][48][1]) if ida["segs"][48][2] == 1 else None
# vertical profile above the centre of a generator finger in the middle and above the gap
gi = [s for s in ida["segs"] if s[2] == 1][12]; gap = [s for s in ida["segs"] if s[2] == 0][24]
for x, lab, ls in [(0.5 * (gi[0] + gi[1]), "above generator", "-"), (0.5 * (gap[0] + gap[1]), "above gap", "--")]:
    z = np.linspace(ida["zc"][0], 40e-6, 300)
    for key, col in [("R", "tab:green"), ("O", "tab:purple")]:
        a.plot(RGI((ida["xc"], ida["zc"]), ida[key])(np.c_[np.full_like(z, x), z]), z * 1e6, ls=ls, color=col,
               label=f"{'FcMeOH' if key == 'R' else 'FcMeOH$^+$'}, {lab}")
a.set_xlabel("concentration / mM"); a.set_ylabel("height above chip / µm"); a.set_title("Vertical profiles (GC mode)", fontsize=10)
a.legend(frameon=False, fontsize=7); a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
fig.suptitle("1 mM FcMeOH, W2G2, generator at 0.5 V on the forward scan (0.1 V/s), collector at 0 V", fontsize=11)
fig.tight_layout(); fig.savefig("redox_concentration_profiles.png", dpi=140)
print("ok")
