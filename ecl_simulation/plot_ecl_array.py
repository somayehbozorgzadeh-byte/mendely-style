"""ECL on finite W2G2 arrays: same layout as the ferrocene figures (CV-type curves, maps, profiles)."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator as RGI
import ecl_gc_simulation as m

NP = int(sys.argv[1]) if len(sys.argv) > 1 else 25
p = dict(m.PARAMS); p["k_lum_so"] = 5e3
D = {g: dict(np.load(f"ecl_array_{g}_N{NP}.npz")) for g in ("ida", "spiral")}
COL = {"ida": "tab:blue", "spiral": "tab:red"}; LAB = {"ida": "flat interdigitated", "spiral": "entwined spiral (rings)"}

# ---------------- Fig 1: traces / CV-type curves
fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
for g, d in D.items():
    t, E = d["t"], d["E"]; sw = ~np.isnan(E)
    ax[0].plot(t, d["photons"] / float(d["gen_area"]) / 1e12 * 1e-6, color=COL[g], label=LAB[g])     # per um2 of generator
    ax[1].plot(E[sw], d["photons"][sw] / float(d["gen_area"]) * 1e-12 / 1e12, color=COL[g], label=LAB[g])
    ax[2].plot(E[sw], d["i_col"][sw] / float(d["gen_area"]) / 10, color=COL[g], label=f"{LAB[g]}: collector")
    ax[2].plot(E[sw], -d["i_gen"][sw] / float(d["gen_area"]) / 10, color=COL[g], ls="--", label=f"{LAB[g]}: generator (−)")
ax[0].axvspan(0, 5, color="tab:blue", alpha=.07); ax[0].text(2.5, ax[0].get_ylim()[1] * 0.9, "pre-hold: gen −0.7 V,\ncollector off", ha="center", va="top", fontsize=8)
ax[0].set_xlabel("time / s"); ax[0].set_ylabel("ECL / 10$^{6}$ photons s$^{-1}$ per µm² generator"); ax[0].set_title("ECL during the whole run", fontsize=10)
ax[1].set_xlabel("E$_{collector}$ / V vs Ag/AgCl"); ax[1].set_ylabel("ECL / 10$^{12}$ photons s$^{-1}$ per mm² generator"); ax[1].set_title("ECL–potential during collector sweep", fontsize=10)
ax[2].set_xlabel("E$_{collector}$ / V vs Ag/AgCl"); ax[2].set_ylabel("current / mA cm$^{-2}$ of generator"); ax[2].set_title("Collector and generator currents", fontsize=10)
for a in ax:
    a.legend(frameon=False, fontsize=7.5); a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
fig.suptitle(f"ECL with luminol, W2G2, {NP} pairs: flat interdigitated vs entwined spiral (gen −0.7 V, 5 s pre-hold, collector 0→0.8 V at 0.1 V/s)", fontsize=11)
fig.tight_layout(); fig.savefig(f"ecl_array_curves_N{NP}.png", dpi=140)


# ---------------- Fig 2: maps + profiles at the ECL peak (E_col = 0.4 V)
def fld(d, key, s, x0, x1, z1, nx=360, nz=150):
    c = d[key]; xc, zc = d["xc"], d["zc"]
    arr = m.solution_rates(c.astype(float), p)[1] * m.NA if s == "em" else c[s]
    xr = np.linspace(x0, x1, nx); zr = np.linspace(zc[0], z1, nz)
    P = np.stack(np.meshgrid(xr, zr, indexing="ij"), -1)
    return xr, zr, RGI((xc, zc), arr, bounds_error=False, fill_value=None)(P)


def electrodes(a, d, x0, x1):
    for s in d["segs"]:
        if s[2] and s[1] > x0 and s[0] < x1:
            a.plot([max(s[0], x0) * 1e6, min(s[1], x1) * 1e6], [0, 0], lw=6, solid_capstyle="butt", clip_on=False,
                   color="tab:blue" if s[2] == 1 else "tab:red")


segs = D["ida"]["segs"]; xend = segs[-1][1]; mid = segs[len(segs) // 2][0]
views = [("ida", mid - 12e-6, mid + 12e-6, "flat interdigitated – middle"), ("ida", xend - 18e-6, xend + 17e-6, "flat interdigitated – outer edge"),
         ("spiral", 104e-6, 128e-6, "entwined spiral – middle (r ≈ 100–130 µm)")]
rows = [(m.iSO, 1e3, "O$_2^{\\bullet-}$ / µM", "Blues"), (m.iRad, 1e6, "L$^{\\bullet-}$ / nM", "Reds"), ("em", 1e-21, "ECL emission / 10$^{21}$ m$^{-3}$ s$^{-1}$", "magma")]
for state, ttl_state in [("c_pk", "collector at 0.4 V (ECL peak)"), ("c_pre", "end of 5 s pre-hold (collector off: cathodic ECL)")]:
    fig = plt.figure(figsize=(16, 12.5))
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 1.1])
    vmax = {}
    for r, (s, sc, lab, cm) in enumerate(rows):
        vmax[r] = max(fld(D[g], state, s, x0, x1, 8e-6)[2].max() * sc for g, x0, x1, _ in views)
    for j, (g, x0, x1, ttl) in enumerate(views):
        for r, (s, sc, lab, cm) in enumerate(rows):
            a = fig.add_subplot(gs[r, j])
            xr, zr, v = fld(D[g], state, s, x0, x1, 8e-6)
            im = a.imshow(v.T * sc, origin="lower", extent=[x0 * 1e6, x1 * 1e6, 0, zr[-1] * 1e6], aspect="auto", cmap=cm, vmin=0, vmax=vmax[r])
            electrodes(a, D[g], x0, x1)
            if j == 2: fig.colorbar(im, ax=a, label=lab, fraction=0.05)
            if j == 0: a.set_ylabel("z / µm")
            if r == 0: a.set_title(ttl, fontsize=10)
            a.set_xlabel("x / µm" if g == "ida" else "r / µm", fontsize=8)
    # line profile at z = 0.2 um (IDA middle), normalised to each species' maximum in the window
    a = fig.add_subplot(gs[3, 0:2]); d = D["ida"]; x0, x1 = mid - 12e-6, mid + 12e-6
    xr = np.linspace(x0, x1, 600)
    for s, nm, col in [(m.iO2, "O$_2$", "0.5"), (m.iSO, "O$_2^{\\bullet-}$", "tab:blue"), (m.iLH, "luminol LH$^-$", "tab:green"),
                       (m.iRad, "L$^{\\bullet-}$", "tab:red"), (m.iHP, "H$_2$O$_2$", "tab:cyan"), ("em", "ECL emission", "k")]:
        arr = m.solution_rates(d[state].astype(float), p)[1] if s == "em" else d[state][s]
        v = RGI((d["xc"], d["zc"]), arr)(np.c_[xr, np.full_like(xr, 0.2e-6)])
        a.plot(xr * 1e6, v / max(v.max(), 1e-30), color=col, lw=2 if s == "em" else 1.3, ls="--" if s == "em" else "-",
               label=f"{nm} (max {v.max()*1e3:.3g} µM)" if s not in ("em", m.iRad) else (f"{nm} (max {v.max()*1e6:.3g} nM)" if s == m.iRad else nm))
    for sg in d["segs"]:
        if sg[2] and sg[1] > x0 and sg[0] < x1:
            a.axvspan(max(sg[0], x0) * 1e6, min(sg[1], x1) * 1e6, color="tab:blue" if sg[2] == 1 else "tab:red", alpha=0.12)
    a.set_xlabel("x / µm (blue band: generator, red band: collector)"); a.set_ylabel("normalised, z = 0.2 µm")
    a.set_title("Across the electrodes 0.2 µm above the surface (flat interdigitated, middle)", fontsize=10)
    a.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.17)); a.set_ylim(-0.02, 1.05)
    a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
    a = fig.add_subplot(gs[3, 2])
    for g in ("ida", "spiral"):
        d = D[g]; ph = m.solution_rates(d[state].astype(float), p)[1]
        inarr = (d["xc"] > d["segs"][0][0]) & (d["xc"] < d["segs"][-1][1])
        prof = (ph[inarr] * d["Vol"][inarr]).sum(0) / d["Vol"][inarr].sum(0)
        a.plot(prof / prof.max(), d["zc"] * 1e6, color=COL[g], label=LAB[g])
    a.set_ylim(0, 40); a.set_xlabel("ECL emission (array-averaged, normalised)"); a.set_ylabel("height above chip / µm")
    a.set_title("How high above the chip the light comes from", fontsize=10); a.legend(frameon=False, fontsize=8)
    a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
    fig.suptitle(f"Luminol ECL, W2G2 ({NP} pairs), gen −0.7 V: {ttl_state}. Common colour scales per row.", fontsize=11)
    fig.tight_layout(); fig.savefig(f"ecl_array_maps_{'peak' if state == 'c_pk' else 'prehold'}_N{NP}.png", dpi=130)

for g, d in D.items():
    E = d["E"]; sw = ~np.isnan(E); A = float(d["gen_area"])
    k = np.argmax(np.where(sw, d["photons"], -1))
    print(f"{g:7s} gen area {A*1e12:.0f} um2 | cathodic (end pre-hold) {d['photons'][25]/A*1e-12:.3e} photons/s/um2 gen | "
          f"sweep peak {d['photons'][k]/A*1e-12:.3e} at {E[k]:.2f} V | ratio {d['photons'][25]/d['photons'][k]:.2f} | "
          f"i_col(0.4 V) {np.interp(0.4, E[sw], d['i_col'][sw])*1e9:.1f} nA, i_gen {np.interp(0.4, E[sw], d['i_gen'][sw])*1e9:.1f} nA")
