"""Hold [HO2-] at the electrode wall (two readings) and compare with the paper's Fig. 5 / S14 targets."""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import ecl_simulation as e, compare_with_paper as cp
g = e.build(2.0)
MODES = [("as printed (wall consumes HO2-)", None), ("no wall consumption of HO2-", "noflux"), ("[HO2-] pinned at wall (Dirichlet)", "dirichlet")]
print("Paper (10 mM H2O2): low pH max~10, wall ~8-10 along whole wall, ctr/edge~0.2, TEL>9.1 | high pH max~30, ring, TEL~4.1\n")
print(f"{'mode':34s} {'pka':>5s} {'pH':>3s} {'max':>6s} {'wall z=1':>8s} {'z=5':>6s} {'z=9':>6s} {'mouth':>6s} {'ctr/edge':>8s} {'TEL5%':>6s}")
res_all = {}
for pka in (11.6, -50.0):
    for name, mode in MODES:
        for ph, rt in ((9.0, 1), (12.0, 2)):
            res = e.simulate(ph, 10.0, rt, g, pka=pka, ho_wall=mode); res_all[(pka, mode, ph)] = res
            r_e, z_e, rc, zc, dr, dz = res["grid"]; hv = res["hv"] / 1000; iw = res["iw"]
            f = lambda z: hv[iw, np.argmin(abs(zc - z))]
            r, p = e.top_view(res)
            print(f"{name:34s} {pka:5.1f} {ph:3g} {hv.max():6.2f} {f(1):8.2f} {f(5):6.2f} {f(9):6.2f} {f(10.5):6.2f} "
                  f"{p[0]/p.max():8.3f} {cp.tel_thr(r,p,0.05):6.2f}")
fig, ax = plt.subplots(2, 3, figsize=(14, 9))
for k, (name, mode) in enumerate(MODES):
    for row, (ph, vmax) in enumerate(((9.0, 10), (12.0, 30))):
        res = res_all[(11.6, mode, ph)]; r_e, z_e, rc, zc, dr, dz = res["grid"]; hv = res["hv"] / 1000
        hv = np.ma.masked_where((rc[:, None] > e.RI) & (zc[None, :] < e.LT), hv)
        a = ax[row, k]; m = a.pcolormesh(r_e, z_e, hv.T, cmap="jet", vmin=0, vmax=vmax)
        a.set_xlim(0, 12); a.set_ylim(0, 20); a.set_aspect("equal"); a.set_title(f"pH {ph:g}: {name}", fontsize=9)
        plt.colorbar(m, ax=a, label="h$\\nu$ / mM")
plt.tight_layout(); plt.savefig("fix_ho2_wall.png", dpi=120)
