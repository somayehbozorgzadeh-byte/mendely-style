"""Scan k4 (L + HO2- -> LO2^2-) to fit the paper's low-pH ECL profile (a FIT, not part of the SI parameter set)."""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import ecl_simulation as e, compare_with_paper as cp
g = e.build(2.0)
SC = [1, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 3e-4]
print("Targets (paper, pH 9, 10 mM): peak ~10 mM, wall ~8-10 mM along z, top-view ctr/edge ~0.2, TEL(5%) > 9.1 um\n")
rows = {}
for mode in (None, "dirichlet"):
    print(f"--- HO2- wall treatment: {mode or 'as printed'} ---")
    print(f"{'k4 (M-1s-1)':>12s} {'max':>6s} {'wall z=1':>8s} {'z=5':>6s} {'z=9':>6s} {'mouth':>6s} {'ctr/edge mid':>12s} {'ctr/edge avg':>12s} {'TEL5% mid':>9s}")
    for sc in SC:
        res = e.simulate(9.0, 10.0, 1, g, ho_wall=mode, k4_scale=sc); rows[(mode, sc)] = res
        r_e, z_e, rc, zc, dr, dz = res["grid"]; hv = res["hv"] / 1000; iw = res["iw"]
        f = lambda z: hv[iw, np.argmin(abs(zc - z))]
        r, p = e.top_view(res); ra, pa = e.top_view(res, "avg")
        print(f"{5e7*sc:12.2e} {hv.max():6.2f} {f(1):8.2f} {f(5):6.2f} {f(9):6.2f} {f(10.5):6.2f} {p[0]/p.max():12.3f} {pa[0]/pa.max():12.3f} {cp.tel_thr(r,p,0.05):9.2f}")
fig, ax = plt.subplots(2, 4, figsize=(16, 8))
for k, sc in enumerate((1, 0.03, 0.003, 3e-4)):
    for row, mode in enumerate((None, "dirichlet")):
        res = rows[(mode, sc)]; r_e, z_e, rc, zc, dr, dz = res["grid"]; hv = np.ma.masked_where((rc[:, None] > e.RI) & (zc[None, :] < e.LT), res["hv"] / 1000)
        a = ax[row, k]; m = a.pcolormesh(r_e, z_e, hv.T, cmap="jet", vmin=0, vmax=10); a.set_xlim(0, 12); a.set_ylim(0, 20); a.set_aspect("equal")
        a.set_title(f"k4={5e7*sc:.1e}, HO2- wall: {mode or 'printed'}", fontsize=9); plt.colorbar(m, ax=a)
plt.tight_layout(); plt.savefig("k4_scan.png", dpi=110)
