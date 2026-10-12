"""Mesh-refinement test for ecl_simulation.py (wall-cell size, z spacing)."""
import sys, time, numpy as np
import ecl_simulation as e
CASES = [(9.0, 1, 1.0), (9.0, 1, 10.0), (12.0, 2, 1.0), (12.0, 2, 10.0)]
levels = [float(x) for x in sys.argv[1:]] or [0.5, 1.0, 2.0, 3.0]
rows = {}
for lv in levels:
    g = e.build(lv); r_e = g[0]; nz = g[3].size
    print(f"\nres={lv}: {g[2].size} x {nz} cells, wall cell {g[4][g[2].searchsorted(e.RI)-1]:.4f} um")
    for ph, rt, c in CASES:
        t0 = time.time(); res = e.simulate(ph, c, rt, g)
        r, p = e.top_view(res); ra, pa = e.top_view(res, "avg")
        out = dict(tel=e.tel(r, p), tel_avg=e.tel(ra, pa), cw=p[0] / p.max(), hv=res["hv"].max(),
                   hv_tot=float((res["hv"] * (np.pi*(g[0][1:]**2-g[0][:-1]**2))[:, None] * g[5][None, :]).sum()))
        rows[(lv, ph, c)] = out
        print(f"  pH {ph:4g} {c:4g} mM: TEL mid={out['tel']:.3f} avg={out['tel_avg']:.3f} "
              f"centre/wall={out['cw']:.4f} max hv={out['hv']:.4g} total hv={out['hv_tot']:.4g} ({time.time()-t0:.0f}s)")
print("\nRelative change vs finest level:")
fin = levels[-1]
for ph, rt, c in CASES:
    for lv in levels[:-1]:
        a, b = rows[(lv, ph, c)], rows[(fin, ph, c)]
        print(f"  pH {ph:g} {c:g} mM res {lv} vs {fin}: dTEL={100*(a['tel']/b['tel']-1):+.1f}%  "
              f"dTotalHv={100*(a['hv_tot']/b['hv_tot']-1):+.1f}%  dMaxHv={100*(a['hv']/b['hv']-1):+.1f}%")
