"""Verification against exact solutions (chemistry switched off, only the O2/O2.- couple).
1) Cottrell: diffusion-limited potential step at a planar electrode (1-D solver).
2) Aoki et al. (1988): steady-state redox-cycling current of an interdigitated array (2-D solver)."""
import json
import numpy as np
import ecl_gc_simulation as m
from ecl_gc_2d import Cell2D

F = m.F
OFF = dict(lum_total=0.0, k_ecl=0.0, k_disp=0.0, k_rad_o2=0.0, k_daq_hp=0.0, k_lum_so=0.0, kd_scale=0.0, k0_SO2=0.0, k0_HP=0.0,
           k0_O2=1e-2)
out = {}

# ---- 1) Cottrell (1-D code): generator stepped to -1.2 V, collector 500 um away, no lateral exchange
p = dict(m.PARAMS); p.update(OFF); p["gap"] = 500e-6; p["width"] = 1.0
t = np.logspace(-3, 0, 25)
r = m.run(p, Eg=-1.2, Ec=0.0, t_end=1.0, n=200, t_eval=np.concatenate([[0], t]))
i_num = r["i_gen"][1:]
i_cot = F * p["O2_bulk"] * np.sqrt(p["D"][m.iO2] / (np.pi * t))
out["cottrell"] = dict(t=t.tolist(), i_num=i_num.tolist(), i_exact=i_cot.tolist(),
                       max_rel_err=float(np.max(np.abs(i_num / i_cot - 1))))
print("Cottrell: max relative error over 1 ms - 1 s = %.2f%%" % (100 * out["cottrell"]["max_rel_err"]))

# ---- 2) Aoki IDA steady redox cycling (2-D code), equal D, both electrodes diffusion-limited
res = []
for w, g in [(2, 2), (2, 10), (10, 10)]:
    p = dict(m.PARAMS); p.update(OFF); D = 2.0e-9; p["D"] = np.full(m.nS, D)
    cell = Cell2D(p, w=w * 1e-6, g=g * 1e-6, Eg=-1.2, H=300e-6, nxs=(8, 18, 8), nz=40)
    sol = cell.integrate(0.4, (0, 60.0), cell.y_bulk(), np.array([0, 50.0, 60.0]))
    c = sol.y[:, -1].reshape(m.nS, cell.nx, cell.nz)
    _, ig, ic, _ = cell.observables(c, 0.4)
    i_cell = ic * (w * 1e-6 / 2)                   # collector current per unit length in the unit cell [A/m]
    x = w / g
    aoki = F * p["O2_bulk"] * D * (0.637 * np.log(2.55 * (1 + x)) - 0.19 / (1 + x) ** 2) / 2   # half a finger per unit cell
    drift = float(np.abs(sol.y[:, -1] - sol.y[:, -2]).max() / np.abs(sol.y[:, -1]).max())
    res.append(dict(w=w, g=g, i_num=float(i_cell), i_aoki=float(aoki), rel_err=float(i_cell / aoki - 1), drift=drift))
    print("Aoki W%dG%d: model %.3e A/m, Aoki %.3e A/m, difference %+.1f%% (drift %.0e)" % (w, g, i_cell, aoki, 100 * (i_cell / aoki - 1), drift))
out["aoki"] = res
json.dump(out, open("verify_analytic.json", "w"), indent=1)
