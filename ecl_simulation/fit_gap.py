"""Fit the W2 gap series (G2/5/10/20) to experiment by tuning two constants:
   k_daq_hp : L(DAQ) + H2O2 -> 3-APA*   [M^-1 s^-1]   (gap-insensitive H2O2 light channel)
   k0_SO2   : O2.- over-reduction rate constant at generator [m/s]
Proxy for the CV signal: steady-state ECL at E_col = 0.4 and 0.7 V, take the larger.
Cost = sum over gaps of (ln model_ratio - ln exp_ratio)^2, ratios relative to G10."""
import json, itertools
import numpy as np
from multiprocessing import Pool
import ecl_gc_simulation as m
from ecl_gc_2d import Cell2D

GAPS = [2, 5, 10, 20]
EXP = {2: 22.5, 5: 19.3, 10: 14.4, 20: 10.0}
KH = [2e2, 2e3, 2e4, 2e5]
KS = [1e-8, 1e-7, 1e-6]
ECS = [0.4, 0.7]
GRID = dict(nxs=(6, 14, 6), nz=24)


def solve(args):
    kh, ks, w, g, Ec = args
    p = dict(m.PARAMS); p["k_daq_hp"] = kh; p["k0_SO2"] = ks
    L = (w + g) * 1e-6
    cell = Cell2D(p, w=w * 1e-6, g=g * 1e-6, Eg=-0.8, H=max(40e-6, 4 * L), **GRID)
    sol = cell.integrate(Ec, (0, 12), cell.y_bulk(), np.array([0, 12.0]))
    c = sol.y[:, -1].reshape(m.nS, cell.nx, cell.nz)
    out = dict(kh=kh, ks=ks, w=w, g=g, Ec=Ec, ecl=cell.observables(c, Ec)[0])
    print("done", kh, ks, w, g, Ec, "%.2e" % out["ecl"], flush=True)
    return out


if __name__ == "__main__":
    jobs = [(kh, ks, 2, g, e) for kh in KH for ks in KS for g in GAPS for e in ECS]
    jobs.sort(key=lambda j: -j[3])
    with Pool(4) as p:
        res = p.map(solve, jobs)
    json.dump(res, open("fit_gap_grid.json", "w"))
