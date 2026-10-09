"""Verification 3: mesh and solver-tolerance convergence of the quantity that matters for the comparison (ECL G2/G20 ratio).
Full chemistry incl. cathodic route (k_lum_so = 5e3), gen -0.7 V, collector held at 0.4 V, H = 300 um, t = 8 s."""
import json
import numpy as np
from multiprocessing import Pool
from scipy.integrate import solve_ivp
import ecl_gc_simulation as m
from ecl_gc_2d import Cell2D

GRIDS = {"coarse": dict(nxs=(6, 12, 6), nz=30), "base (used)": dict(nxs=(8, 18, 8), nz=40), "fine": dict(nxs=(14, 32, 14), nz=64)}


def one(a):
    name, g, rtol = a
    p = dict(m.PARAMS); p["k_lum_so"] = 5e3
    cell = Cell2D(p, w=2e-6, g=g * 1e-6, Eg=-0.7, H=300e-6, **GRIDS[name])
    sol = solve_ivp(cell.rhs(0.4), (0, 8), cell.y_bulk(), method="BDF", jac_sparsity=cell.sparsity, t_eval=[8.0], rtol=rtol, atol=1e-12)
    c = sol.y[:, -1].reshape(m.nS, cell.nx, cell.nz)
    e, ig, ic, _ = cell.observables(c, 0.4)
    print("done", a, "%.4e" % e, flush=True)
    return dict(grid=name, g=g, rtol=rtol, ecl=e, i_gen=ig, i_col=ic, n=int(cell.nx * cell.nz))


if __name__ == "__main__":
    jobs = [("fine", 20, 1e-5), ("fine", 2, 1e-5), ("base (used)", 2, 1e-7), ("base (used)", 20, 1e-7),
            ("base (used)", 2, 1e-5), ("base (used)", 20, 1e-5), ("coarse", 2, 1e-5), ("coarse", 20, 1e-5)]
    with Pool(4) as pl:
        res = pl.map(one, jobs)
    json.dump(res, open("verify_grid.json", "w"), indent=1)
    print("ALL DONE")
