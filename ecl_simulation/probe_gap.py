"""Which superoxide-only constants move the G2/G20 ECL ratio? (steady state, E_col = 0.4 V, W = 2 um)"""
import numpy as np
from multiprocessing import Pool
import ecl_gc_simulation as m
from ecl_gc_2d import Cell2D

VARIANTS = {
    "baseline": {},
    "k_ecl 1e8": {"k_ecl": 1e8},
    "k_ecl 1e7": {"k_ecl": 1e7},
    "k0_SO2 1e-8 (little over-reduction)": {"k0_SO2": 1e-8},
    "k0_SO2 1e-6 (more over-reduction)": {"k0_SO2": 1e-6},
    "collector O2/O2.- k0 1e-9 (sluggish sink)": {"k0_O2_col": 1e-9},
    "dismutation x30": {"kd_scale": 30.0},
}


def solve(a):
    name, g = a
    p = dict(m.PARAMS); p.update(VARIANTS[name])
    L = (2 + g) * 1e-6
    cell = Cell2D(p, w=2e-6, g=g * 1e-6, Eg=-0.8, H=max(40e-6, 4 * L), nxs=(6, 14, 6), nz=24)
    sol = cell.integrate(0.4, (0, 12), cell.y_bulk(), np.array([0, 12.0]))
    c = sol.y[:, -1].reshape(m.nS, cell.nx, cell.nz)
    e = cell.observables(c, 0.4)[0]
    print("done", name, g, "%.3e" % e, flush=True)
    return (name, g, e)


if __name__ == "__main__":
    jobs = [(n, g) for n in VARIANTS for g in (2, 20)]
    with Pool(4) as p:
        r = p.map(solve, jobs)
    d = {(n, g): e for n, g, e in r}
    print("\nvariant -> ECL(G2)/ECL(G20)   [experiment: 2.25]")
    for n in VARIANTS:
        print("%-45s %.2f   (G20 abs %.2e)" % (n, d[(n, 2)] / d[(n, 20)], d[(n, 20)]))
