"""Transient after a simultaneous potential step (gen -0.8 V, col +0.4 V): when do O2.- and L.- reach across the gap?
W = 2 um, gaps 2/5/10/20 um. Records snapshots + mid-gap probe concentrations + ECL(t)."""
import numpy as np
from multiprocessing import Pool
import ecl_gc_simulation as m
from ecl_gc_2d import Cell2D

TS = np.concatenate([[0], np.logspace(-5, np.log10(3.0), 90)])


def one(g):
    w = 2
    L = (w + g) * 1e-6
    cell = Cell2D(m.PARAMS, w=w * 1e-6, g=g * 1e-6, Eg=-0.8, H=max(40e-6, 4 * L), nxs=(8, 18, 8), nz=30)
    sol = cell.integrate(0.4, (0, TS[-1]), cell.y_bulk(), TS)
    C = sol.y.reshape(m.nS, cell.nx, cell.nz, -1)
    ecl = np.array([cell.observables(C[..., k], 0.4)[0] for k in range(len(TS))])
    np.savez(f"transient_G{g}.npz", t=TS, C=C[[m.iSO, m.iRad, m.iHP]].astype(np.float32), xc=cell.xc, zc=cell.zc, ecl=ecl, w=w, g=g,
             emission=np.array([m.solution_rates(C[..., k], cell.p)[1] * m.NA for k in range(0, len(TS), 6)], dtype=np.float32),
             emission_t=TS[::6])
    print("done G", g, flush=True)


if __name__ == "__main__":
    with Pool(4) as p:
        p.map(one, [20, 10, 5, 2])
    print("ALL DONE")
