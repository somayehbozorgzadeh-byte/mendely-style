"""Spatial fields with cathodic ECL (k_lum_so = 1e4): end of the 5 s pre-hold (collector off) and at E_col = 0.4 V in the sweep."""
import numpy as np
from multiprocessing import Pool
import ecl_gc_simulation as m
from ecl_gc_2d import Cell2D

V, EG, T_PRE = 0.1, -0.7, 5.0


def one(dev):
    w, g = dev
    p = dict(m.PARAMS); p["k_lum_so"] = 1e4
    L = (w + g) * 1e-6
    cell = Cell2D(p, w=w * 1e-6, g=g * 1e-6, Eg=EG, H=max(40e-6, 4 * L), nxs=(8, 18, 8), nz=30)
    Ec = lambda t: None if t < T_PRE else V * (t - T_PRE)
    t = np.array([0, 0.3, 1.0, T_PRE - 1e-3, T_PRE + 4.0])
    sol = cell.integrate(Ec, (0, t[-1]), cell.y_bulk(), t)
    C = sol.y.reshape(m.nS, cell.nx, cell.nz, -1)
    em = np.stack([m.solution_rates(C[..., k], p)[1] * m.NA for k in range(len(t))], -1)
    np.savez(f"cath_field_W{w}G{g}.npz", c=C[[m.iSO, m.iRad, m.iHP, m.iLH]].astype(np.float32), em=em.astype(np.float32),
             t=t, xc=cell.xc, zc=cell.zc, w=w, g=g)
    print("done", dev, flush=True)


if __name__ == "__main__":
    with Pool(3) as pl:
        pl.map(one, [(2, 20), (2, 10), (2, 2)])
    print("ALL DONE")
