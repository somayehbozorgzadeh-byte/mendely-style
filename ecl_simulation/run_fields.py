"""Steady-state 2-D fields (E_gen = -0.8 V, E_col = 0.4 V) for the gap series (W2 G2/5/10/20) and width series (W2/5/10/20 G10)."""
import numpy as np
from multiprocessing import Pool
import ecl_gc_simulation as m
from ecl_gc_2d import Cell2D

DEVICES = sorted({(2, g) for g in (2, 5, 10, 20)} | {(w, 10) for w in (2, 5, 10, 20)}, key=lambda d: -(d[0] + d[1]))


def one(dev):
    w, g = dev
    L = (w + g) * 1e-6
    cell = Cell2D(m.PARAMS, w=w * 1e-6, g=g * 1e-6, Eg=-0.8, H=max(40e-6, 4 * L), nxs=(8, 18, 8), nz=30)
    sol = cell.integrate(0.4, (0, 12), cell.y_bulk(), np.array([0, 11.0, 12.0]))
    c = sol.y[:, -1].reshape(m.nS, cell.nx, cell.nz)
    drift = np.abs(sol.y[:, -1] - sol.y[:, -2]).max() / np.abs(sol.y[:, -1]).max()
    np.savez(f"field_W{w}G{g}.npz", c=c, xc=cell.xc, zc=cell.zc, w=w, g=g)
    print("done", dev, "drift %.0e" % drift, flush=True)


if __name__ == "__main__":
    with Pool(4) as p:
        p.map(one, DEVICES)
    print("ALL DONE")
