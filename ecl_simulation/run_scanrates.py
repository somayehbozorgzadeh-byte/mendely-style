"""ECL-potential curves of the w=2um / gap=2um interdigitated device at several scan rates
(collector 0 -> 0.8 -> 0 V, generator -0.8 V). One process per scan rate."""
import sys
import numpy as np
from multiprocessing import Pool
import ecl_gc_simulation as m
from ecl_gc_2d import Cell2D

EG, E1, E2 = -0.8, 0.0, 0.8
RATES = [0.01, 0.05, 0.1, 0.5, 1.0]
GRID = dict(nxs=(8, 18, 8), nz=30)


def one(V):
    cell = Cell2D(m.PARAMS, w=2e-6, g=2e-6, Eg=EG, **GRID)
    tf = (E2 - E1) / V
    Ec = lambda t: E1 + V * t if t <= tf else E2 - V * (t - tf)
    y0 = cell.integrate(E1, (0, 6), cell.y_bulk(), np.array([0, 6.0])).y[:, -1]
    t = np.linspace(0, 2 * tf, 321)
    sol = cell.integrate(Ec, (0, 2 * tf), y0, t)
    shape = (m.nS, cell.nx, cell.nz)
    E = np.array([Ec(x) for x in t])
    obs = np.array([cell.observables(sol.y[:, k].reshape(shape), E[k])[:3] for k in range(len(t))])
    np.savez(f"scan_{V}.npz", t=t, E=E, ecl=obs[:, 0], ig=obs[:, 1], ic=obs[:, 2], tf=tf)
    print(f"done {V} V/s", flush=True)
    return V


if __name__ == "__main__":
    with Pool(len(RATES)) as p:
        p.map(one, RATES)
