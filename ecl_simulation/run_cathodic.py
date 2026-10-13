"""Cathodic ECL during the 5 s pre-hold: add LH- + O2.- -> L.- + H2O2 (k_lum_so) and run the real protocol
(gen -0.7 V, collector open circuit 5 s, then collector 0 -> 0.8 V at 0.1 V/s). Full ECL trace saved."""
import json
import numpy as np
from multiprocessing import Pool
import ecl_gc_simulation as m
from ecl_gc_2d import Cell2D

V, E1, E2, EG, T_PRE = 0.1, 0.0, 0.8, -0.7, 5.0
KX = [1e2, 1e3, 1e4]
DEVICES = [(2, 20), (2, 10), (2, 2)]


def one(a):
    (w, g), kx = a
    p = dict(m.PARAMS); p["k_lum_so"] = kx
    L = (w + g) * 1e-6
    cell = Cell2D(p, w=w * 1e-6, g=g * 1e-6, Eg=EG, H=max(40e-6, 4 * L), nxs=(8, 18, 8), nz=30)
    tf = (E2 - E1) / V
    Ec = lambda t: None if t < T_PRE else E1 + V * (t - T_PRE)
    t = np.concatenate([np.linspace(0, T_PRE, 51)[:-1], T_PRE + np.linspace(0, tf, 161)])
    sol = cell.integrate(Ec, (0, T_PRE + tf), cell.y_bulk(), t)
    shape = (m.nS, cell.nx, cell.nz)
    obs = np.array([cell.observables(sol.y[:, k].reshape(shape), Ec(t[k]))[:3] for k in range(len(t))])
    sw = t >= T_PRE
    out = dict(w=w, g=g, k=kx, t=t.tolist(), E=[None if Ec(x) is None else Ec(x) for x in t], ecl=obs[:, 0].tolist(),
               peak=float(obs[sw, 0].max()), cath_end=float(obs[~sw, 0][-1]), base_sweep_start=float(obs[sw, 0][0]))
    print("done", (w, g), kx, "peak %.3e  cathodic(end of hold) %.3e" % (out["peak"], out["cath_end"]), flush=True)
    return out


if __name__ == "__main__":
    jobs = [(d, k) for d in DEVICES for k in KX]
    with Pool(4) as p:
        res = p.map(one, jobs)
    json.dump(res, open("cathodic_runs.json", "w"))
    print("ALL DONE")
