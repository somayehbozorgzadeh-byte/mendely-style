"""Model vs experiment: collector swept 0 -> 0.8 V at 0.1 V/s, generator -0.8 V.
Gap series (W2 G2/5/10/20) and width series (W2/5/10/20 G10)."""
import json
import numpy as np
from multiprocessing import Pool
import ecl_gc_simulation as m
from ecl_gc_2d import Cell2D

V, E1, E2, EG = 0.1, 0.0, 0.8, -0.8
DEVICES = sorted({(2, g) for g in (2, 5, 10, 20)} | {(w, 10) for w in (2, 5, 10, 20)}, key=lambda d: -(d[0] + d[1]))


def one(dev):
    w, g = dev
    L = (w + g) * 1e-6
    cell = Cell2D(m.PARAMS, w=w * 1e-6, g=g * 1e-6, Eg=EG, H=max(40e-6, 4 * L), nxs=(8, 18, 8), nz=30)
    tf = (E2 - E1) / V
    Ec = lambda t: E1 + V * t
    y0 = cell.integrate(E1, (0, 8), cell.y_bulk(), np.array([0, 8.0])).y[:, -1]
    t = np.linspace(0, tf, 161)
    sol = cell.integrate(Ec, (0, tf), y0, t)
    shape = (m.nS, cell.nx, cell.nz)
    E = Ec(t)
    obs = np.array([cell.observables(sol.y[:, k].reshape(shape), E[k])[:3] for k in range(len(t))])
    ecl = obs[:, 0]
    out = dict(w=w, g=g, E=E.tolist(), ecl=ecl.tolist(), i_col=obs[:, 2].tolist(),
               peak=float(ecl.max()), Epk=float(E[ecl.argmax()]), integral=float(np.trapezoid(ecl, t)))
    print("done", dev, "peak %.3e at %.2f V" % (out["peak"], out["Epk"]), flush=True)
    return out


if __name__ == "__main__":
    with Pool(4) as p:
        res = p.map(one, DEVICES)
    json.dump(res, open("exp_sweep.json", "w"))
