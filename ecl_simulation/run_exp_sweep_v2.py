"""Experimental protocol: generator stepped to -0.7 V and collector disconnected (open circuit) for 5 s,
then collector swept 0 -> 0.8 V at 0.1 V/s (generator stays at -0.7 V)."""
import json
import numpy as np
from multiprocessing import Pool
import ecl_gc_simulation as m
from ecl_gc_2d import Cell2D

V, E1, E2, EG, T_PRE = 0.1, 0.0, 0.8, -0.7, 5.0
DEVICES = sorted({(2, g) for g in (2, 5, 10, 20)} | {(w, 10) for w in (2, 5, 10, 20)}, key=lambda d: -(d[0] + d[1]))


def one(dev):
    w, g = dev
    L = (w + g) * 1e-6
    cell = Cell2D(m.PARAMS, w=w * 1e-6, g=g * 1e-6, Eg=EG, H=max(40e-6, 4 * L), nxs=(8, 18, 8), nz=30)
    tf = (E2 - E1) / V
    # t < T_PRE: collector at open circuit (None); afterwards linear sweep from E1
    Ec = lambda t: None if t < T_PRE else E1 + V * (t - T_PRE)
    t = np.concatenate([np.linspace(0, T_PRE, 11)[:-1], T_PRE + np.linspace(0, tf, 161)])
    sol = cell.integrate(Ec, (0, T_PRE + tf), cell.y_bulk(), t)
    shape = (m.nS, cell.nx, cell.nz)
    E = np.array([Ec(x) for x in t])
    obs = np.array([cell.observables(sol.y[:, k].reshape(shape), E[k])[:3] for k in range(len(t))])
    sw = t >= T_PRE
    ecl = obs[sw, 0]; ts = t[sw] - T_PRE
    out = dict(w=w, g=g, E=E[sw].tolist(), ecl=ecl.tolist(), i_col=obs[sw, 2].tolist(), i_gen=obs[sw, 1].tolist(),
               peak=float(ecl.max()), Epk=float(E[sw][ecl.argmax()]), integral=float(np.trapezoid(ecl, ts)),
               ecl_pre=float(obs[~sw, 0][-1]))
    print("done", dev, "peak %.3e at %.2f V" % (out["peak"], out["Epk"]), flush=True)
    return out


if __name__ == "__main__":
    with Pool(4) as p:
        res = p.map(one, DEVICES)
    json.dump(res, open("exp_sweep_eg07.json", "w"))
    print("ALL DONE")
