"""Real protocol with cathodic ECL (k_lum_so), SAME bulk boundary height H for every device.
H = 300 um (> diffusion length over the 13 s experiment); W2G10 also at H = 600 um as a convergence check."""
import json, sys
import numpy as np
from multiprocessing import Pool
import ecl_gc_simulation as m
from ecl_gc_2d import Cell2D

V, E1, E2, EG, T_PRE = 0.1, 0.0, 0.8, -0.7, 5.0
K = float(sys.argv[1]) if len(sys.argv) > 1 else 1e4
DEVICES = [(20, 10, 300), (2, 20, 300), (10, 10, 300), (5, 10, 300), (2, 10, 300), (2, 5, 300), (2, 2, 300), (2, 10, 600)]


def one(a):
    w, g, H = a
    p = dict(m.PARAMS); p["k_lum_so"] = K
    cell = Cell2D(p, w=w * 1e-6, g=g * 1e-6, Eg=EG, H=H * 1e-6, nxs=(8, 18, 8), nz=40)
    tf = (E2 - E1) / V
    Ec = lambda t: None if t < T_PRE else E1 + V * (t - T_PRE)
    t = np.concatenate([np.linspace(0, T_PRE, 51)[:-1], T_PRE + np.linspace(0, tf, 161)])
    sol = cell.integrate(Ec, (0, T_PRE + tf), cell.y_bulk(), t)
    shape = (m.nS, cell.nx, cell.nz)
    obs = np.array([cell.observables(sol.y[:, k].reshape(shape), Ec(t[k]))[:3] for k in range(len(t))])
    sw = t >= T_PRE
    out = dict(w=w, g=g, H=H, k=K, t=t.tolist(), ecl=obs[:, 0].tolist(), peak=float(obs[sw, 0].max()),
               Epk=float(E1 + V * (t[sw][obs[sw, 0].argmax()] - T_PRE)), cath_end=float(obs[~sw, 0][-1]))
    print("done", a, "peak %.3e cath %.3e" % (out["peak"], out["cath_end"]), flush=True)
    return out


if __name__ == "__main__":
    with Pool(4) as p:
        res = p.map(one, DEVICES if K == 1e4 else [d for d in DEVICES if d[2] == 300])
    json.dump(res, open(f"cathodic_sameH_k{K:.0e}.json", "w"))
    print("ALL DONE")
