"""ECL for every experimental device as a full gold two-arm spiral filling 1 mm x 1 mm.
Protocol: generator -0.7 V from t = 0, collector off 5 s, then collector 0 -> 0.8 V at 0.1 V/s.
Chemistry: ecl_gc_simulation.PARAMS (gold-like) + cathodic route k_lum_so = 5e3.
Each device is saved to its own file; existing files are skipped, so the script can be re-run to resume."""
import os, sys, time
import numpy as np
from multiprocessing import Pool
from scipy.integrate import solve_ivp
import ecl_gc_simulation as m
from ecl_array import ECLArray

V, T_PRE, E2 = 0.1, 5.0, 0.8
R_OUT, R_IN = 500e-6, 8e-6
NCELL = int(os.environ.get("NCELL", 3))
DEVICES = [(2, 2), (2, 5), (2, 10), (5, 10), (2, 20), (10, 10), (20, 10)]       # largest first


def turns(w, g):
    return int((R_OUT - R_IN + g * 1e-6) // (2 * (w + g) * 1e-6))


def fname(w, g):
    return f"ecl_series_W{w}G{g}_n{NCELL}.npz"


def one(dev):
    w, g = dev
    if os.path.exists(fname(w, g)):
        return
    t0 = time.time()
    p = dict(m.PARAMS); p["k_lum_so"] = 5e3
    arr = ECLArray("spiral", p, w=w * 1e-6, g=g * 1e-6, n_pairs=turns(w, g), ncell=NCELL)
    Ec = lambda t: None if t < T_PRE else V * (t - T_PRE)
    t = np.concatenate([np.linspace(0, T_PRE, 26)[:-1], [T_PRE - 1e-3], T_PRE + np.linspace(0, E2 / V, 81)])
    sol = solve_ivp(arr.rhs(Ec), (0, t[-1]), arr.y_bulk(), method="BDF", jac_sparsity=arr.sparsity, t_eval=t, rtol=1e-5, atol=1e-11)
    if not sol.success:
        print("FAILED", dev, sol.message, flush=True); return
    shape = (m.nS, arr.nx, arr.nz)
    obs = [arr.observables(sol.y[:, k].reshape(shape), Ec(t[k]))[:3] for k in range(len(t))]
    k_pre = int(np.argmin(abs(t - (T_PRE - 1e-3)))); k_pk = int(np.argmin(abs(t - (T_PRE + 0.4 / V))))
    np.savez(fname(w, g) + ".tmp.npz", t=t, E=np.array([np.nan if Ec(x) is None else Ec(x) for x in t]),
             photons=np.array([o[0] for o in obs]), i_gen=np.array([o[1] for o in obs]), i_col=np.array([o[2] for o in obs]),
             c_pre=sol.y[:, k_pre].reshape(shape).astype(np.float32), c_pk=sol.y[:, k_pk].reshape(shape).astype(np.float32),
             xc=arr.xc, zc=arr.zc, Vol=arr.Vol, scale=arr.scale, gen_area=arr.gen_area, footprint=arr.footprint, turns=turns(w, g),
             segs=np.array([(s[0], s[1], {"G": 1, "C": 2, "-": 0}[s[2]]) for s in arr.segs]))
    os.replace(fname(w, g) + ".tmp.npz", fname(w, g))
    print("done", dev, "turns", turns(w, g), "cells", arr.nx * arr.nz, "%.0f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    todo = [d for d in DEVICES if not os.path.exists(fname(*d))]
    print("to do:", todo, flush=True)
    with Pool(4) as p:
        p.map(one, todo, chunksize=1)
    print("ALL DONE", flush=True)
