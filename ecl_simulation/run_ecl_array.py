"""ECL on finite W2G2 arrays (flat interdigitated vs entwined spiral), your protocol:
generator -0.7 V from t = 0; collector off for 5 s; then collector 0 -> 0.8 V at 0.1 V/s. k_lum_so = 5e3 (fitted)."""
import sys, time
import numpy as np
from multiprocessing import Pool
import ecl_gc_simulation as m
from ecl_array import ECLArray

V, T_PRE, E2 = 0.1, 5.0, 0.8
NP = int(sys.argv[1]) if len(sys.argv) > 1 else 25


def one(geo):
    t0 = time.time()
    p = dict(m.PARAMS); p["k_lum_so"] = 5e3
    arr = ECLArray(geo, p, n_pairs=NP)
    Ec = lambda t: None if t < T_PRE else V * (t - T_PRE)
    t = np.concatenate([np.linspace(0, T_PRE, 26)[:-1], [T_PRE - 1e-3], T_PRE + np.linspace(0, E2 / V, 81)])
    sol = solve = None
    from scipy.integrate import solve_ivp
    sol = solve_ivp(arr.rhs(Ec), (0, t[-1]), arr.y_bulk(), method="BDF", jac_sparsity=arr.sparsity, t_eval=t, rtol=1e-5, atol=1e-11)
    if not sol.success:
        raise RuntimeError(sol.message)
    shape = (m.nS, arr.nx, arr.nz)
    obs = [arr.observables(sol.y[:, k].reshape(shape), Ec(t[k]))[:3] for k in range(len(t))]
    k_pre = int(np.argmin(abs(t - (T_PRE - 1e-3)))); k_pk = int(np.argmin(abs(t - (T_PRE + 0.4 / V))))
    np.savez(f"ecl_array_{geo}_N{NP}.npz", t=t, E=np.array([np.nan if Ec(x) is None else Ec(x) for x in t]),
             photons=np.array([o[0] for o in obs]), i_gen=np.array([o[1] for o in obs]), i_col=np.array([o[2] for o in obs]),
             c_pre=sol.y[:, k_pre].reshape(shape).astype(np.float32), c_pk=sol.y[:, k_pk].reshape(shape).astype(np.float32),
             xc=arr.xc, zc=arr.zc, Vol=arr.Vol, scale=arr.scale, gen_area=arr.gen_area, footprint=arr.footprint,
             segs=np.array([(s[0], s[1], {"G": 1, "C": 2, "-": 0}[s[2]]) for s in arr.segs]))
    print("done", geo, "N", NP, "cells", arr.nx * arr.nz, "%.0f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    with Pool(2) as pl:
        pl.map(one, ["ida", "spiral"])
    print("ALL DONE")
