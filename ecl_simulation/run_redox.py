"""Ferrocenemethanol redox cycling, W2G2: flat interdigitated array vs entwined spiral.
Generator: 0 -> 0.5 -> 0 V at 0.1 V/s; collector held at 0 V (GC mode) or disconnected (single-electrode mode)."""
import json, time
import numpy as np
from multiprocessing import Pool
from redox_gc_array import Array

V, E1, E2 = 0.1, 0.0, 0.5
T = 2 * (E2 - E1) / V
Eg = lambda t: E1 + V * t if t <= T / 2 else E2 - V * (t - T / 2)


def one(a):
    geo, mode = a
    t0 = time.time()
    arr = Array(geo, w=2e-6, g=2e-6, n_pairs=25)
    res, c = arr.cv(Eg, (lambda t: 0.0) if mode == "GC" else (lambda t: None), T, dt=0.0025 / V)
    out = dict(geo=geo, mode=mode, t=[float(r[0]) for r in res], Eg=[float(r[1]) for r in res],
               i_gen=[float(r[3]) for r in res], i_col=[float(r[4]) for r in res],
               gen_area=arr.gen_area, cells=int(arr.nx * arr.nz))
    np.savez(f"redox_state_{geo}_{mode}.npz", c=c, xc=arr.xc, zc=arr.zc, nx=arr.nx, nz=arr.nz)
    print("done", a, "cells", out["cells"], "%.0f s" % (time.time() - t0), "max i_gen %.3e A" % max(out["i_gen"]), flush=True)
    return out


if __name__ == "__main__":
    with Pool(4) as pl:
        res = pl.map(one, [("ida", "GC"), ("spiral", "GC"), ("ida", "single"), ("spiral", "single")])
    json.dump(res, open("redox_fcmeoh.json", "w"))
    print("ALL DONE")
