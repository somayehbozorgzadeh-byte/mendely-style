"""Concentration fields of FcMeOH / FcMeOH+ when the generator reaches 0.5 V (forward scan, 0.1 V/s, collector 0 V)."""
import numpy as np
from multiprocessing import Pool
from redox_gc_array import Array

V = 0.1


def one(a):
    geo, mode = a
    arr = Array(geo, w=2e-6, g=2e-6, n_pairs=25)
    _, c = arr.cv(lambda t: V * t, (lambda t: 0.0) if mode == "GC" else (lambda t: None), 0.5 / V, dt=0.0025 / V)
    np.savez(f"redox_profile_{geo}_{mode}.npz", c=c, xc=arr.xc, zc=arr.zc, nx=arr.nx, nz=arr.nz,
             segs=np.array([(s[0], s[1], {"G": 1, "C": 2, "-": 0}[s[2]]) for s in arr.segs]))
    print("done", a, flush=True)


if __name__ == "__main__":
    with Pool(3) as p:
        p.map(one, [("ida", "GC"), ("spiral", "GC"), ("ida", "single")])
    print("ALL DONE")
