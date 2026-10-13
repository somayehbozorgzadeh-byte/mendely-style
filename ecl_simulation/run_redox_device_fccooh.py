"""Ferrocenecarboxylic acid redox cycling on the real device (gold W2G2 spiral in 1 mm x 1 mm, 61 turns).
Generator 0 -> 0.6 -> 0 V at 0.1 V/s; collector at 0 V (GC) or disconnected (single)."""
import json
import numpy as np
from multiprocessing import Pool
from redox_gc_array import Array, FCCOOH

V, E2 = 0.1, 0.6
T = 2 * E2 / V
Eg = lambda t: V * t if t <= T / 2 else E2 - V * (t - T / 2)


def one(mode):
    arr = Array("spiral", w=2e-6, g=2e-6, n_pairs=61, r_in=8e-6, ncell=6, p=FCCOOH)
    res, c = arr.cv(Eg, (lambda t: 0.0) if mode == "GC" else (lambda t: None), T, dt=0.0025 / V)
    out = dict(mode=mode, Eg=[float(r[1]) for r in res], i_gen=[float(r[3]) for r in res], i_col=[float(r[4]) for r in res],
               gen_area=arr.gen_area, r_out=float(arr.x_end))
    print("done", mode, "max i_gen %.3e A" % max(out["i_gen"]), flush=True)
    return out


if __name__ == "__main__":
    with Pool(2) as p:
        res = p.map(one, ["GC", "single"])
    json.dump(res, open("redox_device_spiral_fccooh.json", "w"))
    print("ALL DONE")
