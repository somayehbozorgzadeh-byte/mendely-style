"""Ferrocenemethanol redox cycling on the real device: gold two-arm entwined spiral, W2G2, filling 1 mm x 1 mm
(circular spiral, outer radius ~500 um -> 61 turns). Generator 0 -> 0.5 -> 0 V at 0.1 V/s, collector 0 V or off."""
import json
import numpy as np
from multiprocessing import Pool
from redox_gc_array import Array

V, E2 = 0.1, 0.5
T = 2 * E2 / V
Eg = lambda t: V * t if t <= T / 2 else E2 - V * (t - T / 2)


def one(mode):
    arr = Array("spiral", w=2e-6, g=2e-6, n_pairs=61, r_in=8e-6, ncell=6)
    res, c = arr.cv(Eg, (lambda t: 0.0) if mode == "GC" else (lambda t: None), T, dt=0.0025 / V)
    out = dict(mode=mode, Eg=[float(r[1]) for r in res], i_gen=[float(r[3]) for r in res], i_col=[float(r[4]) for r in res],
               gen_area=arr.gen_area, r_out=float(arr.x_end))
    print("done", mode, "max i_gen %.3e A" % max(out["i_gen"]), flush=True)
    return out


if __name__ == "__main__":
    with Pool(2) as p:
        res = p.map(one, ["GC", "single"])
    json.dump(res, open("redox_device_spiral.json", "w"))
    print("ALL DONE")
