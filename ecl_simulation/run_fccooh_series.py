"""FcCOOH (1 mM) redox cycling for all experimental devices, each a two-arm gold spiral filling 1 mm x 1 mm
(outer radius <= ~500 um, so the number of turns falls with pitch). Generator 0 -> 0.6 V at 0.1 V/s, collector 0 V."""
import json
import numpy as np
from multiprocessing import Pool
from redox_gc_array import Array, FCCOOH

V, E2 = 0.1, 0.6
R_OUT, R_IN = 500e-6, 8e-6
DEVICES = [(2, 2), (2, 5), (2, 10), (2, 20), (5, 10), (10, 10), (20, 10)]


def turns(w, g):
    return int((R_OUT - R_IN + g * 1e-6) // (2 * (w + g) * 1e-6))


def one(dev):
    w, g = dev
    arr = Array("spiral", w=w * 1e-6, g=g * 1e-6, n_pairs=turns(w, g), r_in=R_IN, ncell=6, p=FCCOOH)
    res, _ = arr.cv(lambda t: V * t, lambda t: 0.0, E2 / V, dt=0.0025 / V)
    out = dict(w=w, g=g, turns=turns(w, g), r_out=float(arr.x_end), gen_area=arr.gen_area,
               Eg=[float(r[1]) for r in res], i_gen=[float(r[3]) for r in res], i_col=[float(r[4]) for r in res])
    print("done", dev, "turns", out["turns"], "i_gen(0.6) %.3e" % out["i_gen"][-1], flush=True)
    return out


if __name__ == "__main__":
    with Pool(2) as p:
        res = p.map(one, DEVICES)
    json.dump(res, open("fccooh_series.json", "w"))
    print("ALL DONE")
