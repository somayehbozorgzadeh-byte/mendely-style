"""Model vs experiment: gap series (W2) and width series (G10) of interdigitated/spiral devices."""
import json
import numpy as np
from multiprocessing import Pool
import ecl_gc_simulation as m
from ecl_gc_2d import Cell2D

GAPS = [2, 5, 10, 20]      # W2 G*
WIDTHS = [2, 5, 10, 20]    # W* G10
ECS = [0.4, 0.7]


def one(args):
    w, g, Ec = args
    L = (w + g) * 1e-6
    cell = Cell2D(m.PARAMS, w=w * 1e-6, g=g * 1e-6, Eg=-0.8, H=max(40e-6, 4 * L), nxs=(8, 18, 8), nz=30)
    tend = 14.0
    sol = cell.integrate(Ec, (0, tend), cell.y_bulk(), np.array([0, tend - 1, tend]))
    c = sol.y[:, -1].reshape(m.nS, cell.nx, cell.nz)
    ecl_foot, ig, ic, _ = cell.observables(c, Ec)
    drift = float(np.abs(sol.y[:, -1] - sol.y[:, -2]).max() / np.abs(sol.y[:, -1]).max())
    print("done", w, g, Ec, "drift %.0e" % drift, flush=True)
    return dict(w=w, g=g, Ec=Ec, ecl_foot=ecl_foot, i_col=ic, i_gen=ig, drift=drift)


if __name__ == "__main__":
    jobs = sorted({(2, g, e) for g in GAPS for e in ECS} | {(w, 10, e) for w in WIDTHS for e in ECS},
                  key=lambda j: -(j[0] + j[1]))
    with Pool(4) as p:
        res = p.map(one, jobs)
    json.dump(res, open("exp_compare.json", "w"), indent=1)
