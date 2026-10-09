"""ECL vs electrode width (generator = collector width w, gap 2 um), steady state, E_gen=-0.8 V."""
import json
import numpy as np
from multiprocessing import Pool
import ecl_gc_simulation as m
from ecl_gc_2d import Cell2D

WS = [1e-6, 2e-6, 4e-6, 8e-6, 16e-6]
ECS = [0.4, 0.7]


def one(args):
    w, Ec = args
    L = w + 2e-6
    cell = Cell2D(m.PARAMS, w=w, g=2e-6, Eg=-0.8, H=max(40e-6, 4 * L), nxs=(8, 18, 8), nz=30)
    sol = cell.integrate(Ec, (0, 8), cell.y_bulk(), np.array([0, 7.0, 8.0]))
    c = sol.y[:, -1].reshape(m.nS, cell.nx, cell.nz)
    ecl_foot, ig, ic, ph = cell.observables(c, Ec)
    A = cell.hx[:, None] * cell.hz[None, :]
    em_x = (ph * cell.hz[None, :]).sum(1) * m.NA            # photons / m^2 /s per x-slice (z-integrated)
    colmask = cell.col
    P_total = ecl_foot * cell.L                              # photons per m strip length per s, per unit cell
    P_col = np.sum(em_x[colmask] * cell.hx[colmask])
    out = dict(w=w, Ec=Ec, ecl_foot=ecl_foot, P_unit_cell=P_total, P_above_col=P_col,
               P_per_collector_area=P_total / (w / 2), i_col=ic, i_gen=ig, L=cell.L,
               drift=float(np.abs(sol.y[:, -1] - sol.y[:, -2]).max() / np.abs(sol.y[:, -1]).max()))
    print("done", w, Ec, flush=True)
    return out


if __name__ == "__main__":
    jobs = [(w, e) for e in ECS for w in WS]
    with Pool(4) as p:
        res = p.map(one, jobs)
    json.dump(res, open("width_scan.json", "w"), indent=1)
    for e in ECS:
        print(f"\nE_col = {e} V")
        print("  w/um  ECL per footprint  ECL per collector area  ECL per unit length of collector edge pair   drift")
        for r in [r for r in res if r["Ec"] == e]:
            print("  %4.0f  %.3e   %.3e   %.3e   %.0e" % (r["w"] * 1e6, r["ecl_foot"], r["P_per_collector_area"], r["P_unit_cell"], r["drift"]))
