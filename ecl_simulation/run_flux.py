"""Steady-state ROS transport / fate analysis, w=2um gap=2um device, E_gen=-0.8 V.
Budgets are per unit strip length [mol m^-1 s^-1] for the symmetry unit cell (half gen + half col)."""
import numpy as np
from multiprocessing import Pool
import ecl_gc_simulation as m
from ecl_gc_2d import Cell2D

EG = -0.8
POTS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
GRID = dict(nxs=(8, 18, 8), nz=30)
iO2, iSO, iHP, iLH, iRad, iD = m.iO2, m.iSO, m.iHP, m.iLH, m.iRad, m.iD


def budget(cell, c, Ec):
    p = cell.p
    A = cell.hx[:, None] * cell.hz[None, :]
    cc = np.maximum(c, 0)
    fH = 1 / (1 + 10 ** (p["pH"] - p["pKa_HO2"]))
    kd = (p["k_HO2_O2m"] * fH * (1 - fH) + p["k_HO2_HO2"] * fH ** 2) * 1e-3
    r_d = np.sum(kd * cc[iSO] ** 2 * A)
    r_e = np.sum(p["k_ecl"] * 1e-3 * cc[iRad] * cc[iSO] * A)
    r_h = np.sum(p["k_daq_hp"] * 1e-3 * cc[iD] * cc[iHP] * A)
    r_o = np.sum(p["k_rad_o2"] * 1e-3 * cc[iRad] * cc[iO2] * A)
    sg, ng, sc, nc = cell.fluxes(c, Ec)
    wg, wc = cell.hx[cell.gen], cell.hx[cell.col]
    j1 = np.sum((sg[iSO] + ng) / 2 * wg)          # O2 + e -> O2.-
    j2 = np.sum((ng - sg[iSO]) / 2 * wg)          # O2.- -> H2O2 at generator
    sc_ox = -np.sum(sc[iSO] * wc)                 # O2.- -> O2 at collector
    hp_ox = -np.sum(sc[iHP] * wc)                 # H2O2 oxidised at collector
    top = lambda k: np.sum(cell.D[k, 0, 0] * cc[k, :, -1] / (cell.hz[-1] / 2) * cell.hx)
    return dict(E=Ec, gen_O2red=j1, gen_overred=j2, rad_o2=r_o, dismut=r_d, ecl_rad=r_e, ecl_hp=r_h,
                col_SOox=sc_ox, col_HPox=hp_ox, SO_to_bulk=top(iSO), HP_to_bulk=top(iHP),
                photons=p["phi"] * (r_e + r_h), L_ox_col=-np.sum(sc[iLH] * wc))


def one(Ec):
    cell = Cell2D(m.PARAMS, w=2e-6, g=2e-6, Eg=EG, **GRID)
    sol = cell.integrate(Ec, (0, 8), cell.y_bulk(), np.array([0, 7.0, 8.0]))
    y = sol.y[:, -1]
    res = np.abs(cell.rhs(Ec)(8, y)).max() / max(np.abs(y).max(), 1e-12)
    c = y.reshape(m.nS, cell.nx, cell.nz)
    b = budget(cell, c, Ec)
    b["drift"] = np.abs(sol.y[:, -1] - sol.y[:, -2]).max() / np.abs(y).max()
    np.savez(f"flux_state_{Ec}.npz", c=c, xc=cell.xc, zc=cell.zc, xf=cell.xf, zf=cell.zf)
    print("done", Ec, "drift %.1e" % b["drift"], flush=True)
    return b


if __name__ == "__main__":
    with Pool(4) as pool:
        out = pool.map(one, POTS)
    import json
    json.dump(out, open("flux_budget.json", "w"), indent=1)
