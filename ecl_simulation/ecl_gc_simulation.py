"""
Generator-collector ECL simulation (pH 7.4, aqueous buffer).

Generator  : O2 + e-  -> O2.-            (oxygen reduction, cathodic)
Collector  : luminol (LH-) -> L.- -> L   (luminol oxidation, anodic)
Gap        : L.- + O2.- -> 3-APA* -> 3-APA + hv     (ECL)

Model: 1-D finite-volume reaction-diffusion across a gap of width d between two
parallel electrodes (generator at x=0, collector at x=d), integrated in time with
a stiff solver (BDF). Mass supply from the open sides of the gap (strip width w)
is lumped as first-order exchange with bulk solution, k_lat = D*(pi/w)^2.

All rate/electrode constants are representative literature-order values, NOT fits.
Edit PARAMS to match your electrode material / electrolyte.
Units inside the solver: m, s, mol/m^3 (= mM), V vs Ag/AgCl.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import lil_matrix

F = 96485.0
R = 8.314
T = 298.15
f = F / (R * T)
NA = 6.022e23

SPECIES = ["O2", "O2.-", "H2O2", "LH-", "L.-", "L(DAQ)"]
nS = len(SPECIES)
iO2, iSO, iHP, iLH, iRad, iD = range(nS)

PARAMS = dict(
    pH=7.4,
    gap=2e-6,            # generator-collector spacing d [m]
    width=10e-6,         # electrode strip width w for lateral exchange [m]
    O2_bulk=0.26,        # air-saturated, 25 C [mol/m^3]
    lum_total=0.10,      # total luminol [mol/m^3] (100 uM)
    pKa_lum=6.7,         # LH2 <-> LH- ; LH- is the electroactive/ECL form
    D=np.array([2.0e-9, 1.8e-9, 1.5e-9, 0.6e-9, 0.6e-9, 0.6e-9]),  # [m^2/s]
    # --- solution kinetics [M^-1 s^-1] unless stated ---
    k_HO2_O2m=9.7e7, k_HO2_HO2=8.3e5, pKa_HO2=4.8,   # superoxide dismutation (pH dependent)
    k_ecl=1.0e9,         # L.- + O2.- -> 3-APA*
    k_daq_hp=2.0e2,      # L + H2O2 -> 3-APA* (weak channel)
    k_disp=1.0e8,        # 2 L.- -> L + LH-
    k_rad_o2=5.0e3,      # L.- + O2 -> L + O2.-
    k_daq_dec=5.0,       # L -> dark products [s^-1]
    phi=0.01,            # 3-APA* formation/emission efficiency (photons per ECL event)
    # --- electrode kinetics (E vs Ag/AgCl) ---
    E0_O2=-0.53, k0_O2=1e-5, a_O2=0.5,      # O2/O2.-   (E0' ~ -0.33 V vs NHE)
    E0_SO2=-0.40, k0_SO2=1e-7, a_SO2=0.5,   # O2.- -> H2O2 over-reduction at generator
    E0_LH=0.30, k0_LH=1e-5, a_LH=0.5,       # LH- <-> L.- (+H+)
    E0_Rad=0.25, k0_Rad=1e-4, a_Rad=0.5,    # L.- <-> L
    E0_HP=0.60, k0_HP=1e-7, a_HP=0.5,       # H2O2 oxidation at collector (sluggish)
)


def make_grid(d, n=120):
    """Cell faces clustered towards both electrodes."""
    s = np.linspace(0, 1, n + 1)
    xf = d * 0.5 * (1 - np.cos(np.pi * s)) ** 1.0
    xf = d * (0.5 + 0.5 * np.tanh(2.2 * (2 * s - 1)) / np.tanh(2.2))
    xc = 0.5 * (xf[1:] + xf[:-1])
    return xf, xc


def bulk_conc(p):
    frac = 1.0 / (1.0 + 10 ** (p["pKa_lum"] - p["pH"]))
    b = np.zeros(nS)
    b[iO2] = p["O2_bulk"]
    b[iLH] = p["lum_total"] * frac
    return b


def hl(E, E0, k0, a):
    x = f * (E - E0)
    return k0 * np.exp(-a * x), k0 * np.exp((1 - a) * x)   # reduction, oxidation rate consts


def electrode_flux(c0, h, p, Eg, Ec, which):
    """Net production rate [mol/m^2/s] of each species into the adjacent cell, plus
    electron flux (cathodic negative) for the electrode. c0: concentrations in the
    cell touching the electrode; h: width of that cell."""
    D = p["D"]
    c = np.maximum(c0, 0.0)
    s = np.zeros_like(c)   # works for c0 of shape (nS,) or (nS, k) (vectorised over electrode cells)
    ne = 0.0   # mol e- /m^2/s consumed (+) or released (-): i = F*ne cathodic positive
    if which == "gen":
        E = Eg
        # O2 + e <-> O2.-
        kf, kb = hl(E, p["E0_O2"], p["k0_O2"], p["a_O2"])
        j = (kf * c[iO2] - kb * c[iSO]) / (1 + kf * h / (2 * D[iO2]) + kb * h / (2 * D[iSO]))
        s[iO2] -= j; s[iSO] += j; ne += j
        # O2.- + e (+H+) -> H2O2 (irreversible)
        kf, _ = hl(E, p["E0_SO2"], p["k0_SO2"], p["a_SO2"])
        j = kf * c[iSO] / (1 + kf * h / (2 * D[iSO]))
        s[iSO] -= j; s[iHP] += j; ne += j
    else:
        E = Ec
        # LH- -> L.- + e
        kf, kb = hl(E, p["E0_LH"], p["k0_LH"], p["a_LH"])
        j = (kf * c[iLH] - kb * c[iRad]) / (1 + kf * h / (2 * D[iLH]) + kb * h / (2 * D[iRad]))   # net reduction (L.- -> LH-)
        # note: here "Ox"=L.-, "Red"=LH-; recompute properly
        kf, kb = hl(E, p["E0_LH"], p["k0_LH"], p["a_LH"])
        j = (kf * c[iRad] - kb * c[iLH]) / (1 + kf * h / (2 * D[iRad]) + kb * h / (2 * D[iLH]))
        s[iRad] -= j; s[iLH] += j; ne += j
        # L.- -> L + e
        kf, kb = hl(E, p["E0_Rad"], p["k0_Rad"], p["a_Rad"])
        j = (kf * c[iD] - kb * c[iRad]) / (1 + kf * h / (2 * D[iD]) + kb * h / (2 * D[iRad]))
        s[iD] -= j; s[iRad] += j; ne += j
        # O2.- <-> O2 + e   (re-oxidation of superoxide)
        kf, kb = hl(E, p["E0_O2"], p.get("k0_O2_col", p["k0_O2"]), p["a_O2"])
        j = (kf * c[iO2] - kb * c[iSO]) / (1 + kf * h / (2 * D[iO2]) + kb * h / (2 * D[iSO]))
        s[iO2] -= j; s[iSO] += j; ne += j
        # H2O2 -> O2 + 2H+ + 2e (irreversible, 2 e-)
        _, kb = hl(E, p["E0_HP"], p["k0_HP"], p["a_HP"])
        j = kb * c[iHP] / (1 + kb * h / (2 * D[iHP]))
        s[iHP] -= j; s[iO2] += j; ne -= 2 * j
    return s, ne


def solution_rates(c, p):
    """c: (nS, N) in mol/m^3. Returns dc/dt from homogeneous reactions and the
    ECL photon-generation rate profile [photons-equivalents mol/m^3/s]."""
    c = np.maximum(c, 0.0)
    fH = 1.0 / (1.0 + 10 ** (p["pH"] - p["pKa_HO2"]))     # fraction as HO2.
    kd = (p["k_HO2_O2m"] * fH * (1 - fH) + p["k_HO2_HO2"] * fH ** 2) * 1e-3 * p.get("kd_scale", 1.0)   # per event, m^3/mol/s
    r_d = kd * c[iSO] ** 2                                # 2 O2.- -> H2O2 + O2
    r_e = p["k_ecl"] * 1e-3 * c[iRad] * c[iSO]            # L.- + O2.- -> 3-APA*
    r_h = p["k_daq_hp"] * 1e-3 * c[iD] * c[iHP]           # L + H2O2 -> 3-APA*
    r_p = p["k_disp"] * 1e-3 * c[iRad] ** 2               # 2 L.- -> L + LH-
    r_o = p["k_rad_o2"] * 1e-3 * c[iRad] * c[iO2]         # L.- + O2 -> L + O2.-
    r_l = p["k_daq_dec"] * c[iD]
    r_x = p.get("k_lum_so", 0.0) * 1e-3 * c[iLH] * c[iSO]  # LH- + O2.- -> L.- + H2O2 (homogeneous luminol oxidation; cathodic ECL route)
    dc = np.zeros_like(c)
    dc[iSO] += -2 * r_d - r_e + r_o - r_x
    dc[iO2] += r_d - r_o
    dc[iHP] += r_d - r_h + r_x
    dc[iRad] += -r_e - 2 * r_p - r_o + r_x
    dc[iD] += r_p + r_o - r_h - r_l
    dc[iLH] += r_p - r_x
    photons = p["phi"] * (r_e + r_h)
    return dc, photons


def build(p, Eg, Ec, n=120):
    xf, xc = make_grid(p["gap"], n)
    hcell = np.diff(xf)
    dxc = np.diff(xc)
    D = p["D"][:, None]
    bulk = bulk_conc(p)[:, None]
    klat = p["D"][:, None] * (np.pi / p["width"]) ** 2

    def rhs(t, y):
        c = y.reshape(nS, n)
        flux = D * (c[:, :-1] - c[:, 1:]) / dxc          # +x face flux (right-going)
        dc = np.zeros_like(c)
        dc[:, :-1] -= flux / hcell[:-1]
        dc[:, 1:] += flux / hcell[1:]
        dr, _ = solution_rates(c, p)
        dc += dr
        dc += klat * (bulk - c)
        sg, _ = electrode_flux(c[:, 0], hcell[0], p, Eg, Ec, "gen")
        sc, _ = electrode_flux(c[:, -1], hcell[-1], p, Eg, Ec(t) if callable(Ec) else Ec, "col")
        dc[:, 0] += sg / hcell[0]
        dc[:, -1] += sc / hcell[-1]
        return dc.ravel()

    sp = lil_matrix((nS * n, nS * n), dtype=int)
    for i in range(n):
        for a in range(nS):
            for b in range(nS):
                sp[a * n + i, b * n + i] = 1
            if i > 0:
                sp[a * n + i, a * n + i - 1] = 1
            if i < n - 1:
                sp[a * n + i, a * n + i + 1] = 1
    return rhs, sp.tocsr(), xf, xc, hcell


def run(p=None, Eg=-0.8, Ec=0.7, t_end=10.0, n=120, t_eval=None, y0=None):
    p = dict(PARAMS if p is None else p)
    rhs, sp, xf, xc, hcell = build(p, Eg, Ec, n)
    if y0 is None:
        y0 = np.repeat(bulk_conc(p)[:, None], n, axis=1).ravel()
    if t_eval is None:
        t_eval = np.linspace(0, t_end, 201)
    sol = solve_ivp(rhs, (0, t_end), y0, method="BDF", jac_sparsity=sp, t_eval=t_eval,
                    rtol=1e-6, atol=1e-11)
    if not sol.success:
        raise RuntimeError(sol.message)
    out = dict(t=sol.t, xc=xc, xf=xf, hcell=hcell, p=p, Eg=Eg, Ec=Ec, n=n)
    C = sol.y.reshape(nS, n, -1)
    out["C"] = C
    ecl, ig, ic = [], [], []
    prof = None
    for k in range(C.shape[2]):
        c = C[:, :, k]
        _, ph = solution_rates(c, p)
        ecl.append(np.sum(ph * hcell))                   # mol photons /m^2/s
        _, neg = electrode_flux(c[:, 0], hcell[0], p, Eg, Ec, "gen")
        _, nec = electrode_flux(c[:, -1], hcell[-1], p, Eg, Ec, "col")
        ig.append(F * neg)                               # A/m^2, cathodic +
        ic.append(-F * nec)                              # A/m^2, anodic +
        prof = ph
    out["ecl"] = np.array(ecl) * NA                      # photons m^-2 s^-1
    out["i_gen"] = np.array(ig)                          # cathodic magnitude
    out["i_col"] = np.array(ic)                          # anodic
    out["ecl_profile"] = prof * p["phi"] * 0 + solution_rates(C[:, :, -1], p)[1]
    return out


def steady(p=None, Eg=-0.8, Ec=0.7, t_end=15.0, n=100):
    r = run(p, Eg, Ec, t_end=t_end, n=n, t_eval=np.array([0.0, t_end]))
    return r["ecl"][-1], r["i_gen"][-1], r["i_col"][-1], r
