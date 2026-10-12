"""
Time-dependent 2D-axisymmetric reaction-diffusion simulation of ECL from
luminol/H2O2 at a gold microtube electrode (MTE), following the COMSOL model in
the Supporting Information (Section 9, eqs S1-S22, Table S2) of

    Zhou, Hu, Guo, Su, Fundamental Research 2, 682 (2022).

Geometry (Fig. S12): tube interior r < r_i for 0 < z < L is electrolyte, the
electrode is the wall r = r_i (0 < z < L); r_i < r < r_o, z < L is insulating
membrane; above z = L the electrolyte is open and the outer/top boundaries are
held at bulk concentrations.  A 1 s potential pulse is simulated; the ECL
image is the photon count  h_nu = int(rate dt)  accumulated over the pulse.

Low pH  (Route I):  HO2- -> O2.- (k1) ; LH- -> LH. (k2) at the wall
                    LH. + LH. -> LH2 + L (k3) ; L + HO2- -> LO2^2- (k4) -> hv (k5)
High pH (Route II): HO2- -> O2.- (k1), O2.- -> O2 (k6), LH- -> L.- (k2) at the wall
                    L.- + O2.- -> LO2^2- (k7) -> hv (k5)

Assumptions beyond the SI text (flagged in README): H2O2 pKa = 11.6 for
[HO2-]; luminol fully present as LH-; L diffuses (S10 prints no Laplacian);
LO2^2- (k5 = 2e5 1/s) treated as quasi-steady so photon rate = k4[L][HO2-]
(Route I) or k7[L.-][O2.-] (Route II); the wall flux uses the concentration
at half a cell from the wall corrected for the diffusion resistance of that
half cell.
Units: um, s, uM.
"""
import numpy as np
import scipy.sparse as sp
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Table S2 -------------------------------------------------------------
RO, RI, LT = 10.25, 9.05, 10.7            # um
DL, DH = 6.6e-6 * 1e8, 1e-5 * 1e8         # cm2/s -> um2/s
K1, K2, K6 = 1e-3 * 1e4, 1.2e-3 * 1e4, 1.0 * 1e4   # cm/s -> um/s
M = 1e-6                                   # M-1 s-1 -> uM-1 s-1
K3, K4, K7 = 1.8e9 * M, 5e7 * M, 2.3e8 * M
C_LH = 1000.0                              # uM luminol (as LH-)
PKA_H2O2 = 11.6                            # assumed
ZTOP, T_PULSE = 35.0, 1.0                  # um above tube mouth, s

def hmax(ph, h2o2_mM, pka=None):
    pka = PKA_H2O2 if pka is None else pka
    return h2o2_mM * 1e3 / (1 + 10 ** (pka - ph))     # uM HO2-

# ---- grid -------------------------------------------------------------------
def graded(n_ref, d0, g, total, from_end, coarse=0.9):
    w, x = [], 0.0
    d = d0
    while x + d < total * 0.999 and len(w) < n_ref:
        w.append(d); x += d; d *= g
    rest = total - x
    n_rest = max(1, int(np.ceil(rest / coarse)))
    w += [rest / n_rest] * n_rest
    return np.array(w[::-1] if from_end else w)

def build(res=1.0):
    """res scales the resolution: wall cell 0.04/res um, growth 1.18**(1/res),
    z spacing 0.5/res um, so res=2 roughly halves every cell size."""
    wr = graded(int(60 * res * 2), 0.04 / res, 1.18 ** (1 / res), RI, True, 0.9 / res)
    r_e = np.r_[0, np.cumsum(wr)]
    r_e = np.r_[r_e, np.linspace(RI, RO, 5 + int(round(4 * (res - 1))))[1:]]
    z_in = np.linspace(0, LT, int(round(21 * res)) + 1)
    z_out = LT + np.cumsum(np.r_[graded(int(40 * res * 2), 0.5 / res, 1.2 ** (1 / res), ZTOP, False, 0.9 / res)])
    z_e = np.r_[z_in, z_out]
    rc, zc = 0.5 * (r_e[1:] + r_e[:-1]), 0.5 * (z_e[1:] + z_e[:-1])
    dr, dz = np.diff(r_e), np.diff(z_e)
    return r_e, z_e, rc, zc, dr, dz

def operators(r_e, z_e, rc, zc, dr, dz):
    nr, nz = rc.size, zc.size
    idx = lambda i, j: i * nz + j
    solid = np.zeros((nr, nz), bool)
    for i in range(nr):
        for j in range(nz):
            solid[i, j] = rc[i] > RI and zc[j] < LT
    vol = np.pi * (r_e[1:]**2 - r_e[:-1]**2)[:, None] * dz[None, :]
    rows, cols, vals = [], [], []
    bvec = np.zeros(nr * nz)                 # Dirichlet coupling (bulk)
    diag = np.zeros(nr * nz)
    def link(a, b, area, dist):
        g = area / dist
        for p, q in ((a, b), (b, a)):
            rows.append(p[0]); cols.append(q[0]); vals.append(g / vol_f[p[1]])
            diag[p[0]] -= g / vol_f[p[1]]
    vol_f = vol.ravel()
    for i in range(nr):
        for j in range(nz):
            if solid[i, j]:
                continue
            a = (idx(i, j), idx(i, j))
            if i + 1 < nr and not solid[i + 1, j]:
                area = 2 * np.pi * r_e[i + 1] * dz[j]
                link((idx(i, j), idx(i, j)), (idx(i + 1, j), idx(i + 1, j)), area, rc[i + 1] - rc[i])
            if j + 1 < nz and not solid[i, j + 1]:
                area = np.pi * (r_e[i + 1]**2 - r_e[i]**2)
                link((idx(i, j), idx(i, j)), (idx(i, j + 1), idx(i, j + 1)), area, zc[j + 1] - zc[j])
    # fix duplicated link() accumulation: each pair processed once above (i+1, j+1 only)
    for i in range(nr):
        for j in range(nz):
            if solid[i, j]:
                continue
            k = idx(i, j)
            if j == nz - 1:                               # top: bulk
                area = np.pi * (r_e[i + 1]**2 - r_e[i]**2); g = area / (0.5 * dz[j])
                diag[k] -= g / vol[i, j]; bvec[k] += g / vol[i, j]
            if i == nr - 1 and zc[j] >= LT:               # side above mouth: bulk
                area = 2 * np.pi * r_e[i + 1] * dz[j]; g = area / (0.5 * dr[i])
                diag[k] -= g / vol[i, j]; bvec[k] += g / vol[i, j]
    Lap = sp.csr_matrix((vals, (rows, cols)), shape=(nr * nz, nr * nz)) + sp.diags(diag)
    # electrode: last interior column (r < RI) next to solid, z < LT
    iw = np.where(rc < RI)[0].max()
    wall = np.zeros(nr * nz); keff_fac = np.zeros(nr * nz)
    for j in range(nz):
        if zc[j] < LT:
            k = idx(iw, j)
            wall[k] = 2 * RI * dz[j] * np.pi / vol[iw, j]       # (area/volume)
            keff_fac[k] = 0.5 * (RI - rc[iw])                  # half-cell distance
    return Lap.tocsr(), bvec, wall, keff_fac, solid.ravel(), vol.ravel(), iw

def simulate(ph, h2o2_mM, route, grid=None, t_end=T_PULSE, n_out=1, pka=None, dL_L=1.0, ho_wall=None):
    """pka: H2O2 pKa (default PKA_H2O2); dL_L: Route-I L diffusion coefficient as a fraction of D_L (0 = no L diffusion, as eq. S10 is printed).
    ho_wall: None = SI as printed (HO2- consumed at the wall, eq. S7/S17); 'noflux' = no wall consumption of HO2-;
    'dirichlet' = [HO2-] pinned at its bulk value in the wall cell."""
    r_e, z_e, rc, zc, dr, dz = grid or build()
    Lap, bvec, wall, hres, solid, vol, iw = operators(r_e, z_e, rc, zc, dr, dz)
    n = rc.size * zc.size
    fl = ~solid
    hb = hmax(ph, h2o2_mM, pka)
    kw = lambda k, D: np.where(wall > 0, k / (1 + k * hres / D), 0.0)   # effective wall k
    k1w, k2w, k6w = kw(K1, DH), kw(K2, DL), kw(K6, DH)
    if route == 1:      # LH-, LH., L, HO2-, hv
        names = ["LH-", "LH.", "L", "HO2-", "hv"]
        bulk = [C_LH, 0, 0, hb, 0]
    else:               # LH-, L.-, HO2-, O2.-, hv
        names = ["LH-", "L.-", "HO2-", "O2.-", "hv"]
        bulk = [C_LH, 0, hb, 0, 0]
    ns = 5
    y0 = np.zeros(ns * n)
    for s in range(ns):
        y0[s*n:(s+1)*n] = np.where(fl, bulk[s], 0.0)
    D = [DL, DL, DL * dL_L, DH, 0] if route == 1 else [DL, DL, DH, DH, 0]

    def rhs(t, y):
        Y = y.reshape(ns, n); out = np.zeros_like(Y)
        for s in range(4):
            out[s] = D[s] * (Lap @ Y[s] + bvec * bulk[s])
        if route == 1:
            lhm, lh, L, ho, _ = Y
            out[0] -= k2w * wall * lhm
            out[1] += k2w * wall * lhm - K3 * lh * lh
            out[2] += K3 * lh * lh - K4 * L * ho
            out[3] += -(0 if ho_wall else 1) * k1w * wall * ho - K4 * L * ho
            if ho_wall == 'dirichlet': out[3] += 1e7 * (wall > 0) * (hb - ho)
            out[4] = K4 * L * ho
        else:
            lhm, Lr, ho, o2, _ = Y
            rate = K7 * Lr * o2
            out[0] -= k2w * wall * lhm
            out[1] += k2w * wall * lhm - rate
            out[2] += -(0 if ho_wall else 1) * k1w * wall * ho
            if ho_wall == 'dirichlet': out[2] += 1e7 * (wall > 0) * (hb - ho)
            out[3] += k1w * wall * ho - k6w * wall * o2 - rate
            out[4] = rate
        out[:, solid] = 0
        return out.ravel()

    # sparsity: local coupling between species + Laplacian stencil
    S = sp.kron(sp.csr_matrix(np.ones((ns, ns))), (abs(Lap) + sp.eye(n)) > 0, format="csr")
    sol = solve_ivp(rhs, (0, t_end), y0, method="BDF", jac_sparsity=S,
                    rtol=1e-5, atol=1e-9, t_eval=[t_end])
    Y = sol.y[:, -1].reshape(ns, n)
    return dict(hv=Y[4].reshape(rc.size, zc.size), Y=Y, names=names, grid=(r_e, z_e, rc, zc, dr, dz),
                iw=iw, ok=sol.success)

def top_view(res, mode="mid"):
    """Radial ECL profile inside the tube: mid-height (default, away from the
    open mouth where reagents are replenished) or averaged over tube depth."""
    r_e, z_e, rc, zc, dr, dz, = res["grid"]
    iw = res["iw"]; zin = zc < LT
    if mode == "mid":
        return rc[:iw + 1], res["hv"][:iw + 1, np.argmin(abs(zc - LT / 2))]
    prof = (res["hv"][:iw + 1][:, zin] * dz[zin][None, :]).sum(axis=1) / LT
    return rc[:iw + 1], prof

def tel(r, prof, frac=np.exp(-1)):
    p = prof / prof.max(); below = np.where(p < frac)[0]
    return RI - r[below.max()] if below.size else RI

if __name__ == "__main__":
    grid = build()
    runs = {}
    for ph, route, lab in ((9.0, 1, "low pH 9 (Route I)"), (12.0, 2, "high pH 12 (Route II)")):
        for c in (0.1, 1.0, 10.0):
            res = simulate(ph, c, route, grid)
            r, p = top_view(res); runs[(ph, c)] = (r, p, res)
            ra, pa = top_view(res, "avg")
            print(f"{lab:24s} H2O2={c:5.1f} mM  ok={res['ok']}  TEL(1/e) mid-height={tel(r,p):5.2f} um, "
                  f"depth-avg={tel(ra,pa):5.2f} um  centre/wall={p[0]/p.max():.3f}  max hv={res['hv'].max():.3g}")
    fig, ax = plt.subplots(2, 3, figsize=(13, 8))
    for k, (ph, ttl) in enumerate(((9.0, "Low pH (9), 10 mM H2O2"), (12.0, "High pH (12), 10 mM H2O2"))):
        r, p, res = runs[(ph, 10.0)]
        g = np.linspace(-RI, RI, 301); X, Y = np.meshgrid(g, g)
        Z = np.where(np.hypot(X, Y) <= RI, np.interp(np.hypot(X, Y), r, p / p.max()), np.nan)
        m = ax[0, k].pcolormesh(X, Y, Z, shading="auto", cmap="viridis"); ax[0, k].set_aspect("equal")
        ax[0, k].set_title(ttl + " (top view)"); plt.colorbar(m, ax=ax[0, k], label="norm. ECL")
    for ph, ls in ((9.0, "-"), (12.0, "--")):
        r, p, _ = runs[(ph, 10.0)]
        ax[0, 2].plot(np.r_[-r[::-1], r], np.r_[p[::-1], p] / p.max(), ls, label=f"pH {ph:g}")
    ax[0, 2].set_title("Radial profiles (cf. Fig. 5c)"); ax[0, 2].legend()
    for k, ph in enumerate((9.0, 12.0)):
        for c in (0.1, 1.0, 10.0):
            r, p, _ = runs[(ph, c)]
            ax[1, k].plot(np.r_[-r[::-1], r], np.r_[p[::-1], p] / p.max(), label=f"{c:g} mM H2O2")
        ax[1, k].set_title(f"pH {ph:g}: effect of [H2O2] (cf. Fig. 6{'g' if ph==9 else 'h'})"); ax[1, k].legend()
    for a in ax.ravel()[2:]: a.set_xlabel("Position / um")
    ax[1, 2].axis("off")
    ax[1, 2].text(0, 0.5, "Model: SI Section 9 / Table S2\n2D axisymmetric, 1 s pulse\nAssumed: pKa(H2O2)=11.6, L diffuses,\nquasi-steady LO2(2-)", va="center")
    plt.tight_layout(); plt.savefig("ecl_simulation.png", dpi=130); print("saved")
