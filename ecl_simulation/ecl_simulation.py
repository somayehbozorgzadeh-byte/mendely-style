"""
Reaction-diffusion simulation of the luminol/H2O2 ECL layer inside a gold
microtube electrode (Zhou, Hu, Guo, Su, Fundamental Research 2, 682 (2022)).

Top view = 2D disc, radius R, uniform along the tube axis, so the problem is
radially symmetric: 1D cylindrical reaction-diffusion, species injected at the
wall (r = R), no-flux at the centre. Steady state, solved with solve_bvp.

Route I  (low pH):  LH. + LH. -> L (+LH2)      k1 = 1.8e9 M-1 s-1  (paper)
                    L + HO2-  -> LO2^2- -> hv   k2 = 5.0e7 M-1 s-1  (paper)
                    emission rate = k2 [HO2-] [L]
Route II (high pH): L.- and O2.- are short-lived (tau ~ 40 ms, paper) and react
                    at the wall: L.- + O2.- -> LO2^2- -> hv
                    emission rate = k3 [L.-][O2.-]   (k3 ASSUMED, only shapes
                    the normalised profile, not its width)
"""
import numpy as np
from scipy.integrate import solve_bvp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = 5e-6 * 1e-4 * 1e12          # 5e-6 cm2/s -> um2/s  (paper)
R = 9.05                        # um, inner radius (ID = 18.1 um, paper)
K1, K2 = 1.8e9, 5.0e7           # M-1 s-1 (paper)
K3 = 1e9                        # M-1 s-1 (assumed)
TAU = 0.040                     # s, radical lifetime (paper)
PKA_H2O2 = 11.6                 # H2O2 <-> HO2- (literature)
J = 1e5                         # uM*um/s wall flux ~ D*c_bulk/delta for 1 mM luminol (estimate)

def ho2(ph, h2o2_M):
    return h2o2_M / (1 + 10 ** (PKA_H2O2 - ph))

def solve(rhs, nvar, bc_wall):
    r = np.linspace(1e-3, R, 400)
    def f(r, y):  # y = [c1, c1', c2, c2', ...]
        out = []
        for i in range(nvar):
            out += [y[2*i+1], rhs(i, y[0::2], y)[0] - y[2*i+1] / r]
        return np.array(out)
    def bc(ya, yb):
        res = []
        for i in range(nvar):
            res += [ya[2*i+1], yb[2*i+1] - bc_wall[i]]
        return np.array(res)
    y0 = np.ones((2*nvar, r.size)) * 1e-3
    return solve_bvp(f, bc, r, y0, tol=1e-6, max_nodes=200000)

def route1(ph, h2o2_mM):
    """Concentration unit: M (flux J in M*um/s per um^2 scale, arbitrary)."""
    kq = K2 * ho2(ph, h2o2_mM * 1e-3)            # s-1, L consumption
    def rhs(i, c, y):
        lh, l = np.maximum(c[0], 0), np.maximum(c[1], 0)
        if i == 0:   return [(2 * K1 * lh**2) / D * 1e0, 0]   # LH. loss
        return [(kq * l - K1 * lh**2) / D, 0]                  # L balance
    # scale: use micromolar to keep numbers sane
    return rhs, kq

def run_route1(ph, h2o2_mM):
    kq = K2 * ho2(ph, h2o2_mM * 1e-3)            # s-1
    k1 = K1 * 1e-6                               # per uM per s
    def rhs(i, c, y):
        lh, l = np.maximum(c[0], 0), np.maximum(c[1], 0)
        if i == 0:
            return [2 * k1 * lh**2 / D]
        return [(kq * l - k1 * lh**2) / D]
    sol = solve(rhs, 2, [J / D, 0.0])           # LH. injected, L no-flux
    L = np.maximum(sol.sol(sol.x)[2], 0)
    return sol.x, kq * L, kq

def run_route2():
    kd = 1 / TAU
    def rhs(i, c, y):
        return [kd * np.maximum(c[i], 0) / D]
    sol = solve(rhs, 2, [J / D, J / D])
    a, b = np.maximum(sol.sol(sol.x)[0], 0), np.maximum(sol.sol(sol.x)[2], 0)
    return sol.x, a * b

def tel(x, I, frac=np.exp(-1)):
    """distance from the wall at which normalised intensity falls to 1/e."""
    I = I / I.max()
    below = np.where(I < frac)[0]
    return R - x[below.max()] if below.size else R

def disc(x, I, n=301):
    g = np.linspace(-R, R, n); X, Y = np.meshgrid(g, g)
    return X, Y, np.where(X**2 + Y**2 <= R**2, np.interp(np.hypot(X, Y), x, I), np.nan)

if __name__ == "__main__":
    out = {}
    # Fig 5 analogue: low pH (Route I, pH 9, 10 mM H2O2) vs high pH (Route II)
    x1, I1, kq = run_route1(9.0, 10.0)
    x2, I2 = run_route2()
    print(f"Route I  pH 9, 10 mM H2O2: kq={kq:.0f} s-1, TEL(1/e)={tel(x1,I1):.2f} um")
    print(f"Route II               : TEL(1/e)={tel(x2,I2):.2f} um, "
          f"diffusion length (D*tau)^0.5={np.sqrt(D*TAU):.1f} um")

    fig, ax = plt.subplots(2, 3, figsize=(13, 8))
    for a, (x, I, t) in zip(ax[0, :2], [(x1, I1, "Route I (low pH)"), (x2, I2, "Route II (high pH)")]):
        X, Y, Z = disc(x, I / I.max()); m = a.pcolormesh(X, Y, Z, shading="auto", cmap="viridis")
        a.set_aspect("equal"); a.set_title(t); plt.colorbar(m, ax=a, label="norm. ECL")
    for x, I, lab, mk in [(x1, I1, "Low pH (Route I)", "k-o"), (x2, I2, "High pH (Route II)", "-o")]:
        r_all = np.r_[-x[::-1], x]; I_all = np.r_[I[::-1], I] / I.max()
        ax[0, 2].plot(r_all, I_all, label=lab)
    ax[0, 2].set_xlabel("Position / um"); ax[0, 2].set_ylabel("Normalised intensity"); ax[0, 2].legend()
    ax[0, 2].set_title("Radial profiles (cf. paper Fig. 5c)")

    # Fig 6g analogue: Route I, effect of [H2O2]
    for c in (0.1, 1, 10):
        x, I, _ = run_route1(9.0, c); out[c] = tel(x, I)
        ax[1, 0].plot(np.r_[-x[::-1], x], np.r_[I[::-1], I] / I.max(), label=f"{c} mM H2O2")
    ax[1, 0].set_title("Route I, pH 9: effect of [H2O2] (cf. Fig. 6g)")
    ax[1, 0].set_xlabel("Position / um"); ax[1, 0].set_ylabel("Normalised intensity"); ax[1, 0].legend()
    print("Route I TEL vs [H2O2] (mM):", {k: round(v, 2) for k, v in out.items()})

    # TEL vs pH (Route I only), 1 mM luminol / 10 mM H2O2
    phs = np.arange(7, 12.1, 0.5)
    tels = [tel(*run_route1(p, 10.0)[:2]) for p in phs]
    ax[1, 1].plot(phs, tels, "o-"); ax[1, 1].set_xlabel("pH"); ax[1, 1].set_ylabel("TEL (1/e)  / um")
    ax[1, 1].set_title("Route I: TEL vs pH (HO2- consumption of L)")
    ax[1, 2].axis("off")
    ax[1, 2].text(0, 0.5, "Parameters from paper: D=5e-6 cm2/s,\nk1=1.8e9, k2=5e7 M-1s-1, tau(O2.-)=40 ms,\nID=18.1 um.\nAssumed: pKa(H2O2)=11.6, k3=1e9,\nL.- lifetime = O2.- lifetime.", fontsize=10, va="center")
    plt.tight_layout(); plt.savefig("ecl_simulation.png", dpi=130)
    print("saved ecl_simulation.png")
