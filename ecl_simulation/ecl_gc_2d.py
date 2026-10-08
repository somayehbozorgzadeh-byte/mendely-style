"""
2-D coplanar generator/collector (interdigitated / entwined-spiral) ECL model.

Geometry (spiral treated as a long array of straight strips; curvature/end effects neglected):
   period = gen(w) | gap(g) | col(w) | gap(g)
Symmetry unit cell: x in [0, L], L = w/2 + g + w/2, generator centre at x=0,
collector centre at x=L (mirror/insulating walls at both ends).
Electrodes sit on the plane z=0; z=H is bulk solution (fixed concentrations).
Chemistry and constants are those of ecl_gc_simulation.py.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import coo_matrix
import ecl_gc_simulation as m

nS = m.nS


def _cluster(a, b, n, mode):
    s = np.linspace(0, 1, n + 1)
    if mode == "right":      # fine near b
        u = 1 - (1 - s) ** 1.8
        u = 1 - (1 - s) ** 2
    elif mode == "left":     # fine near a
        u = s ** 2
    else:                    # fine near both ends
        u = 0.5 + 0.5 * np.tanh(2.0 * (2 * s - 1)) / np.tanh(2.0)
    return a + (b - a) * u


def make_grid(w, g, H=40e-6, nxs=(10, 24, 10), nz=36, dz0=20e-9):
    x1, x2, L = w / 2, w / 2 + g, w + g
    xf = np.concatenate([_cluster(0, x1, nxs[0], "right")[:-1],
                         _cluster(x1, x2, nxs[1], "both")[:-1],
                         _cluster(x2, L, nxs[2], "left")])
    lo, hi = 1.0001, 3.0                      # geometric ratio so that sum = H
    for _ in range(80):
        r = 0.5 * (lo + hi)
        tot = dz0 * (r ** nz - 1) / (r - 1)
        lo, hi = (r, hi) if tot < H else (lo, r)
    zf = np.concatenate([[0], np.cumsum(dz0 * r ** np.arange(nz))])
    return xf, zf


class Cell2D:
    def __init__(self, p, w=2e-6, g=2e-6, Eg=-0.8, H=40e-6, **gridkw):
        self.p, self.Eg, self.w, self.g = dict(p), Eg, w, g
        self.xf, self.zf = make_grid(w, g, H, **gridkw)
        self.xc, self.zc = 0.5 * (self.xf[1:] + self.xf[:-1]), 0.5 * (self.zf[1:] + self.zf[:-1])
        self.hx, self.hz = np.diff(self.xf), np.diff(self.zf)
        self.nx, self.nz = len(self.hx), len(self.hz)
        self.L = self.xf[-1]
        self.gen = self.xc < w / 2
        self.col = self.xc > w / 2 + g
        self.bulk = m.bulk_conc(self.p)[:, None, None]
        self.D = self.p["D"][:, None, None]
        self.dxc, self.dzc = np.diff(self.xc), np.diff(self.zc)
        self.sparsity = self._sparsity()

    def _sparsity(self):
        nx, nz = self.nx, self.nz
        idx = lambda a, i, j: (a * nx + i) * nz + j
        rows, cols = [], []
        I, J = np.meshgrid(np.arange(nx), np.arange(nz), indexing="ij")
        for a in range(nS):
            for b in range(nS):
                rows.append(idx(a, I, J).ravel()); cols.append(idx(b, I, J).ravel())
            for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                ok = (I + di >= 0) & (I + di < nx) & (J + dj >= 0) & (J + dj < nz)
                rows.append(idx(a, I[ok], J[ok])); cols.append(idx(a, I[ok] + di, J[ok] + dj))
        r, c = np.concatenate(rows), np.concatenate(cols)
        return coo_matrix((np.ones(len(r), dtype=int), (r, c)), shape=(nS * nx * nz,) * 2).tocsr()

    def fluxes(self, c, Ec):
        """net production [mol/m^2/s] of every species at electrode cells, and e- flux"""
        sg, ng = m.electrode_flux(c[:, self.gen, 0], self.hz[0], self.p, self.Eg, Ec, "gen")
        sc, nc = m.electrode_flux(c[:, self.col, 0], self.hz[0], self.p, self.Eg, Ec, "col")
        return sg, ng, sc, nc

    def rhs(self, Ec_fun):
        D, hx, hz, nx, nz = self.D, self.hx, self.hz, self.nx, self.nz

        def f(t, y):
            c = y.reshape(nS, nx, nz)
            dc = np.zeros_like(c)
            fx = D * (c[:, :-1] - c[:, 1:]) / self.dxc[None, :, None]
            dc[:, :-1] -= fx / hx[None, :-1, None]
            dc[:, 1:] += fx / hx[None, 1:, None]
            fz = D * (c[:, :, :-1] - c[:, :, 1:]) / self.dzc[None, None, :]
            dc[:, :, :-1] -= fz / hz[None, None, :-1]
            dc[:, :, 1:] += fz / hz[None, None, 1:]
            dc[:, :, -1] += D[:, :, 0] * (self.bulk[:, :, 0] - c[:, :, -1]) / (hz[-1] / 2) / hz[-1]
            dc += m.solution_rates(c, self.p)[0]
            Ec = Ec_fun(t) if callable(Ec_fun) else Ec_fun
            sg, _, sc, _ = self.fluxes(c, Ec)
            dc[:, self.gen, 0] += sg / hz[0]
            dc[:, self.col, 0] += sc / hz[0]
            return dc.ravel()
        return f

    def y_bulk(self):
        return np.broadcast_to(self.bulk, (nS, self.nx, self.nz)).ravel().copy()

    def integrate(self, Ec_fun, t_span, y0, t_eval):
        sol = solve_ivp(self.rhs(Ec_fun), t_span, y0, method="BDF", jac_sparsity=self.sparsity,
                        t_eval=t_eval, rtol=1e-5, atol=1e-11)
        if not sol.success:
            raise RuntimeError(sol.message)
        return sol

    def observables(self, c, Ec):
        """ECL per footprint area [photons m^-2 s^-1], electrode current densities [A/m^2]."""
        ph = m.solution_rates(c, self.p)[1]
        ecl = np.sum(ph * self.hx[:, None] * self.hz[None, :]) / self.L * m.NA
        _, ng, _, nc = self.fluxes(c, Ec)
        wg, wc = self.hx[self.gen], self.hx[self.col]
        i_gen = m.F * np.sum(ng * wg) / wg.sum()          # cathodic +
        i_col = -m.F * np.sum(nc * wc) / wc.sum()         # anodic +
        return ecl, i_gen, i_col, ph
