"""
Full ECL chemistry (ecl_gc_simulation.py) on a FINITE generator-collector array:
flat interdigitated (planar x-z cross-section) or entwined spiral (axisymmetric concentric rings).
Grid and geometry factors are taken from redox_gc_array.Array; integration with stiff BDF + sparse Jacobian.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import coo_matrix
import ecl_gc_simulation as m
from redox_gc_array import Array

nS = m.nS


class ECLArray:
    def __init__(self, geometry, p, w=2e-6, g=2e-6, n_pairs=25, Eg=-0.7, far=300e-6, ncell=4, finger_len=200e-6):
        a = Array(geometry, w=w, g=g, n_pairs=n_pairs, finger_len=finger_len, far=far, ncell=ncell)
        self.geo, self.p, self.Eg = geometry, dict(p), Eg
        for k in ("xf", "zf", "xc", "zc", "hx", "hz", "nx", "nz", "gen", "col", "Az", "Ax_face", "Vol", "scale", "segs"):
            setattr(self, k, getattr(a, k))
        self.D = self.p["D"][:, None, None]
        self.bulk = m.bulk_conc(self.p)[:, None, None]
        self.dxc, self.dzc = np.diff(self.xc), np.diff(self.zc)
        self.Gx = self.Ax_face[1:-1, :][None] / self.dxc[None, :, None]        # geometric conductances (without D)
        self.Gz = self.Az[None, :, None] / self.dzc[None, None, :]
        self.Gtop = self.Az / (self.hz[-1] / 2)
        self.Gright = self.Ax_face[-1, :] / (self.hx[-1] / 2)
        self.Gleft = self.Ax_face[0, :] / (self.hx[0] / 2) if geometry == "ida" else None
        self.sparsity = self._sparsity()

    def _sparsity(self):
        nx, nz = self.nx, self.nz
        idx = lambda s, i, j: (s * nx + i) * nz + j
        I, J = np.meshgrid(np.arange(nx), np.arange(nz), indexing="ij")
        rows, cols = [], []
        for s in range(nS):
            for t in range(nS):
                rows.append(idx(s, I, J).ravel()); cols.append(idx(t, I, J).ravel())
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ok = (I + di >= 0) & (I + di < nx) & (J + dj >= 0) & (J + dj < nz)
                rows.append(idx(s, I[ok], J[ok])); cols.append(idx(s, I[ok] + di, J[ok] + dj))
        r, c = np.concatenate(rows), np.concatenate(cols)
        return coo_matrix((np.ones(len(r), dtype=np.int8), (r, c)), shape=(nS * nx * nz,) * 2).tocsr()

    def fluxes(self, c, Ec):
        sg, ng = m.electrode_flux(c[:, self.gen, 0], self.hz[0], self.p, self.Eg, Ec, "gen")
        if Ec is None:
            sc, nc = np.zeros_like(c[:, self.col, 0]), np.zeros(self.col.sum())
        else:
            sc, nc = m.electrode_flux(c[:, self.col, 0], self.hz[0], self.p, self.Eg, Ec, "col")
        return sg, ng, sc, nc

    def rhs(self, Ec_fun):
        D, nx, nz = self.D, self.nx, self.nz

        def f(t, y):
            c = y.reshape(nS, nx, nz)
            q = np.zeros_like(c)                                   # net inflow, mol/s (per rad or per m)
            fx = D * self.Gx * (c[:, :-1] - c[:, 1:])
            q[:, :-1] -= fx; q[:, 1:] += fx
            fz = D * self.Gz * (c[:, :, :-1] - c[:, :, 1:])
            q[:, :, :-1] -= fz; q[:, :, 1:] += fz
            q[:, :, -1] += D[:, :, 0] * self.Gtop[None] * (self.bulk[:, :, 0] - c[:, :, -1])
            q[:, -1, :] += D[:, 0, :] * self.Gright[None] * (self.bulk[:, 0, :] - c[:, -1, :])
            if self.Gleft is not None:
                q[:, 0, :] += D[:, 0, :] * self.Gleft[None] * (self.bulk[:, 0, :] - c[:, 0, :])
            Ec = Ec_fun(t) if callable(Ec_fun) else Ec_fun
            sg, _, sc, _ = self.fluxes(c, Ec)
            q[:, self.gen, 0] += sg * self.Az[self.gen]
            q[:, self.col, 0] += sc * self.Az[self.col]
            dc = q / self.Vol[None] + m.solution_rates(c, self.p)[0]
            return dc.ravel()
        return f

    def y_bulk(self):
        return np.broadcast_to(self.bulk, (nS, self.nx, self.nz)).ravel().copy()

    def observables(self, c, Ec):
        ph = m.solution_rates(c, self.p)[1]
        photons = np.sum(ph * self.Vol) * self.scale * m.NA                    # photons/s, whole device
        _, ng, _, nc = self.fluxes(c, Ec)
        i_gen = m.F * np.sum(ng * self.Az[self.gen]) * self.scale             # cathodic +, A
        i_col = -m.F * np.sum(nc * self.Az[self.col]) * self.scale if Ec is not None else 0.0   # anodic +, A
        return photons, i_gen, i_col, ph

    @property
    def gen_area(self):
        return float(np.sum(self.Az[self.gen]) * self.scale)

    @property
    def footprint(self):
        x0, x1 = self.segs[0][0], self.segs[-1][1]
        return float(np.pi * x1 ** 2) if self.geo == "spiral" else float((x1 - x0) * self.scale)
