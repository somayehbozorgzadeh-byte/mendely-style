"""
Generator-collector redox cycling of a single fast couple (ferrocenemethanol, FcMeOH -> FcMeOH+ + e-)
on a FINITE electrode array, with no ECL chemistry.

geometry = "ida"    : planar 2-D cross-section (x, z) through all fingers of an interdigitated array;
                      currents per unit finger length, multiplied by the finger length.
geometry = "spiral" : axisymmetric (r, z) model of an entwined two-arm spiral approximated as concentric
                      alternating generator/collector rings (the standard approximation for a tightly wound spiral).

Finite-volume grid, backward-Euler time stepping; Butler-Volmer kinetics (linear in concentration at fixed E,
so each time step is one sparse linear solve).  Units: m, s, mol/m^3, V vs Ag/AgCl.
"""
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import splu

F, R_, T = 96485.0, 8.314, 298.15
f = F / (R_ * T)

FCMEOH = dict(c_bulk=1.0, D_R=7.8e-10, D_O=7.0e-10, E0=0.20, k0=6e-3, alpha=0.5)   # R = FcMeOH, O = FcMeOH+


def _seg(a, b, n, beta=2.0):
    s = np.linspace(0, 1, n + 1)
    return a + (b - a) * (0.5 + 0.5 * np.tanh(beta * (2 * s - 1)) / np.tanh(beta))


def _grow(a, h0, L, r=1.15):
    out = [a]; h = h0
    while out[-1] < a + L:
        out.append(out[-1] + h); h *= r
    return np.array(out)


class Array:
    def __init__(self, geometry="ida", w=2e-6, g=2e-6, n_pairs=25, r_in=8e-6, finger_len=200e-6,
                 far=500e-6, ncell=6, dz0=20e-9, p=FCMEOH):
        self.geo, self.p, self.w, self.g, self.N, self.Lf = geometry, dict(p), w, g, n_pairs, finger_len
        # electrode layout along x (or r): G g C g G g C ...
        x0 = r_in if geometry == "spiral" else 0.0
        segs = []; x = x0
        for k in range(n_pairs):
            for kind in ("G", "-", "C", "-"):
                L = w if kind in "GC" else g
                if k == n_pairs - 1 and kind == "-" and len(segs) and segs[-1][2] == "C":
                    break
                segs.append((x, x + L, kind)); x += L
        self.segs, self.x_end = segs, x
        xf = [np.array([x0])]
        if geometry == "spiral" and x0 > 0:
            xf = [np.concatenate([np.linspace(0, x0, 6)[:-1], [x0]])]
        elif geometry == "ida":
            xf = [(x0 - _grow(0, 0.3e-6, far))[::-1]]                 # insulating region to the left
        for a, b, _ in segs:
            xf.append(_seg(a, b, ncell)[1:])
        xf.append(_grow(x, 0.3e-6, far)[1:])
        self.xf = np.unique(np.concatenate(xf))
        self.zf = _grow(0.0, dz0, far, r=1.18)
        self.xc = 0.5 * (self.xf[1:] + self.xf[:-1]); self.zc = 0.5 * (self.zf[1:] + self.zf[:-1])
        self.hx, self.hz = np.diff(self.xf), np.diff(self.zf)
        self.nx, self.nz = len(self.hx), len(self.hz)
        kind = np.full(self.nx, "-")
        for a, b, k in segs:
            kind[(self.xc > a) & (self.xc < b)] = k
        self.gen, self.col = kind == "G", kind == "C"
        self._geometry_factors()
        self._assemble()

    def _geometry_factors(self):
        xf, hz = self.xf, self.hz
        if self.geo == "spiral":        # per radian -> multiply by 2*pi for totals
            self.Az = 0.5 * (xf[1:] ** 2 - xf[:-1] ** 2)                 # z-face area of each column
            self.Ax_face = xf[:, None] * hz[None, :]                      # x(r)-face areas
            self.scale = 2 * np.pi
        else:                           # per unit finger length -> multiply by finger length
            self.Az = self.hx.copy()
            self.Ax_face = np.ones_like(xf)[:, None] * hz[None, :]
            self.scale = self.Lf
        self.Vol = self.Az[:, None] * hz[None, :]

    def idx(self, s, i, j):
        return (s * self.nx + i) * self.nz + j

    def _assemble(self):
        nx, nz, p = self.nx, self.nz, self.p
        N = 2 * nx * nz
        rows, cols, vals = [], [], []
        rhs_b = np.zeros(N)
        I, J = np.meshgrid(np.arange(nx), np.arange(nz), indexing="ij")
        for s, D, cb in ((0, p["D_R"], p["c_bulk"]), (1, p["D_O"], 0.0)):
            # x-direction internal faces
            dx = np.diff(self.xc)
            Gx = D * self.Ax_face[1:-1, :] / dx[:, None]                  # conductance (nx-1, nz)
            a = self.idx(s, I[:-1], J[:-1]).ravel(); b = self.idx(s, I[1:], J[1:]).ravel(); g = Gx.ravel()
            rows += [a, b, a, b]; cols += [a, b, b, a]; vals += [g, g, -g, -g]
            dzc = np.diff(self.zc)
            Gz = D * self.Az[:, None] / dzc[None, :]
            a = self.idx(s, I[:, :-1], J[:, :-1]).ravel(); b = self.idx(s, I[:, 1:], J[:, 1:]).ravel(); g = Gz.ravel()
            rows += [a, b, a, b]; cols += [a, b, b, a]; vals += [g, g, -g, -g]
            # far boundaries (Dirichlet bulk): top and the outer x side(s)
            gt = D * self.Az / (self.hz[-1] / 2); k = self.idx(s, np.arange(nx), nz - 1)
            rows.append(k); cols.append(k); vals.append(gt); rhs_b[k] += gt * cb
            gr = D * self.Ax_face[-1, :] / (self.hx[-1] / 2); k = self.idx(s, nx - 1, np.arange(nz))
            rows.append(k); cols.append(k); vals.append(gr); rhs_b[k] += gr * cb
            if self.geo == "ida":
                gl = D * self.Ax_face[0, :] / (self.hx[0] / 2); k = self.idx(s, 0, np.arange(nz))
                rows.append(k); cols.append(k); vals.append(gl); rhs_b[k] += gl * cb
        self.A0 = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N)).tocsr()
        self.rhs_b = rhs_b
        self.V = np.concatenate([self.Vol.ravel(), self.Vol.ravel()])

    def _bv(self, E, mask):
        """Linearised electrode terms: net oxidation flux j = kox*cR - kred*cO (with half-cell correction)."""
        p, h = self.p, self.hz[0] / 2
        x = f * (E - p["E0"])
        kred, kox = p["k0"] * np.exp(-p["alpha"] * x), p["k0"] * np.exp((1 - p["alpha"]) * x)
        den = 1 + kox * h / p["D_R"] + kred * h / p["D_O"]
        i = np.where(mask)[0]
        return i, kox / den, kred / den

    def electrode_matrix(self, Eg, Ec):
        rows, cols, vals = [], [], []
        for E, mask in ((Eg, self.gen), (Ec, self.col)):
            if E is None:
                continue
            i, a, b = self._bv(E, mask)
            A = self.Az[i]
            r, o = self.idx(0, i, 0), self.idx(1, i, 0)
            # R consumed at rate a*cR - b*cO ; O produced at the same rate
            rows += [r, r, o, o]; cols += [r, o, r, o]; vals += [a * A, -b * A, -a * A, b * A]
        N = len(self.V)
        if not rows:
            return coo_matrix((N, N)).tocsr()
        return coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N)).tocsr()

    def current(self, c, E, mask):
        if E is None:
            return 0.0
        i, a, b = self._bv(E, mask)
        cR = c[self.idx(0, i, 0)]; cO = c[self.idx(1, i, 0)]
        return F * np.sum((a * cR - b * cO) * self.Az[i]) * self.scale        # A, anodic positive

    def cv(self, Eg_fun, Ec_fun, t_end, dt):
        c = np.concatenate([np.full(self.nx * self.nz, self.p["c_bulk"]), np.zeros(self.nx * self.nz)])
        t = 0.0; out = []
        Mv = diags(self.V / dt)
        while t < t_end - 1e-12:
            t += dt
            Eg, Ec = Eg_fun(t), Ec_fun(t)
            A = (Mv + self.A0 + self.electrode_matrix(Eg, Ec)).tocsc()
            c = splu(A).solve(self.V / dt * c + self.rhs_b)
            out.append((t, Eg, Ec, self.current(c, Eg, self.gen), self.current(c, Ec, self.col)))
        return np.array(out, dtype=object), c

    @property
    def gen_area(self):
        return float(np.sum(self.Az[self.gen]) * self.scale)

# Ferrocenecarboxylic acid at pH 7.4 (deprotonated, FcCOO- / FcCOO): literature-typical values, edit to your data.
FCCOOH = dict(c_bulk=1.0, D_R=5.7e-10, D_O=5.7e-10, E0=0.32, k0=1e-3, alpha=0.5)
