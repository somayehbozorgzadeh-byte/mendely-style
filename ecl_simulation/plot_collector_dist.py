"""Distribution of ECL / collector current along the collector width (E_col = 0.4 V steady state)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ecl_gc_simulation as m
from ecl_gc_2d import Cell2D

cell = Cell2D(m.PARAMS, w=2e-6, g=2e-6, Eg=-0.8, nxs=(8, 18, 8), nz=30)
d = np.load("flux_state_0.4.npz"); c = d["c"]
ph = m.solution_rates(c, cell.p)[1] * m.NA
em_x = (ph * cell.hz[None, :]).sum(1)                 # photons m^-2(strip) per x  [per unit length: * hx]
_, _, sc, nc = cell.fluxes(c, 0.4)
jcol = -m.F * nc / 10                                 # mA/cm2 local anodic current
xs = cell.xc[cell.col]
wc = cell.hx[cell.col]
em_col = em_x[cell.col]
frac_total = np.sum(em_x * cell.hx)
print("ECL emitted above collector strip (x in 3-4 um): %.1f%%" % (100 * np.sum(em_col * wc) / frac_total))
print("ECL emitted in gap region (x in 1-3 um):          %.1f%%" % (100 * np.sum(em_x[(cell.xc > 1e-6) & (cell.xc < 3e-6)] * cell.hx[(cell.xc > 1e-6) & (cell.xc < 3e-6)]) / frac_total))
print("ECL above generator (x<1 um):                     %.1f%%" % (100 * np.sum(em_x[cell.gen] * cell.hx[cell.gen]) / frac_total))
# distance from the gap-facing collector edge (x=3 um); collector half-strip x=3..4 um is mirrored (full width 2 um centered on x=4)
dist = (xs - 3e-6) * 1e6
for lim in (0.25, 0.5, 1.0):
    sel = dist <= lim
    print("within %.2f um of gap-facing collector edge: %.1f%% of collector-surface ECL, %.1f%% of collector current (that is %.0f%% of its half-width)"
          % (lim, 100 * np.sum(em_col[sel] * wc[sel]) / np.sum(em_col * wc), 100 * np.sum(jcol[sel] * wc[sel]) / np.sum(jcol * wc),
             100 * lim / 1.0))
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].plot(cell.xc * 1e6, em_x / em_x.max(), "k")
ax[0].axvspan(0, 1, color="tab:blue", alpha=.12, label="generator"); ax[0].axvspan(3, 4, color="tab:red", alpha=.12, label="collector (centre at 4 µm)")
ax[0].set_xlabel("x / µm"); ax[0].set_ylabel("ECL, integrated over z (normalised)"); ax[0].legend(frameon=False)
ax[0].set_title("Where along x the light is produced", fontsize=10)
ax[1].plot(dist, jcol, color="tab:red", label="collector current density")
ax[1].set_xlabel("distance from gap-facing collector edge / µm   (collector centre at 1 µm)")
ax[1].set_ylabel("local anodic current / mA cm$^{-2}$", color="tab:red")
b = ax[1].twinx(); b.plot(dist, em_col / em_col.max(), "k--", label="ECL (norm.)"); b.set_ylabel("ECL above surface, normalised")
ax[1].set_title("Across the collector surface", fontsize=10)
for a in ax: a.spines["top"].set_visible(False)
fig.tight_layout(); fig.savefig("ecl_collector_distribution.png", dpi=150)
