"""2 um gap / 2 um wide entwined electrodes: collector CV 0 -> 0.8 -> 0 V at 0.1 V/s, generator at -0.8 V."""
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ecl_gc_simulation as m
from ecl_gc_2d import Cell2D

EG, E1, E2, V = -0.8, 0.0, 0.8, 0.1
cell = Cell2D(m.PARAMS, w=2e-6, g=2e-6, Eg=EG)
tf = (E2 - E1) / V
Ec = lambda t: E1 + V * t if t <= tf else E2 - V * (t - tf)

t0 = time.time()
y0 = cell.integrate(E1, (0, 6), cell.y_bulk(), np.array([0, 6.0])).y[:, -1]
t = np.linspace(0, 2 * tf, 321)
sol = cell.integrate(Ec, (0, 2 * tf), y0, t)
print("solve time %.0f s" % (time.time() - t0))
shape = (m.nS, cell.nx, cell.nz)
E = np.array([Ec(x) for x in t])
obs = np.array([cell.observables(sol.y[:, k].reshape(shape), E[k])[:3] for k in range(len(t))])
ecl, ig, ic = obs[:, 0], obs[:, 1] / 10, obs[:, 2] / 10     # mA/cm2
np.savetxt("cv2d_0p1Vs.csv", np.c_[t, E, ecl, ic, ig], delimiter=",", comments="",
           header="t_s,E_collector_V_vsAgAgCl,ECL_photons_per_footprint_m-2_s-1,i_collector_mA_cm-2,i_generator_mA_cm-2")
fwd = t <= tf
ipk = np.argmax(ecl[fwd])
print(f"forward ECL peak {ecl[ipk]:.3e} photons m-2 s-1 at {E[ipk]:.2f} V; i_col {ic[ipk]:.2f}, i_gen {ig[ipk]:.2f} mA/cm2")
print(f"i_col baseline(0 V) {ic[0]:.2f}, at 0.8 V {ic[fwd][-1]:.2f} mA/cm2; ECL at 0.8 V {ecl[fwd][-1]:.3e}")
print(f"reverse ECL peak {ecl[~fwd].max():.3e} at {E[~fwd][np.argmax(ecl[~fwd])]:.2f} V")

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for a, y, lab, col in [(ax[0], ic, "collector current / mA cm$^{-2}$", "tab:red"),
                       (ax[1], ecl / 1e12, "ECL / 10$^{12}$ photons m$^{-2}$ s$^{-1}$ (footprint)", "k")]:
    a.plot(E[fwd], y[fwd], color=col, label="forward")
    a.plot(E[~fwd], y[~fwd], "--", color=col, alpha=.6, label="reverse")
    a.set_xlabel("E$_{collector}$ / V vs Ag/AgCl"); a.set_ylabel(lab); a.legend(frameon=False)
    a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
ax[0].plot(E[fwd], ig[fwd], color="tab:blue", label="generator")
ax[0].legend(frameon=False)
ax[0].set_title("w = 2 µm, gap = 2 µm, E$_{gen}$ = −0.8 V, 0.1 V/s"); ax[1].set_title("ECL–potential curve")
fig.tight_layout(); fig.savefig("ecl_cv_2d_0p1Vs.png", dpi=150)

# concentration / emission maps at the ECL peak (forward sweep)
k = int(np.argmax(np.where(fwd, ecl, 0)))
c = sol.y[:, k].reshape(shape)
ph = m.solution_rates(c, cell.p)[1] * m.NA
X, Z = np.meshgrid(cell.xc * 1e6, cell.zc * 1e6, indexing="ij")
fig, ax = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
zmax = 8
for a, (arr, ttl, cm) in zip(ax, [(c[m.iSO] * 1e3, "O$_2^{\\bullet-}$ / µM", "Blues"),
                                  (c[m.iRad] * 1e6, "L$^{\\bullet-}$ / nM", "Reds"),
                                  (c[m.iLH] * 1e3, "LH$^-$ / µM", "Greys"),
                                  (ph / 1e21, "emission / 10$^{21}$ photons m$^{-3}$ s$^{-1}$", "magma")]):
    pc = a.pcolormesh(X, Z, arr, shading="auto", cmap=cm); fig.colorbar(pc, ax=a); a.set_title(ttl, fontsize=9)
    a.set_ylim(0, zmax); a.set_xlabel("x / µm")
    a.axvspan(0, 1, ymax=0.02, color="tab:blue", lw=0); a.axvspan(3, 4, ymax=0.02, color="tab:red", lw=0)
ax[0].set_ylabel("z / µm (above electrodes)")
fig.suptitle(f"E$_{{col}}$ = {E[k]:.2f} V  (blue bar: generator half-strip, red bar: collector half-strip)", fontsize=10)
fig.tight_layout(); fig.savefig("ecl_maps_2d.png", dpi=150)
