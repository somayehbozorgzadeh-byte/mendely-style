"""Collector potential sweep 0 -> 0.8 V (-> 0 V) at 0.1 V/s, generator held at -0.8 V."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import ecl_gc_simulation as m

EG, E1, E2, V = -0.8, 0.0, 0.8, 0.1
tf = (E2 - E1) / V
Ec = lambda t: E1 + V * t if t <= tf else E2 - V * (t - tf)
n = 100
p = dict(m.PARAMS)

# equilibrate with collector at E1
y0 = m.run(p, Eg=EG, Ec=E1, t_end=8, n=n, t_eval=np.array([0, 8.0]))["C"][:, :, -1].ravel()
rhs, sp, xf, xc, h = m.build(p, EG, Ec, n)
t = np.linspace(0, 2 * tf, 801)
sol = solve_ivp(rhs, (0, 2 * tf), y0, method="BDF", jac_sparsity=sp, t_eval=t, rtol=1e-6, atol=1e-11)
C = sol.y.reshape(m.nS, n, -1)
E, ecl, ig, ic = [], [], [], []
for k, tk in enumerate(t):
    c = C[:, :, k]
    E.append(Ec(tk))
    ecl.append(np.sum(m.solution_rates(c, p)[1] * h) * m.NA)
    ig.append(m.F * m.electrode_flux(c[:, 0], h[0], p, EG, E[-1], "gen")[1] / 10)
    ic.append(-m.F * m.electrode_flux(c[:, -1], h[-1], p, EG, E[-1], "col")[1] / 10)
E, ecl, ig, ic = map(np.array, (E, ecl, ig, ic))
np.savetxt("cv_0p1Vs.csv", np.c_[t, E, ecl, ic, ig], delimiter=",",
           header="t_s,E_collector_V_vsAgAgCl,ECL_photons_m-2_s-1,i_collector_mA_cm-2(anodic+),i_generator_mA_cm-2(cathodic+)", comments="")

fwd = t <= tf
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for a, y, lab in [(ax[0], ic, "collector current / mA cm$^{-2}$"), (ax[1], ecl / 1e12, "ECL / 10$^{12}$ photons m$^{-2}$ s$^{-1}$")]:
    a.plot(E[fwd], y[fwd], "tab:red" if a is ax[0] else "k", label="forward (0 → 0.8 V)")
    a.plot(E[~fwd], y[~fwd], "--", color="tab:red" if a is ax[0] else "k", alpha=.6, label="reverse")
    a.set_xlabel("E$_{collector}$ / V vs Ag/AgCl"); a.set_ylabel(lab); a.legend(frameon=False)
    a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
ax[0].set_title("CV: collector (E$_{gen}$ = −0.8 V, 0.1 V/s)"); ax[1].set_title("ECL–potential curve")
fig.tight_layout(); fig.savefig("ecl_cv_0p1Vs.png", dpi=150)
i = np.argmax(ecl[fwd])
print(f"forward ECL peak {ecl[i]:.3e} at {E[i]:.2f} V; i_col there {ic[i]:.2f} mA/cm2")
print(f"ECL at 0.8 V fwd {ecl[fwd][-1]:.3e}; reverse peak {ecl[~fwd].max():.3e} at {E[~fwd][np.argmax(ecl[~fwd])]:.2f} V")
