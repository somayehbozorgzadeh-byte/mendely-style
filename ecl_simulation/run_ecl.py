"""Run the generator-collector ECL simulation and save figures."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ecl_gc_simulation as m

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
EG, EC = -0.8, 0.7

# 1) transient after simultaneous potential step (gen -0.8 V, col +0.7 V)
tr = m.run(Eg=EG, Ec=EC, t_end=0.5, t_eval=np.linspace(0, 0.5, 251))
# 2) steady state
e_ss, ig, ic, ss = m.steady(Eg=EG, Ec=EC, t_end=10)
C, xc = ss["C"][:, :, -1], ss["xc"] * 1e6

fig, ax = plt.subplots(2, 3, figsize=(14, 7.5))
a = ax[0, 0]
a.plot(tr["t"], tr["ecl"] / 1e12, "k", label="ECL")
a.set_xlabel("t / s"); a.set_ylabel("ECL / 10$^{12}$ photons m$^{-2}$ s$^{-1}$")
b = a.twinx(); b.spines["right"].set_visible(True)
b.plot(tr["t"], tr["i_gen"] / 10, "tab:blue", label="generator (cathodic)")
b.plot(tr["t"], tr["i_col"] / 10, "tab:red", label="collector (anodic)")
b.set_ylabel("|i| / mA cm$^{-2}$"); b.legend(loc="center right", frameon=False)
a.set_title("Transient after double potential step")

a = ax[0, 1]
for k, name, col in [(m.iO2, "O$_2$", "tab:gray"), (m.iSO, "O$_2^{\\bullet-}$ (×1)", "tab:blue"),
                     (m.iHP, "H$_2$O$_2$", "tab:cyan")]:
    a.plot(xc, C[k] * 1e3, label=name, color=col)
a.set_xlabel("x / µm  (generator → collector)"); a.set_ylabel("c / µM"); a.legend(frameon=False)
a.set_title("Steady-state: O$_2$ branch")

a = ax[0, 2]
for k, name, col in [(m.iLH, "LH$^-$", "tab:gray"), (m.iRad, "L$^{\\bullet-}$ (×100)", "tab:red"),
                     (m.iD, "diazaquinone", "tab:orange")]:
    a.plot(xc, C[k] * 1e3 * (100 if k == m.iRad else 1), label=name, color=col)
a.set_xlabel("x / µm"); a.set_ylabel("c / µM"); a.legend(frameon=False)
a.set_title("Steady-state: luminol branch")

# ECL emission profile
_, ph = m.solution_rates(C, ss["p"])
a = ax[1, 0]
a.plot(xc, ph * m.NA / 1e21, "k")
a.fill_between(xc, 0, ph * m.NA / 1e21, alpha=.2, color="k")
a.set_xlabel("x / µm"); a.set_ylabel("emission rate / 10$^{21}$ photons m$^{-3}$ s$^{-1}$")
a.set_title("Where the light is generated")

# 3) collector potential scan @ Eg = -0.8
Ecs = np.linspace(0.0, 0.9, 19)
res = [m.steady(Eg=EG, Ec=e, t_end=8, n=80) for e in Ecs]
a = ax[1, 1]
a.plot(Ecs, [r[0] / 1e12 for r in res], "ko-", ms=4, label="ECL")
a.set_xlabel("E$_{collector}$ / V vs Ag/AgCl (E$_{gen}$ = −0.8 V)"); a.set_ylabel("ECL / 10$^{12}$ photons m$^{-2}$ s$^{-1}$")
b = a.twinx(); b.spines["right"].set_visible(True)
b.plot(Ecs, [r[2] / 10 for r in res], "tab:red", label="i$_{col}$"); b.set_ylabel("i$_{col}$ / mA cm$^{-2}$")
a.set_title("Collector potential scan")

# 4) generator potential scan @ Ec = +0.7
Egs = np.linspace(-1.1, -0.2, 19)
res2 = [m.steady(Eg=e, Ec=EC, t_end=8, n=80) for e in Egs]
a = ax[1, 2]
a.plot(Egs, [r[0] / 1e12 for r in res2], "ko-", ms=4)
a.set_xlabel("E$_{generator}$ / V vs Ag/AgCl (E$_{col}$ = +0.7 V)"); a.set_ylabel("ECL / 10$^{12}$ photons m$^{-2}$ s$^{-1}$")
b = a.twinx(); b.spines["right"].set_visible(True)
b.plot(Egs, [r[1] / 10 for r in res2], "tab:blue"); b.set_ylabel("|i$_{gen}$| / mA cm$^{-2}$")
a.set_title("Generator potential scan")
fig.tight_layout(); fig.savefig("ecl_generator_collector.png", dpi=150)

# 5) gap dependence
gaps = np.array([0.5, 1, 2, 5, 10, 20]) * 1e-6
rg = []
for g in gaps:
    p = dict(m.PARAMS); p["gap"] = g
    rg.append(m.steady(p, Eg=EG, Ec=EC, t_end=8, n=80))
fig, a = plt.subplots(figsize=(5.5, 4))
a.loglog(gaps * 1e6, [r[0] / 1e12 for r in rg], "ko-")
a.set_xlabel("gap d / µm"); a.set_ylabel("ECL / 10$^{12}$ photons m$^{-2}$ s$^{-1}$ (per electrode area)")
a.set_title("Gap dependence"); fig.tight_layout(); fig.savefig("ecl_gap_dependence.png", dpi=150)

print(f"pH {m.PARAMS['pH']}, d = {m.PARAMS['gap']*1e6:.1f} um, Eg={EG} V, Ec={EC} V")
print(f"steady ECL = {e_ss:.3e} photons m^-2 s^-1;  i_gen = {ig/10:.2f} mA/cm2;  i_col = {ic/10:.2f} mA/cm2")
print("peak collector potential:", Ecs[int(np.argmax([r[0] for r in res]))], "V")
print("peak generator potential:", Egs[int(np.argmax([r[0] for r in res2]))], "V")
print("gap scan [um, ECL/1e12]:", [(round(g*1e6,1), round(r[0]/1e12,2)) for g, r in zip(gaps, rg)])
