import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import ecl_gc_simulation as m

B = json.load(open("flux_budget.json"))
E = np.array([b["E"] for b in B])
g = lambda k: np.array([b[k] for b in B])

# ---- Fig 1: flux maps at ECL peak potential --------------------------------
Epk = 0.4
d = np.load(f"flux_state_{Epk}.npz")
c, xc, zc = d["c"], d["xc"], d["zc"]
p = m.PARAMS
ZMAX = 6e-6
xr = np.linspace(xc[0], xc[-1], 160); zr = np.linspace(zc[0], ZMAX, 120)
XR, ZR = np.meshgrid(xr, zr, indexing="ij")

def field(k):
    gx, gz = np.gradient(c[k], xc, zc, edge_order=2)
    jx, jz = -p["D"][k] * gx, -p["D"][k] * gz
    f = lambda a: RegularGridInterpolator((xc, zc), a)(np.stack([XR, ZR], -1))
    return f(c[k]), f(jx), f(jz)

_, ph = (None, m.solution_rates(c, p)[1] * m.NA)
phi = RegularGridInterpolator((xc, zc), ph)(np.stack([XR, ZR], -1))
fig, ax = plt.subplots(1, 3, figsize=(15, 4.3), sharey=True)
panels = [(m.iSO, 1e3, "O$_2^{\\bullet-}$ / µM  +  diffusive flux", "Blues"),
          (m.iRad, 1e6, "L$^{\\bullet-}$ / nM  +  diffusive flux", "Reds"),
          (m.iHP, 1e3, "H$_2$O$_2$ / µM  +  diffusive flux", "Greens")]
for a, (k, sc, ttl, cm) in zip(ax[:2], panels[:2]):
    cf, jx, jz = field(k)
    pc = a.pcolormesh(xr * 1e6, zr * 1e6, (cf * sc).T, cmap=cm, shading="auto")
    fig.colorbar(pc, ax=a, pad=0.02)
    spd = np.hypot(jx, jz).T
    a.streamplot(xr * 1e6, zr * 1e6, jx.T, jz.T, color="k", density=1.5, linewidth=0.5 + 1.8 * spd / spd.max(), arrowsize=0.9)
    a.contour(xr * 1e6, zr * 1e6, phi.T, levels=[0.25 * phi.max()], colors="m", linewidths=1.3, linestyles="--")
    a.set_title(ttl, fontsize=10)
cf, jx, jz = field(m.iHP)
ax[2].pcolormesh(xr * 1e6, zr * 1e6, (phi / 1e21).T, cmap="magma", shading="auto")
pc = ax[2].collections[0]; fig.colorbar(pc, ax=ax[2], pad=0.02)
ax[2].set_title("ECL emission / 10$^{21}$ photons m$^{-3}$ s$^{-1}$", fontsize=10)
for a in ax:
    a.set_xlabel("x / µm"); a.set_ylim(0, ZMAX * 1e6); a.set_xlim(0, 4)
    a.plot([0, 1], [0, 0], color="tab:blue", lw=7, solid_capstyle="butt", clip_on=False)
    a.plot([3, 4], [0, 0], color="tab:red", lw=7, solid_capstyle="butt", clip_on=False)
ax[0].set_ylabel("z / µm"); 
fig.suptitle(f"w = 2 µm, gap = 2 µm, E$_{{gen}}$ = −0.8 V, E$_{{col}}$ = {Epk} V (steady state).  "
             "Blue bar: generator (O$_2$ → O$_2^{\\bullet-}$);  red bar: collector (LH$^-$ → L$^{\\bullet-}$).  "
             "Magenta dashed: 25% emission contour", fontsize=9)
fig.tight_layout(); fig.savefig("ecl_flux_maps.png", dpi=150)

# ---- Fig 2: budgets --------------------------------------------------------
src = g("gen_O2red") + g("rad_o2")
fates = [("ECL (L$^{\\bullet-}$ + O$_2^{\\bullet-}$)", g("ecl_rad"), "gold"),
         ("dismutation → H$_2$O$_2$ + O$_2$", 2 * g("dismut"), "tab:cyan"),
         ("re-oxidised at collector", g("col_SOox"), "tab:red"),
         ("over-reduced at generator", g("gen_overred"), "tab:purple"),
         ("lost to bulk", g("SO_to_bulk"), "0.6")]
bal = sum(f[1] for f in fates) / src
fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
xs = np.arange(len(E)); bot = np.zeros(len(E))
for name, v, col in fates:
    ax[0].bar(xs, 100 * v / src, bottom=bot, color=col, label=name, width=0.7); bot += 100 * v / src
ax[0].set_xticks(xs, [f"{e:g}" for e in E]); ax[0].set_ylabel("% of O$_2^{\\bullet-}$ produced")
ax[0].set_xlabel("E$_{collector}$ / V"); ax[0].set_title("Fate of superoxide made at the generator", fontsize=10)
ax[0].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)

hsrc = g("dismut") + g("gen_overred")
hf = [("lost to bulk", g("HP_to_bulk"), "0.6"), ("oxidised at collector", g("col_HPox"), "tab:red"),
      ("ECL via L + H$_2$O$_2$", g("ecl_hp"), "gold")]
bot = np.zeros(len(E))
for name, v, col in hf:
    ax[1].bar(xs, 100 * v / hsrc, bottom=bot, color=col, label=name, width=0.7); bot += 100 * v / hsrc
ax[1].set_xticks(xs, [f"{e:g}" for e in E]); ax[1].set_xlabel("E$_{collector}$ / V")
ax[1].set_ylabel("% of H$_2$O$_2$ produced"); ax[1].set_title("Fate of H$_2$O$_2$ (second ROS)", fontsize=10)
ax[1].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)

ax[2].plot(E, g("photons") * m.NA / g("gen_O2red") / m.NA * 1e0 * 1e3, "ko-")
ax[2].set_xlabel("E$_{collector}$ / V"); ax[2].set_ylabel("photons per O$_2$ reduced at generator / 10$^{-3}$")
ax[2].set_title("ROS → light yield (φ$_{ECL}$ = 1% assumed)", fontsize=10)
for a in ax:
    a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
fig.tight_layout(); fig.savefig("ecl_ros_budget.png", dpi=150)

print("steady-state drift:", np.array2string(g("drift"), precision=1))
print("SO balance (sinks/sources):", np.array2string(bal, precision=3))
b = B[list(E).index(Epk)]
print(f"\nAt E_col = {Epk} V [nmol m^-1 s^-1, per unit cell]:")
for k in ["gen_O2red", "rad_o2", "gen_overred", "dismut", "ecl_rad", "ecl_hp", "col_SOox", "col_HPox", "SO_to_bulk", "HP_to_bulk", "L_ox_col"]:
    print(f"  {k:12s} {b[k]*1e9:10.4f}")
print("percent of superoxide produced:", {n: round(100 * v[list(E).index(Epk)] / src[list(E).index(Epk)], 1) for n, v, _ in fates})
