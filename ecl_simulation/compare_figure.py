"""Side-view maps (cf. SI Fig. S14) and top-view profiles (cf. Fig. 5c) from ecl_simulation.py."""
import sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import ecl_simulation as e
g = e.build(2.0)
runs = {ph: e.simulate(ph, 10.0, rt, g) for ph, rt in ((9.0, 1), (12.0, 2))}
fig, ax = plt.subplots(1, 3, figsize=(15, 5.2), gridspec_kw=dict(width_ratios=[1, 1, 1.15]))
for a, (ph, vmax, ttl) in zip(ax[:2], ((9.0, 10, "Low pH (9)"), (12.0, 30, "High pH (12)"))):
    r_e, z_e, rc, zc, dr, dz = runs[ph]["grid"]; hv = runs[ph]["hv"] / 1000.0
    hv = np.ma.masked_where((rc[:, None] > e.RI) & (zc[None, :] < e.LT), hv)
    m = a.pcolormesh(r_e, z_e, hv.T, cmap="jet", vmin=0, vmax=vmax, shading="flat")
    a.set_xlim(0, e.RO + 2); a.set_ylim(0, 20); a.set_aspect("equal"); a.set_title(ttl + ", 10 mM H2O2 (side view)")
    a.set_xlabel("r / um"); a.set_ylabel("z / um"); plt.colorbar(m, ax=a, label="h$\\nu$ / mM")
for ph, ls, lab in ((9.0, "o-", "Low pH, mid-height"), (12.0, "o--", "High pH, mid-height")):
    r, p = e.top_view(runs[ph]); ra, pa = e.top_view(runs[ph], "avg")
    ax[2].plot(np.r_[-r[::-1], r], np.r_[p[::-1], p] / p.max(), ls, ms=3, label=lab)
ax[2].set_xlabel("Position / um"); ax[2].set_ylabel("Normalised intensity"); ax[2].set_title("Radial profile (cf. Fig. 5c)"); ax[2].legend()
plt.tight_layout(); plt.savefig("compare_side_view.png", dpi=130)
for ph in runs:
    hv = runs[ph]["hv"]/1000; r_e, z_e, rc, zc, dr, dz = runs[ph]["grid"]
    i, j = np.unravel_index(np.argmax(hv), hv.shape); print(f"pH {ph:g}: max {hv.max():.2f} mM at r={rc[i]:.2f}, z={zc[j]:.2f} um")
