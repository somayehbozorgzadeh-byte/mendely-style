import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = sys.argv[1] if len(sys.argv) > 1 else "redox_device_spiral.json"
NAME = sys.argv[2] if len(sys.argv) > 2 else "FcMeOH"
EREF = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
R = {r["mode"]: r for r in json.load(open(SRC))}
gc, sg = R["GC"], R["single"]
E = np.array(gc["Eg"]); n = len(E) // 2
fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
ax[0].plot(E, np.array(gc["i_gen"]) * 1e6, "tab:red", label="generator (collector at 0 V)")
ax[0].plot(E, np.array(gc["i_col"]) * 1e6, "tab:red", ls="--", label="collector")
ax[0].plot(E, np.array(sg["i_gen"]) * 1e6, "k:", label="generator alone (collector off)")
ax[0].set_xlabel("E$_{generator}$ / V vs Ag/AgCl"); ax[0].set_ylabel("current / µA"); ax[0].legend(frameon=False, fontsize=8)
ax[0].set_title(f"1 mM {NAME}, 0.1 V/s — whole device", fontsize=10)
k = int(np.argmin(np.abs(E[:n] - EREF)))
ig, ic, i1, i1pk = gc["i_gen"][k], gc["i_col"][k], sg["i_gen"][k], max(sg["i_gen"])
vals = [-ic / ig, ig / i1, ig / i1pk]
ax[1].bar(["collection\nefficiency", f"amplification\n(vs single, at {EREF} V)", "amplification\n(vs single-electrode peak)"], vals, color="tab:red")
for i, v in enumerate(vals): ax[1].text(i, v * 1.02, f"{v:.2f}", ha="center")
ax[1].set_title("Figures of merit", fontsize=10)
for a in ax: a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
fig.suptitle(f"Gold W2G2 entwined spiral in 1 mm × 1 mm (61 turns, r$_{{out}}$ = {gc['r_out']*1e6:.0f} µm, {gc['gen_area']*1e6:.3f} mm² per electrode)", fontsize=11)
fig.tight_layout(); fig.savefig(SRC.replace(".json", ".png"), dpi=140)
print(f"i_gen({EREF} V) {ig*1e6:.2f} uA, i_col {ic*1e6:.2f} uA, single at 0.5 V {i1*1e6:.2f} uA, single peak {i1pk*1e6:.2f} uA")
print(f"CE {vals[0]:.3f}, AF(0.5V) {vals[1]:.2f}, AF(vs peak) {vals[2]:.2f}, j_gen {ig/gc['gen_area']/10:.3f} mA/cm2")
