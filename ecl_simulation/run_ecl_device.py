"""ECL on the real device: gold two-arm entwined spiral, W2G2, filling 1 mm x 1 mm (61 turns, r_out ~ 494 um).
Protocol: generator -0.7 V from t = 0, collector off for 5 s, then collector 0 -> 0.8 V at 0.1 V/s.
Gold = the default kinetic set in ecl_gc_simulation.PARAMS; k_lum_so = 5e3 (fitted earlier)."""
import time
import numpy as np
from scipy.integrate import solve_ivp
import ecl_gc_simulation as m
from ecl_array import ECLArray

V, T_PRE, E2 = 0.1, 5.0, 0.8
t0 = time.time()
p = dict(m.PARAMS); p["k_lum_so"] = 5e3
arr = ECLArray("spiral", p, n_pairs=61, ncell=4)
print("cells", arr.nx * arr.nz, "unknowns", m.nS * arr.nx * arr.nz, flush=True)
Ec = lambda t: None if t < T_PRE else V * (t - T_PRE)
t = np.concatenate([np.linspace(0, T_PRE, 26)[:-1], [T_PRE - 1e-3], T_PRE + np.linspace(0, E2 / V, 81)])
sol = solve_ivp(arr.rhs(Ec), (0, t[-1]), arr.y_bulk(), method="BDF", jac_sparsity=arr.sparsity, t_eval=t, rtol=1e-5, atol=1e-11)
if not sol.success:
    raise RuntimeError(sol.message)
shape = (m.nS, arr.nx, arr.nz)
obs = [arr.observables(sol.y[:, k].reshape(shape), Ec(t[k]))[:3] for k in range(len(t))]
k_pre = int(np.argmin(abs(t - (T_PRE - 1e-3)))); k_pk = int(np.argmin(abs(t - (T_PRE + 0.4 / V))))
np.savez("ecl_device_spiral_gold.npz", t=t, E=np.array([np.nan if Ec(x) is None else Ec(x) for x in t]),
         photons=np.array([o[0] for o in obs]), i_gen=np.array([o[1] for o in obs]), i_col=np.array([o[2] for o in obs]),
         c_pre=sol.y[:, k_pre].reshape(shape).astype(np.float32), c_pk=sol.y[:, k_pk].reshape(shape).astype(np.float32),
         xc=arr.xc, zc=arr.zc, Vol=arr.Vol, scale=arr.scale, gen_area=arr.gen_area, footprint=arr.footprint,
         segs=np.array([(s[0], s[1], {"G": 1, "C": 2, "-": 0}[s[2]]) for s in arr.segs]))
print("ALL DONE %.0f s" % (time.time() - t0), flush=True)
