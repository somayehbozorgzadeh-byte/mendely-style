"""Compare ecl_simulation.py against the paper's COMSOL results (Fig. 5, SI Fig. S14).

Targets read off the published figures (10 mM H2O2, 1 mM luminol):
  Fig. S14 / 5a  low pH : hv colour scale 0-10 mM, maximum at the electrode wall,
                          Fig. 5a top view runs 2 (centre) -> 10 (edge) mM
  Fig. S14 / 5b  high pH: hv colour scale 0-30 mM, thin ring at the wall, ~0 at centre
  text           TEL (visibility cut-off, camera) > 9.1 um low pH, ca. 4.1 um high pH
"""
import numpy as np, sys
import ecl_simulation as e
g = e.build(2.0)
VARIANTS = {   # name: (pka, dL_L)  pka=-50 -> [HO2-] = total H2O2 (no pKa; literal Table S2 reading)
    "A  pKa 11.6, L diffuses (my default)": (11.6, 1.0),
    "B  [HO2-]=c0(H2O2), L diffuses":        (-50.0, 1.0),
    "C  [HO2-]=c0(H2O2), L fixed (S10 literal)": (-50.0, 0.0),
    "D  pKa 11.6, L fixed":                  (11.6, 0.0),
}
def stats(res):
    r_e, z_e, rc, zc, dr, dz = res["grid"]; hv = res["hv"] / 1000.0; iw = res["iw"]
    zin = zc < e.LT; mid = np.argmin(abs(zc - e.LT / 2))
    avg = (hv[:iw + 1][:, zin] * dz[zin]).sum(1) / e.LT
    return dict(max=hv.max(), edge_mid=hv[iw, mid], centre_mid=hv[0, mid], centre_avg=avg[0], edge_avg=avg[-1],
                tel5=lambda: None)
def tel_thr(r, p, frac):
    p = p / p.max(); b = np.where(p < frac)[0]; return e.RI - r[b.max()] if b.size else e.RI
if __name__ == "__main__":
    out = []
    print(f"{'variant':46s} {'pH':>3s} {'max hv/mM':>9s} {'ctr/edge(mid)':>13s} {'ctr/edge(avg)':>13s} "
          f"{'TEL5%':>6s} {'TEL1%':>6s}")
    print("PAPER (10 mM H2O2)".ljust(46) + "  9      ~10          ~0.2(5a)                >9.1 (cut-off)")
    print(" " * 46 + " 12      ~30          ~0          ~0.0           ca. 4.1 (cut-off)")
    for name, (pka, dl) in VARIANTS.items():
        for ph, rt in ((9.0, 1), (12.0, 2)):
            res = e.simulate(ph, 10.0, rt, g, pka=pka, dL_L=dl); s = stats(res)
            r, p = e.top_view(res); ra, pa = e.top_view(res, "avg")
            print(f"{name:46s} {ph:3g} {s['max']:9.2f} {s['centre_mid']/s['edge_mid']:13.3f} "
                  f"{s['centre_avg']/s['edge_avg']:13.3f} {tel_thr(r,p,0.05):6.2f} {tel_thr(r,p,0.01):6.2f}")
