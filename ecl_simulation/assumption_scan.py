"""Scan the two assumptions not fixed by the SI: H2O2 pKa and whether/how fast L diffuses."""
import numpy as np, ecl_simulation as e
g = e.build(1.0)
PKAS = [10.5, 11.0, 11.6, 11.75, 12.0]
DLS = [0.0, 0.5, 1.0, 1.5]      # D_L(L) / D_L
def run(ph, rt, c, **kw):
    res = e.simulate(ph, c, rt, g, **kw); r, p = e.top_view(res); ra, pa = e.top_view(res, "avg")
    return e.tel(r, p), e.tel(ra, pa), p[0] / p.max()
print("== pKa scan (L diffuses, D_L) -- TEL(1/e) mid-height / depth-avg [um], centre/wall ==")
for ph, rt in ((9.0, 1), (12.0, 2)):
    for c in (1.0, 10.0):
        print(f"pH {ph:g}, {c:g} mM H2O2")
        for pka in PKAS:
            t, ta, cw = run(ph, rt, c, pka=pka)
            print(f"   pKa {pka:5.2f}: [HO2-]={e.hmax(ph,c,pka):9.1f} uM  TEL={t:5.2f}/{ta:5.2f}  c/w={cw:.3f}")
print("\n== L-diffusion scan (Route I, pH 9, pKa 11.6) ==")
for c in (0.1, 1.0, 10.0):
    print(f"{c:g} mM H2O2")
    for f in DLS:
        t, ta, cw = run(9.0, 1, c, dL_L=f)
        print(f"   D_L(L) = {f:3.1f} x D_L: TEL={t:5.2f}/{ta:5.2f}  c/w={cw:.3f}")
print("\n== joint scan, pH 9, 10 mM (TEL mid-height, um) ==")
print("pKa\\D_L(L)  " + "  ".join(f"{f:4.1f}" for f in DLS))
for pka in PKAS:
    print(f"{pka:5.2f}       " + "  ".join(f"{run(9.0,1,10.0,pka=pka,dL_L=f)[0]:4.2f}" for f in DLS))
print("\n== pH 8 (more HO2-depleted regime), 10 mM, pKa scan ==")
for pka in PKAS:
    t, ta, cw = run(8.0, 1, 10.0, pka=pka); print(f"   pKa {pka:5.2f}: TEL={t:5.2f}/{ta:5.2f} c/w={cw:.3f}")
