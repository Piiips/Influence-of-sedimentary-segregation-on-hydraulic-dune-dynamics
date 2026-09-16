import sys
sys.path.append('.')
import numpy as np
import analisis_figuras_jfm as af

eta_a = float(np.mean(af.A.eta_active_layer(af.M)[af.DUNE]))
phf = af.A._phi_faces(af.M, af._load('dif', 0.7)[0])
ell_a = float(np.median(af.ell_faces(phf)[af.DUNE][:, -6:]))

print(f"eta_a = {eta_a}, ell_a = {ell_a}, h_max = {af.M.h.max()}")

def calc_C(phi0, ell, eta_lo, eta_hi):
    n = 200
    e = np.linspace(eta_lo, eta_hi, n)
    z = (e - eta_lo) * af.M.h.max()
    lo, hi = -40.0, 40.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        p = 1.0 / (1.0 + np.exp(mid + z / ell))
        if p.mean() > phi0:
            lo = mid
        else:
            hi = mid
    logC = 0.5 * (lo + hi)
    return np.exp(logC), logC

for p0 in af.PHIS:
    C, logC = calc_C(p0, ell_a, eta_a, 1.0)
    print(f"phi0 = {p0}: C = {C:.4e}, ln(C) = {logC:.4f}")

