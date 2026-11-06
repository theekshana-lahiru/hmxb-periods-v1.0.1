"""Cen X-3: fit of the apparent pulse periods of the 10-ks segments of XMM-Newton observation
0400550201 with the orbital Doppler curve (Sect. 4.10, Fig. 4), and barycentric correction of
the spin period for the Earth's velocity. Writes cenx3_doppler.json."""
import numpy as np
from astropy.coordinates import SkyCoord, get_body_barycentric_posvel
from astropy.time import Time
import astropy.units as u
from scipy.optimize import least_squares

import common

# Falanga et al. (2015): T_pi/2 (MJD), P_orb (d), Pdot/P (1/yr), a_x sin i (lt-s)
T0, P0, PDP_YR, ASINI = 50506.788423, 2.08704106, -1.800e-6, 39.6612
PDOT = PDP_YR * P0 / 365.25
RA, DEC = 170.31329, -60.62378


def cycles(t):
    a, b, c = 0.5 * P0 * PDOT, P0, T0 - np.asarray(t, float)
    return (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)


def segments(obs_id):
    d = common.xmm_json("V779 Cen")
    rows = []
    for o in d["observations"]:
        if o["obs_id"] != obs_id:
            continue
        for sg in o.get("segments", []):
            sig = sg.get("significant") or []
            if sig:
                tm = o["mjd_start"] + 0.5 * (sg["segment"][0] + sg["segment"][1]) / 86400.0
                rows.append((tm, sig[0]["P"], sig[0]["sigma_P"]))
    return np.array(rows)


def earth_factor(mjd):
    nvec = SkyCoord(ra=RA * u.deg, dec=DEC * u.deg).cartesian.xyz.value
    _, vel = get_body_barycentric_posvel("earth", Time(mjd, format="mjd", scale="tt"))
    return float(np.dot(vel.xyz.to(u.km / u.s).value, nvec)) / 299792.458


def main():
    seg = segments("0400550201")
    t, P, sP = seg.T
    lam = 2 * np.pi * cycles(t) + np.pi / 2
    Porb_s = P0 * 86400.0

    def resid(p):
        Ps, A, dl = p
        return (P - Ps * (1 + A * np.cos(lam + dl))) / sP
    r = least_squares(resid, [4.805, 2 * np.pi * ASINI / Porb_s, 0.0])
    J = r.jac
    cov = np.linalg.inv(J.T @ J)
    chi2 = float(np.sum(r.fun ** 2))
    dof = len(P) - 3
    sc = max(chi2 / dof, 1.0)
    pe = np.sqrt(np.diag(cov) * sc)
    Ps, A, dl = r.x
    uE = earth_factor(t.mean())
    out = dict(n_segments=len(P), Ps_topo=Ps, Ps_topo_err=pe[0], asini=A * Porb_s / (2 * np.pi),
               asini_err=pe[1] * Porb_s / (2 * np.pi), dphase=dl / (2 * np.pi), dphase_err=pe[2] / (2 * np.pi),
               chi2=chi2, dof=dof, error_scale=float(np.sqrt(sc)), earth_u_over_c=uE, Ps_bary=Ps * (1 + uE),
               phases=list(cycles(t) % 1.0), periods=list(P), errors=list(sP), times=list(t))
    s01 = segments("0111010101")
    if len(s01):
        t1, P1, e1 = s01[0]
        A_lit = 2 * np.pi * ASINI / Porb_s
        P1c = P1 / (1 + A_lit * np.cos(2 * np.pi * cycles(t1) + np.pi / 2))
        out.update(P2001_app=P1, P2001_err=e1, P2001_bary=P1c * (1 + earth_factor(t1)))
    common.dump(out, "cenx3_doppler.json")
    print({k: v for k, v in out.items() if not isinstance(v, list)})


if __name__ == "__main__":
    main()
