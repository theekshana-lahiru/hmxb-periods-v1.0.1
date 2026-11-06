"""Bounds on the timing corrections that were not applied to the apparent XMM spin periods
(Sect. 4.9, Table A2):
  Earth   u/c, the barycentric velocity of the Earth towards the source at the observation
  XMM     2.6e-5, the largest orbital speed of the spacecraft (7.7 km/s at perigee)
  binary  v_max/c of the neutron star for sin i = 1: Kepler orbit with the BAT (or catalogue)
          orbital period, M_NS = 1.4 Msun, companion mass from the catalogue (else 15 Msun for Be
          and 25 Msun for other systems) and eccentricity from the catalogue (else 0.5 for Be,
          0 otherwise): v_max = 2 pi a_NS / P (1 + e) / sqrt(1 - e^2); 2e-3 if the orbit is unknown.
A difference from the catalogue is attributed to a change of the spin period only if
|dP/P| > delta_max + 3 sigma_rel, sigma_rel combining our formal error and the catalogue error
(half a unit of the last quoted digit if the catalogue gives none). Writes xmm_timing_bounds.csv.
"""
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord, get_body_barycentric_posvel
from astropy.time import Time
import astropy.units as u

import common
import config

C = 299792.458
GM_SUN = 1.32712440018e20
D_XMM = 2.6e-5


def rounding_unc(v):
    s = np.format_float_positional(float(v), trim="-")
    dec = len(s.split(".")[1]) if "." in s else 0
    return 0.5 * 10 ** (-dec)


def main():
    s = common.load_summary()
    w = common.load_catalogue()
    rows = []
    for _, r in s[s["P_spin (s)"].notna()].iterrows():
        n, cn = r["Star (folder)"], r["Catalogue name"]
        d = common.xmm_json(n)
        b = d["best"]
        o = next(o for o in d["observations"] if o["obs_id"] == b["obs_id"])
        tmid = o["mjd_start"] + 0.5 * o["T"] / 86400.0
        nvec = SkyCoord(ra=float(w.loc[cn, "RAdeg"]) * u.deg, dec=float(w.loc[cn, "DEdeg"]) * u.deg).cartesian.xyz.value
        _, vel = get_body_barycentric_posvel("earth", Time(tmid, format="mjd", scale="tt"))
        uE = float(np.dot(vel.xyz.to(u.km / u.s).value, nvec)) / C
        Porb = r["P_orb (d)"] if np.isfinite(r["P_orb (d)"]) else r["Literature P_orb (d)"]
        be = str(r["Class"]).startswith("Be")
        Mc = w.loc[cn, "Mo"] if np.isfinite(w.loc[cn, "Mo"]) else (15.0 if be else 25.0)
        e = w.loc[cn, "Eccentricity"] if np.isfinite(w.loc[cn, "Eccentricity"]) else (0.5 if be else 0.0)
        if np.isfinite(Porb):
            Ps = Porb * 86400.0
            a = (GM_SUN * (Mc + 1.4) * Ps ** 2 / (4 * np.pi ** 2)) ** (1 / 3)
            vmax = 2 * np.pi * a * Mc / (Mc + 1.4) / Ps * (1 + e) / np.sqrt(1 - e ** 2) / 1e3
            dorb = vmax / C
        else:
            vmax, dorb = np.nan, 2e-3
        dmax = abs(uE) + D_XMM + dorb
        Pc = r["Literature P_spin (s)"]
        ec = w.loc[cn, "Spin_period_err"]
        ec_used, ec_src = (ec, "catalogue") if (np.isfinite(ec) and ec > 0) else (rounding_unc(Pc), "rounding")
        frac = (r["P_spin (s)"] - Pc) / Pc
        srel = np.hypot(r["sigma P_spin (s)"] / r["P_spin (s)"], ec_used / Pc)
        rows.append(dict(source=n, catalogue_name=cn, obs_id=b["obs_id"], mid_mjd=tmid, P_app=r["P_spin (s)"],
                         sigma=r["sigma P_spin (s)"], P_cat=Pc, cat_unc=ec_used, cat_unc_source=ec_src,
                         frac_diff_pct=100 * frac, earth_pct=100 * uE, xmm_pct=100 * D_XMM, orbit_pct=100 * dorb,
                         vmax_kms=vmax, delta_max_pct=100 * dmax, sigma_rel_pct=100 * srel,
                         spin_change=bool(abs(frac) > dmax + 3 * srel)))
    df = pd.DataFrame(rows)
    df.to_csv(config.OUT / "xmm_timing_bounds.csv", index=False)
    print(df[["source", "frac_diff_pct", "delta_max_pct", "sigma_rel_pct", "spin_change"]].to_string(index=False))
    print("differences exceeding the timing bound:", int(df.spin_change.sum()), "of", len(df))


if __name__ == "__main__":
    main()
