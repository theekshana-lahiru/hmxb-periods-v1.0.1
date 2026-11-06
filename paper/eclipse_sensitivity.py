"""Sensitivity of the eclipse-only O-C timing (Sect. 4.5, Table 5) to the number of segments K,
to the phase window W around mid-eclipse and to single segments (jackknife), and the effective
epoch of each mean BAT period.

Effective epoch: for a period that changes linearly with time, the best-fitting constant period of
a light curve equals the instantaneous period at t_eff = tbar + mu3 / (2 mu2), where tbar, mu2 and
mu3 are the weighted mean, variance and third central moment of the observation times (weights of
the period fit). The literature ephemerides are evaluated at t_eff, not at the nominal middle of
the data. Writes eclipse_sensitivity.json."""
import warnings

import numpy as np

import config  # noqa: F401 (puts hmxb/ on sys.path)
import bat_period_pipeline as bat
import common
import period_derivatives as pdv

warnings.filterwarnings("ignore")


def effective_epoch(t, w):
    tb = np.sum(w * t) / np.sum(w)
    m2 = np.sum(w * (t - tb) ** 2) / np.sum(w)
    m3 = np.sum(w * (t - tb) ** 3) / np.sum(w)
    return float(tb), float(tb + 0.5 * m3 / m2)


def main():
    out = {}
    for name, (folder, kind, nh, W0, lit) in pdv.SRC.items():
        js = common.bat_json(folder)
        P = js["best"]["P"]
        dpath, opath = bat.find_bat_files(common.star_dir(folder))
        lc = bat.load_bat_lc(opath if kind == "orbit" else dpath)
        dy, _ = bat.effective_errors(lc.rate, lc.err, "semi")
        w = 1.0 / dy ** 2
        tb, teff = effective_epoch(lc.t, w)
        Pe, ePe = pdv.ephemeris_period(lit, teff)
        res = dict(t_mean=tb, t_eff=teff, P_eph_teff=Pe, e_P_eph_teff=ePe, P_bat=P, sigma_P_bat=js["best"]["sigma_P"])
        if W0 is not None:
            var = {}
            for K in (7, 10, 14):
                for W in (round(W0 - 0.05, 2), W0, round(W0 + 0.05, 2)):
                    try:
                        tk, oc, soc = pdv.oc_eclipse(lc.t, lc.rate, w, P, W, K, kind)
                        c, ec, chi2r = pdv.quad_fit(tk, oc, soc)
                        var[f"K={K},W={W:.2f}"] = dict(PdP_yr=2 * c[2] * 365.25, e_PdP_yr=2 * ec[2] * 365.25,
                                                       chi2r=chi2r, n=len(tk))
                    except Exception as ex:          # too few points in a narrow window
                        var[f"K={K},W={W:.2f}"] = dict(error=str(ex))
            tk, oc, soc = pdv.oc_eclipse(lc.t, lc.rate, w, P, W0, 10, kind)
            jk = np.array([2 * pdv.quad_fit(tk[np.arange(len(oc)) != i], oc[np.arange(len(oc)) != i],
                                            soc[np.arange(len(oc)) != i])[0][2] * 365.25 for i in range(len(oc))])
            res["eclipse_variants"] = var
            res["eclipse_jackknife"] = dict(PdP_yr=float(jk.mean()),
                                            e_PdP_yr=float(np.sqrt((len(jk) - 1) / len(jk) * np.sum((jk - jk.mean()) ** 2))))
            ok = [v["PdP_yr"] for v in var.values() if "PdP_yr" in v]
            res["eclipse_range"] = (float(min(ok)), float(max(ok)))
        out[name] = res
        line = f"{name:12s} t_eff {teff:.0f} (mean {tb:.0f})  P_eph(t_eff) {Pe:.9f} +/- {ePe:.1e}"
        if W0 is not None:
            line += (f" | eclipse range {res['eclipse_range'][0]*1e6:+.2f}..{res['eclipse_range'][1]*1e6:+.2f}"
                     f"  jackknife +/- {res['eclipse_jackknife']['e_PdP_yr']*1e6:.2f} (1e-6/yr)")
        print(line, flush=True)
    common.dump(out, "eclipse_sensitivity.json")


if __name__ == "__main__":
    main()
