"""Table 5 of the paper (orbital period changes) from period_derivatives.json and
eclipse_sensitivity.json. Also compares the BAT period at the reference epoch MJD 57242, taken from
the linear term of the O-C fit, P(57242) = P_BAT (1 + c1), with the literature ephemerides
evaluated at the same epoch. Writes tab_pdot.tex and pdot_summary.json."""
import json

import numpy as np

import common
import config
import period_derivatives as pdv

REF = {"Cen X-3": (r"\citet{falanga2015}", "eclipses to 2011"),
       "Cyg X-3": (r"\citet{bhargava2017}", "minima 1979--2015"),
       "GX 301-2": (r"\citet{doroshenko2010}", "periastra to 2005"),
       "OAO 1657-415": (r"\citet{falanga2015}", "eclipses to 2011"),
       "Vela X-1": (r"\citet{falanga2015}", "eclipses to 2011"),
       "4U 1538-52": (r"\citet{falanga2015}", "eclipses to 2011")}
LIT = {"Cen X-3": "$-1.800\\pm0.001$", "Cyg X-3": "$+0.991\\pm0.004$", "GX 301-2": "$-33\\pm4$",
       "OAO 1657-415": "$-3.4\\pm0.1$", "Vela X-1": "$-0.1\\pm0.3$", "4U 1538-52": "$-0.7\\pm0.8$"}   # as published
TEXNAME = {"Cen X-3": "Cen~X-3", "Cyg X-3": "Cyg~X-3", "GX 301-2": "GX~301$-$2", "OAO 1657-415": "OAO~1657$-$415",
           "Vela X-1": "Vela~X-1", "4U 1538-52": "4U~1538$-$52"}


def ndec(e):
    """Decimals that give the error two significant digits."""
    if not np.isfinite(e) or e <= 0:
        return 2
    return max(0, 1 - int(np.floor(np.log10(e))))


def pm(v, e, scale=1e6):
    v, e = v * scale, e * scale
    d = ndec(e)
    return f"${v:+.{d}f}\\pm{e:.{d}f}$".replace("+-", "-")


def rng(a, b, scale=1e6, d=1):
    return f"$[{a * scale:+.{d}f},{b * scale:+.{d}f}]$"


def main():
    R = json.load(open(config.OUT / "period_derivatives.json"))
    E = json.load(open(config.OUT / "eclipse_sensitivity.json"))
    rows, summ = [], {}
    for name, r in R.items():
        lit = pdv.SRC[name][4]
        P = r["P"]
        Pe, ePe = pdv.ephemeris_period(lit, config.TREF)
        t, oc, soc = (np.array(r["oc"][k]) for k in ("t", "oc", "soc"))
        c, ec, _ = pdv.quad_fit(t, oc, soc)
        Poc, ePoc = P * (1 + c[1]), P * ec[1]
        zP = (Poc - Pe) / np.hypot(ePoc, ePe)
        t10, jk = r["tests"]["K=10"], r["tests"]["jackknife"]
        e = E.get(name, {})
        d = 1 if t10["e_PdP_yr"] > 1e-6 else 2
        row = dict(name=name, z_halves=r["halves_z"], PdP=t10["PdP_yr"], e_PdP=t10["e_PdP_yr"], chi2r=t10["chi2r"],
                   range=r["range_PdP_yr"], jackknife=jk["e_PdP_yr"], lit=r["lit_PdP_yr"], e_lit=r["lit_e_PdP_yr"],
                   P_eph=Pe, e_P_eph=ePe, P_oc=Poc, e_P_oc=ePoc, z_P=zP, t_eff=e.get("t_eff"),
                   two_interval=[(x["q"], x["PdP_yr"], x["e_PdP_yr"], x["z"]) for x in r["two_interval"]])
        cells = [TEXNAME[name], f"${r['halves_z']:+.1f}$".replace("+-", "-"), pm(t10["PdP_yr"], t10["e_PdP_yr"]),
                 rng(*r["range_PdP_yr"], d=d), f"{jk['e_PdP_yr'] * 1e6:.{d}f}"]
        if "eclipse" in r:
            te, ye, se = (np.array(r["eclipse"][k]) for k in ("t", "oc", "soc"))
            ce, ece, _ = pdv.quad_fit(te, ye, se)
            Pec, ePec = P * (1 + ce[1]), P * ece[1]
            row.update(PdP_ecl=r["eclipse"]["PdP_yr"], e_PdP_ecl=r["eclipse"]["e_PdP_yr"], ecl_range=e["eclipse_range"],
                       ecl_jackknife=e["eclipse_jackknife"]["e_PdP_yr"], P_ecl=Pec, e_P_ecl=ePec,
                       z_P_ecl=(Pec - Pe) / np.hypot(ePec, ePe))
            cells += [pm(r["eclipse"]["PdP_yr"], r["eclipse"]["e_PdP_yr"]), rng(*e["eclipse_range"], d=d)]
            zcell = f"${zP:+.1f}$ (${row['z_P_ecl']:+.1f}$)"
        else:
            cells += ["--", "--"]
            zcell = f"${zP:+.1f}$"
        cells += [LIT[name], zcell.replace("+-", "-")]
        rows.append(" & ".join(c.replace("+-", "-") for c in cells) + r" \\")
        summ[name] = row
    hdr = (r"\begin{table*}[t]" "\n"
           r"\caption{Orbital period changes from the BAT light curves. $z_{1/2}$: difference of the periods measured in "
           r"the two halves of the data in units of its formal error (a detection statistic, not a measurement of "
           r"$\dot P$). $\dot P/P$ (O$-$C): curvature of the O$-$C diagram of ten segments timed with the full-profile "
           r"template (errors scaled by $\sqrt{\chi^2_\nu}$ when $\chi^2_\nu>1$); range: all variants with 5--21 segments, "
           r"2--4 harmonics and division points at the 0.3--0.7 quantiles of the observation times; $\sigma_{\rm jk}$: "
           r"jackknife error over the ten segments. Eclipse only: timing of the eclipse within $\pm0.15$ in phase "
           r"($\pm0.20$ for Cen~X-3) and range over 7--14 segments and windows $\pm0.05$ around these values. "
           r"$z_P$: difference between the BAT period at MJD 57242 (from the linear O$-$C term; eclipse timing in "
           r"parentheses) and the literature ephemeris at the same epoch, in units of the combined formal error. "
           r"All $\dot P/P$ values in $10^{-6}$\,yr$^{-1}$. Literature values and ephemerides: \citet{falanga2015} for "
           r"Cen~X-3, OAO~1657$-$415, Vela~X-1 and 4U~1538$-$52 (eclipse times to 2011), \citet{bhargava2017} for Cyg~X-3 "
           r"(X-ray minima 1979--2015) and \citet{doroshenko2010} for GX~301$-$2 (periastron epochs to 2005).}\label{tab:pdot}" "\n"
           r"\centering\setlength{\tabcolsep}{3.8pt}" "\n"
           r"\begin{tabular}{@{}lrrcrrcrr@{}}" "\n" r"\toprule" "\n"
           r" & & \multicolumn{3}{c}{Full profile} & \multicolumn{2}{c}{Eclipse only} & & \\" "\n"
           r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}" "\n"
           r"Source & $z_{1/2}$ & $\dot P/P$ (O$-$C) & Range & $\sigma_{\rm jk}$ & $\dot P/P$ & Range & "
           r"$\dot P/P$ (lit.) & $z_P$\\" "\n" r"\midrule" "\n")
    with open(config.OUT / "tab_pdot.tex", "w") as fh:
        fh.write(hdr + "\n".join(rows) + "\n" + r"\bottomrule" "\n" r"\end{tabular}" "\n" r"\end{table*}" "\n")
    common.dump(summ, "pdot_summary.json")
    for n, r in summ.items():
        print(f"{n:12s} zP {r['z_P']:+.1f}" + (f" zP_ecl {r['z_P_ecl']:+.1f}" if "z_P_ecl" in r else "")
              + f"  P_oc {r['P_oc']:.9f}({r['e_P_oc']:.1e}) P_eph {r['P_eph']:.9f}({r['e_P_eph']:.1e})")


if __name__ == "__main__":
    main()
