"""Orbital period changes from the Swift/BAT light curves (Sect. 4.5, Table 4, Fig. 3).

1. Two-interval method: periods measured separately before and after a division point
   (quantiles 0.3-0.7 of the observation times), Pdot = dP / dt between the weighted mean times.
2. O-C timing: a template (constant + n harmonics) is fitted to the whole light curve at the mean
   period; in each of K segments of equal duration the phase shift of the template is fitted
   (A + B T(phi - delta)), O-C = delta * P, and a parabola O-C = c0 + c1 tau + c2 tau^2 gives
   Pdot/P = 2 c2 (errors scaled by sqrt(chi2_red) when chi2_red > 1). Sensitivity: K = 5-21,
   number of harmonics, jackknife over segments.
3. Eclipse-only timing for eclipsing systems: binned template fitted only within +/- W in phase
   of mid-eclipse.
Assumptions: constant Pdot over 2005-2026 and a modulation whose shape does not change; the
tests above quantify departures from the second assumption.
Writes period_derivatives.json.
"""
import warnings

import numpy as np

import config  # noqa: F401 (puts hmxb/ on sys.path)
import bat_period_pipeline as bat
import common
from hmxb_common import harmonic_period_fit

warnings.filterwarnings("ignore")
TREF = config.TREF
# folder, light curve, harmonics of the adopted fit, eclipse window (None = not eclipsing), literature
SRC = {
    "Cen X-3": ("V779 Cen", "orbit", 4, 0.20, dict(kind="frac", PdP=-1.800e-6, ePdP=0.001e-6, P0=2.08704106, eP0=3e-8,
                                                   T0=50506.788423, ref="Falanga et al. (2015)")),
    "Cyg X-3": ("V1521 Cyg", "orbit", 3, None, dict(kind="abs", P0=0.19968476, eP0=3e-8, T0=40949.384, Pdot=5.42e-10,
                                                    ePdot=0.02e-10, ref="Bhargava et al. (2017)")),
    "GX 301-2": ("BP Cru", "daily", 4, None, dict(kind="abs", P0=41.506, eP0=0.003, T0=43906.06, Pdot=-3.7e-6,
                                                  ePdot=0.5e-6, ref="Doroshenko et al. (2010)")),
    "OAO 1657-415": ("OAO 1657-415", "daily", 3, 0.15, dict(kind="frac", PdP=-3.4e-6, ePdP=0.1e-6, P0=10.447355,
                                                            eP0=9.2e-5, T0=52674.1199, ref="Falanga et al. (2015)")),
    "Vela X-1": ("GP Vel", "daily", 3, 0.15, dict(kind="frac", PdP=-0.1e-6, ePdP=0.3e-6, P0=8.964427, eP0=1.2e-5,
                                                  T0=42611.349, ref="Falanga et al. (2015)")),
    "4U 1538-52": ("QV Nor", "orbit", 4, 0.15, dict(kind="frac", PdP=-0.7e-6, ePdP=0.8e-6, P0=3.7284140, eP0=7.6e-6,
                                                    T0=52855.061, ref="Falanga et al. (2015)")),
}


def ephemeris_period(lit, t):
    """Period of a literature ephemeris at time t (MJD) and its error; the errors of P0 and of the
    derivative are added in quadrature (their covariance is not published)."""
    if lit["kind"] == "frac":
        dt = (t - lit["T0"]) / 365.25
        return lit["P0"] * (1 + lit["PdP"] * dt), float(np.hypot(lit["eP0"], lit["P0"] * lit["ePdP"] * abs(dt)))
    return lit["P0"] + lit["Pdot"] * (t - lit["T0"]), float(np.hypot(lit["eP0"], lit["ePdot"] * abs(t - lit["T0"])))


def design(phi, n):
    cols = [np.ones_like(phi)]
    for k in range(1, n + 1):
        cols += [np.cos(2 * np.pi * k * phi), np.sin(2 * np.pi * k * phi)]
    return np.column_stack(cols)


def template_coeffs(t, y, w, f, n):
    X = design(f * (t - TREF), n)
    return np.linalg.lstsq(X * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]


def shift_fit(t, y, w, model, grid):
    """Best phase shift of model(phase_shift) -> template values, with A + B*template fitted."""
    sw = np.sqrt(w)
    chi = np.empty(grid.size)
    for i, d in enumerate(grid):
        T = model(d)
        X = np.column_stack([np.ones_like(T), T])
        coef = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
        r = (y - X @ coef) * sw
        chi[i] = r @ r
    return chi


def best_shift(t, y, w, model, coarse, nfine=161):
    chi = shift_fit(t, y, w, model, coarse)
    j = int(np.argmin(chi))
    step = coarse[1] - coarse[0]
    fine = coarse[j] + np.linspace(-2 * step, 2 * step, nfine)
    chif = shift_fit(t, y, w, model, fine)
    k = int(np.argmin(chif))
    dof = max(len(y) - 3, 1)
    sc = max(chif[k] / dof, 1.0)
    dch = (chif - chif[k]) / sc
    L = np.where(dch[:k] >= 1.0)[0]
    R = np.where(dch[k:] >= 1.0)[0]
    if L.size and R.size:
        sig = 0.5 * (fine[k + R[0]] - fine[L[-1]])
    else:
        a = np.polyfit(fine - fine[k], dch, 2)[0]
        sig = 1 / np.sqrt(a) if a > 0 else np.nan
    return fine[k], sig


def unwrap(oc, P):
    out = np.array(oc, float)
    for i in range(1, len(out)):
        while out[i] - out[i - 1] > P / 2:
            out[i] -= P
        while out[i] - out[i - 1] < -P / 2:
            out[i] += P
    return out


def quad_fit(tk, oc, soc):
    tau = np.asarray(tk) - TREF
    W = 1 / np.asarray(soc) ** 2
    A = np.column_stack([np.ones_like(tau), tau, tau ** 2])
    cov = np.linalg.inv(A.T @ (A * W[:, None]))
    c = cov @ (A.T @ (W * oc))
    r = oc - A @ c
    dof = len(oc) - 3
    chi2r = float(np.sum(W * r ** 2) / dof) if dof > 0 else np.nan
    return c, np.sqrt(np.diag(cov) * max(chi2r, 1.0)), chi2r


def oc_template(t, y, w, P, n, K):
    f = 1.0 / P
    beta = template_coeffs(t, y, w, f, n)
    edges = np.linspace(t.min(), t.max() + 1e-6, K + 1)
    rows = []
    coarse = np.linspace(-0.5, 0.5, 720, endpoint=False)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (t >= a) & (t < b)
        if m.sum() < 30:
            continue
        phi = f * (t[m] - TREF)

        def model(d, phi=phi):
            return design(phi - d, n)[:, 1:] @ beta[1:]
        d, sd = best_shift(t[m], y[m], w[m], model, coarse)
        rows.append((np.sum(w[m] * t[m]) / np.sum(w[m]), d * P, sd * P))
    rows = np.array(rows)
    return rows[:, 0], unwrap(rows[:, 1], P), rows[:, 2]


def oc_eclipse(t, y, w, P, W, K=10, kind="orbit"):
    nb = 200 if kind == "orbit" else 50
    phi = ((t - TREF) / P) % 1.0
    b = np.minimum((phi * nb).astype(int), nb - 1)
    prof = np.bincount(b, w * y, nb) / np.bincount(b, w, nb)
    k = max(1, nb // 50)
    sm = np.convolve(np.r_[prof[-k:], prof, prof[:k]], np.ones(2 * k + 1) / (2 * k + 1), "valid")
    ph0 = (np.argmin(sm) + 0.5) / nb
    cen = (np.arange(nb) + 0.5) / nb
    xs, ys = np.r_[cen - 1, cen, cen + 1], np.r_[prof, prof, prof]
    edges = np.linspace(t.min(), t.max() + 1e-6, K + 1)
    rows = []
    for a, bnd in zip(edges[:-1], edges[1:]):
        dph = (((t - TREF) / P - ph0 + 0.5) % 1.0) - 0.5
        m = (t >= a) & (t < bnd) & (np.abs(dph) < W)
        if m.sum() < 20:
            continue
        ph = (t[m] - TREF) / P

        def model(d, ph=ph):
            return np.interp((ph - d) % 1.0, xs, ys)
        d, sd = best_shift(t[m], y[m], w[m], model, np.linspace(-0.04, 0.04, 161), nfine=161)
        rows.append((np.sum(w[m] * t[m]) / np.sum(w[m]), d * P, sd * P))
    rows = np.array(rows)
    ok = np.isfinite(rows[:, 2])
    return rows[ok, 0], rows[ok, 1], rows[ok, 2]


def two_interval(lc, P, nh, q):
    ts = np.quantile(lc.t, q)
    Ps, tm = [], []
    for m in (lc.t < ts, lc.t >= ts):
        sub = lc.subset(m)
        dys, _ = bat.effective_errors(sub.rate, sub.err, "semi")
        fit = harmonic_period_fit(sub.t, sub.rate, dys, 1.0 / P, nharm=nh, halfwidth=1.0 / sub.T)
        Ps.append((fit["P"], fit["sigma_P"]))
        wm = 1 / dys ** 2
        tm.append(np.sum(wm * sub.t) / np.sum(wm))
    dP = Ps[1][0] - Ps[0][0]
    e = np.hypot(Ps[0][1], Ps[1][1])
    dt = tm[1] - tm[0]
    return dict(q=q, P1=Ps[0][0], e1=Ps[0][1], P2=Ps[1][0], e2=Ps[1][1], t1=tm[0], t2=tm[1],
                PdP_yr=dP / dt / P * 365.25, e_PdP_yr=e / dt / P * 365.25, z=dP / e)


def main():
    results = {}
    for name, (folder, kind, nh, W, lit) in SRC.items():
        js = common.bat_json(folder)
        P, sP = js["best"]["P"], js["best"]["sigma_P"]
        dpath, opath = bat.find_bat_files(common.star_dir(folder))
        lc = bat.load_bat_lc(opath if kind == "orbit" else dpath)
        dy, _ = bat.effective_errors(lc.rate, lc.err, "semi")
        w = 1.0 / dy ** 2
        t, y = lc.t, lc.rate
        res = dict(P=P, sigma_P=sP, n_harm=nh, lc=kind, tests={}, lit_ref=lit["ref"])
        for K in (5, 7, 10, 14, 21):
            tk, oc, soc = oc_template(t, y, w, P, nh, K)
            c, ec, chi2r = quad_fit(tk, oc, soc)
            res["tests"][f"K={K}"] = dict(PdP_yr=2 * c[2] * 365.25, e_PdP_yr=2 * ec[2] * 365.25, chi2r=chi2r)
            if K == 10:
                res["oc"] = dict(t=tk, oc=oc, soc=soc, coef=c, ecoef=ec, chi2r=chi2r)
                jk = []
                for i in range(len(oc)):
                    m = np.arange(len(oc)) != i
                    jk.append(2 * quad_fit(tk[m], oc[m], soc[m])[0][2] * 365.25)
                jk = np.array(jk)
                res["tests"]["jackknife"] = dict(PdP_yr=float(jk.mean()),
                                                 e_PdP_yr=float(np.sqrt((len(jk) - 1) / len(jk) * np.sum((jk - jk.mean()) ** 2))))
        for n2 in sorted({1, 2} - {nh}):
            tk, oc, soc = oc_template(t, y, w, P, n2, 10)
            c, ec, chi2r = quad_fit(tk, oc, soc)
            res["tests"][f"n={n2}"] = dict(PdP_yr=2 * c[2] * 365.25, e_PdP_yr=2 * ec[2] * 365.25, chi2r=chi2r)
        res["two_interval"] = [two_interval(lc, P, nh, q) for q in (0.3, 0.4, 0.5, 0.6, 0.7)]
        if W is not None:
            tk, oc, soc = oc_eclipse(t, y, w, P, W, 10, kind)
            c, ec, chi2r = quad_fit(tk, oc, soc)
            res["eclipse"] = dict(t=tk, oc=oc, soc=soc, coef=c, PdP_yr=2 * c[2] * 365.25,
                                  e_PdP_yr=2 * ec[2] * 365.25, chi2r=chi2r, window=W)
        res["P_eph"], res["e_P_eph"] = ephemeris_period(lit, TREF)
        if lit["kind"] == "frac":
            res["lit_PdP_yr"], res["lit_e_PdP_yr"] = lit["PdP"], lit["ePdP"]
        else:
            res["lit_PdP_yr"] = lit["Pdot"] / res["P_eph"] * 365.25
            res["lit_e_PdP_yr"] = lit["ePdot"] / res["P_eph"] * 365.25
        hv = next(c for c in js["candidates"] if abs(c["P"] - P) < 1e-9 * P)["halves"]["halves"]
        res["halves_z"] = (hv[1]["P"] - hv[0]["P"]) / np.hypot(hv[0]["sigma_P"], hv[1]["sigma_P"])
        results[name] = res
        t10 = res["tests"]["K=10"]
        # sensitivity range: all well-constrained variants (error < 3x that of the K = 10 fit); the
        # single-sinusoid template (n = 1) cannot follow eclipses or flares and is not used
        e10 = t10["e_PdP_yr"]
        rng = [v["PdP_yr"] for k, v in res["tests"].items()
               if (k.startswith("K=") or (k.startswith("n=") and k != "n=1")) and v["e_PdP_yr"] < 3 * e10]
        rng += [x["PdP_yr"] for x in res["two_interval"]]
        res["range_PdP_yr"] = (float(min(rng)), float(max(rng)))
        print(f"{name:12s} halves z = {res['halves_z']:+.1f} | O-C K=10: {t10['PdP_yr']*1e6:+.2f} +/- {t10['e_PdP_yr']*1e6:.2f}"
              f" (chi2r {t10['chi2r']:.2f}) | range {res['range_PdP_yr'][0]*1e6:+.2f}..{res['range_PdP_yr'][1]*1e6:+.2f}"
              f" | jackknife +/- {res['tests']['jackknife']['e_PdP_yr']*1e6:.2f}"
              + (f" | eclipse {res['eclipse']['PdP_yr']*1e6:+.2f} +/- {res['eclipse']['e_PdP_yr']*1e6:.2f}" if 'eclipse' in res else "")
              + f" | lit {res['lit_PdP_yr']*1e6:+.3f} +/- {res['lit_e_PdP_yr']*1e6:.3f}  (1e-6/yr)", flush=True)
    common.dump(results, "period_derivatives.json")


if __name__ == "__main__":
    main()
