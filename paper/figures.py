"""Figures of the paper. Usage:  python figures.py [1 2 3 4 5 6]   (default: all)

  Fig. 1  comparison with the catalogue (orbital periods; apparent spin periods with timing bounds)
  Fig. 2  AX J1820.5-1434: periodogram and folded light curves          (needs the BAT light curve)
  Fig. 3  O-C diagrams of Cen X-3, Cyg X-3 and GX 301-2      (needs period_derivatives.json)
  Fig. 4  Cen X-3: folded BAT light curve and XMM segment periods       (needs light curve, cenx3_doppler.json)
  Fig. 5  half-to-half period differences of the blind detections
  Fig. 6  spin period against orbital period
Run after accounting.py, catalogue_comparison.py, xmm_timing_bounds.py, period_derivatives.py and
cenx3_doppler.py. Figures are written to config.OUT as PDF and PNG."""
import json
import math
import sys

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullFormatter

import config
import common
from figstyle import (COL, COL_W, C_BE, C_GRID, C_INK, C_INK2, C_SFXT, C_SG, FULL_W, MARK, cls_group, plt,
                      setup)

setup()
SHORT = {"V779 Cen": "Cen X-3", "V1521 Cyg": "Cyg X-3", "BP Cru": "GX 301-2", "V2246 Cyg": "EXO 2030+375",
         "V479 Sct": "LS 5039", "GP Vel": "Vela X-1", "V662 Cas": "2S 0114+650", "QV Nor": "4U 1538-52",
         "V884 Sco": "4U 1700-37", "H 1907+097": "4U 1907+09", "V725 Tau": "A 0535+26", "V850 Cen": "GX 304-1",
         "V635 Cas": "4U 0115+63", "BQ Cam": "V 0332+53", "V441 Pup": "3A 0726-260", "RX J1037.5-5647": "4U 1036-56",
         "IGR J17252-3616": "EXO 1722-363"}


def short(n):
    return SHORT.get(n, n)


def save(fig, stem):
    fig.savefig(config.OUT / f"{stem}.pdf")
    fig.savefig(config.OUT / f"{stem}.png", dpi=200)
    plt.close(fig)
    print("saved", config.OUT / f"{stem}.pdf")


def summary_with_kinds():
    s = common.load_summary()
    kind = {}
    for d in common.all_bat():
        b = d.get("best") or {}
        kind[d["name"]] = b.get("kind") if b else None
    s["kind"] = s["Star (folder)"].map(kind)
    s["grp"] = s["Class"].map(cls_group)
    return s


# ------------------------------------------------------------------------------------------------
def fig1():
    s = summary_with_kinds()
    cc = pd.read_csv(config.OUT / "catalogue_comparison.csv").set_index("Star (folder)")
    tb = pd.read_csv(config.OUT / "xmm_timing_bounds.csv").set_index("source")
    det = s[s["Star (folder)"].isin(cc.index)].copy()
    det["z"] = det["Star (folder)"].map(cc["z"])
    det["evol"] = det["Star (folder)"].isin(common.EVOLVING)
    det = det[det["z"].notna()]
    fig, (ax, a2) = plt.subplots(1, 2, figsize=(FULL_W, 2.7), gridspec_kw=dict(wspace=0.3))
    for g in ("Be", "sg", "SFXT", "other"):
        for k, fill in (("blind", True), ("targeted", False)):
            m = (det.grp == g) & (det.kind == k)
            if m.any():
                ax.scatter(det.loc[m, "Literature P_orb (d)"], det.loc[m, "z"], s=16, marker=MARK[g],
                           facecolors=COL[g] if fill else "none", edgecolors=COL[g], linewidths=0.8, zorder=3)
    ev = det[det.evol]
    ax.scatter(ev["Literature P_orb (d)"], ev["z"], s=70, marker="o", facecolors="none", edgecolors=C_INK,
               linewidths=0.6, zorder=4)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=3, linscale=1.3)
    ax.axhspan(-3, 3, color="#efeeea", zorder=0)
    for yv in (-3, 3):
        ax.axhline(yv, color=C_INK2, lw=0.5, ls=":")
    ax.axhline(0, color=C_INK2, lw=0.5)
    ax.set_xlabel("Catalogue orbital period (d)")
    ax.set_ylabel(r"$(P_{\rm BAT} - P_{\rm cat})/\sigma_{\rm comb}$ (formal)")
    txt_pos = {"V1521 Cyg": (0.45, 300, "left"), "AX J1820.5-1434": (12, 140, "right"),
               "AX J1631.9-4752": (0.45, 9, "left"), "V479 Sct": (0.45, 5.0, "left"),
               "IGR J16328-4726": (30, 20, "left"), "XTE J1829-098": (400, 9, "left"),
               "IGR J16418-4532": (0.45, -5.5, "left"), "OAO 1657-415": (0.45, -11, "left"),
               "V2246 Cyg": (150, -5.5, "left"), "BP Cru": (150, -14, "left"), "V779 Cen": (0.5, 0.9, "left")}
    for _, r in det.iterrows():
        n = r["Star (folder)"]
        if n in txt_pos:
            xt, yt, ha = txt_pos[n]
            ax.annotate(short(n), (r["Literature P_orb (d)"], r.z), xytext=(xt, yt), textcoords="data",
                        fontsize=6.0, ha=ha, va="center", color=C_INK,
                        arrowprops=dict(arrowstyle="-", lw=0.4, color=C_INK2, shrinkA=1, shrinkB=2))
    ax.set_xlim(0.12, 3000)
    ax.set_ylim(-30, 600)
    ax.set_yticks([-30, -10, -3, 0, 3, 10, 30, 100, 300])
    ax.set_yticklabels(["$-30$", "$-10$", "$-3$", "0", "3", "10", "30", "100", "300"])
    ax.set_title("(a) Orbital periods (Swift/BAT)", loc="left", fontsize=8)

    sp = s[s["P_spin (s)"].notna()].copy()
    sp["frac"] = sp["Star (folder)"].map(tb["frac_diff_pct"])
    sp["dmax"] = sp["Star (folder)"].map(tb["delta_max_pct"])
    sp["chg"] = sp["Star (folder)"].map(tb["spin_change"]).astype(bool)
    a2.axhspan(-sp["dmax"].max(), sp["dmax"].max(), color="#efeeea", zorder=0)
    for g in ("Be", "sg", "SFXT", "other"):
        for chg in (True, False):
            m = (sp.grp == g) & (sp.chg == chg)
            if m.any():
                a2.scatter(sp.loc[m, "Literature P_spin (s)"], sp.loc[m, "frac"], s=16, marker=MARK[g],
                           facecolors=COL[g] if chg else "white", edgecolors=COL[g], linewidths=0.8, zorder=3)
    a2.set_xscale("log")
    a2.axhline(0, color=C_INK2, lw=0.5)
    a2.set_xlabel("Catalogue spin period (s)")
    a2.set_ylabel(r"$(P_{\rm app} - P_{\rm cat})/P_{\rm cat}$ (%)")
    off2 = {"V662 Cas": (-5, 0, "right"), "MAXI J0655-013": (-5, 0, "right"), "IGR J16418-4532": (5, -1, "left"),
            "BP Cru": (6, -2, "left"), "AX J1820.5-1434": (-4, 5, "right"), "H 1907+097": (5, 5, "left"),
            "IGR J19294+1816": (5, 3, "left"), "BQ Cam": (5, -4, "left"), "IGR J17252-3616": (-5, -5, "right")}
    for _, r in sp.iterrows():
        if r["Star (folder)"] in off2:
            dx, dy, ha = off2[r["Star (folder)"]]
            a2.annotate(short(r["Star (folder)"]), (r["Literature P_spin (s)"], r.frac), xytext=(dx, dy),
                        textcoords="offset points", fontsize=6.0, ha=ha, va="center")
    a2.set_xlim(2, 2e4)
    a2.set_ylim(-9, 2.5)
    a2.set_title("(b) Apparent spin periods (XMM-Newton)", loc="left", fontsize=8)
    hand = [Line2D([], [], marker=MARK[g], ls="", color=COL[g], mfc=COL[g], ms=4.5, label=lab)
            for g, lab in (("Be", "Be"), ("sg", "supergiant"), ("SFXT", "SFXT"), ("other", "other/unclassified"))]
    fig.legend(handles=hand, loc="lower center", fontsize=6.5, ncol=4, handletextpad=0.2, columnspacing=1.4,
               bbox_to_anchor=(0.5, -0.04))
    ha = [Line2D([], [], marker="o", ls="", color=C_INK2, mfc=C_INK2, ms=4, label="blind detection"),
          Line2D([], [], marker="o", ls="", color=C_INK2, mfc="none", ms=4, label="targeted confirmation"),
          Line2D([], [], marker="o", ls="", color=C_INK, mfc="none", ms=7.5, mew=0.6,
                 label="documented $\\dot P_{\\rm orb}$")]
    ax.legend(handles=ha, loc="upper right", fontsize=5.8, handletextpad=0.3, borderaxespad=0.3)
    hb = [Line2D([], [], marker="o", ls="", color=C_INK2, mfc=C_INK2, ms=4, label="beyond timing bound"),
          Line2D([], [], marker="o", ls="", color=C_INK2, mfc="none", ms=4, label="within timing bound")]
    a2.legend(handles=hb, loc="lower left", fontsize=5.8, handletextpad=0.3, borderaxespad=0.3)
    fig.subplots_adjust(bottom=0.2)
    save(fig, "Fig1_comparison")


# ------------------------------------------------------------------------------------------------
def fig2():
    import bat_period_pipeline as bat
    js = common.bat_json("AX J1820.5-1434")
    dpath, _ = bat.find_bat_files(common.star_dir("AX J1820.5-1434"))
    lc = bat.load_bat_lc(dpath)
    pg = bat.compute_periodogram(lc, weighting="semi")
    P_best, thr = js["best"]["P"], js["threshold_daily"]
    sim99 = (js.get("simulation") or {}).get("threshold_99")
    per = 1.0 / pg.freq
    fig = plt.figure(figsize=(FULL_W, 2.25))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 1, 1], wspace=0.32)
    ax = fig.add_subplot(gs[0])
    m = (per > 15) & (per < 600)
    ax.plot(per[m], pg.strength[m], color=C_INK, lw=0.6)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([20, 50, 100, 200, 500]))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    h1 = ax.axhline(thr, color=C_SG, ls="--", lw=0.9, label="blind search, 1% (analytic)")
    h2 = ax.axhline(sim99, color=C_BE, ls="-.", lw=0.8, label="red-noise simulations, 99%") if sim99 else None
    h3 = ax.axhline(bat.threshold_median_norm(1, 0.01), color=C_INK2, ls=":", lw=0.9, label="single trial, 1%")
    S_best = pg.strength_at(P_best)[1]
    ax.annotate(f"{P_best:.1f} d", (P_best, S_best), xytext=(6, -2), textcoords="offset points", ha="left",
                va="top", fontsize=7)
    ax.axvline(54.0, color=C_SFXT, lw=0.9)
    ax.text(51.5, 21.8, "54.0 d\n(catalogue)", color=C_INK2, fontsize=6.5, ha="right", va="bottom")
    ax.set_xlim(15, 600)
    ax.set_ylim(0, 36)
    ax.legend(handles=[h for h in (h1, h2, h3) if h is not None], loc="upper right", fontsize=6.3,
              handlelength=2.2, borderaxespad=0.3)
    ax.set_xlabel("Period (d)")
    ax.set_ylabel("Strength  $S$ = power / local continuum")
    ax.text(0.02, 0.96, "(a)", transform=ax.transAxes, va="top", fontsize=8)
    dy, _ = bat.effective_errors(lc.rate, lc.err, "semi")
    w = 1.0 / dy ** 2
    for k, (P, lab) in enumerate(((P_best, rf"$P_{{\rm mod}}$ = {P_best:.2f} d (modulation)"),
                                  (54.0, "$P$ = 54.0 d (catalogue)"))):
        a = fig.add_subplot(gs[k + 1])
        pr = bat.fold_profile(lc.t, lc.rate, w, P, nbins=16, t0=53416.0, by_cycle=True)
        ph = np.concatenate([pr["phase"], pr["phase"] + 1])
        mu = np.concatenate([pr["mean"], pr["mean"]]) * 1e3
        er = np.concatenate([pr["err"], pr["err"]]) * 1e3
        a.errorbar(ph, mu, yerr=er, fmt="o", ms=3, color=C_BE if k == 0 else C_INK2, ecolor=C_INK2,
                   elinewidth=0.6, capsize=0, mew=0)
        a.set_xlim(0, 2)
        a.set_xlabel("Phase")
        if k == 0:
            a.set_ylabel(r"Rate ($10^{-3}$ counts cm$^{-2}$ s$^{-1}$)")
        a.set_title(lab, fontsize=7.5)
        a.text(0.03, 0.03, "(b)" if k == 0 else "(c)", transform=a.transAxes, va="bottom", fontsize=8)
    lims = [a.get_ylim() for a in fig.axes[1:]]
    for a in fig.axes[1:]:
        a.set_ylim(min(l[0] for l in lims), max(l[1] for l in lims))
    save(fig, "Fig2_AXJ1820")


# ------------------------------------------------------------------------------------------------
def _quad(t, y, s, c2_fixed=None):
    tau = np.asarray(t) - config.TREF
    W = 1 / np.asarray(s) ** 2
    if c2_fixed is None:
        A = np.column_stack([np.ones_like(tau), tau, tau ** 2])
        return np.linalg.solve(A.T @ (A * W[:, None]), A.T @ (W * y))
    A = np.column_stack([np.ones_like(tau), tau])
    c = np.linalg.solve(A.T @ (A * W[:, None]), A.T @ (W * (y - c2_fixed * tau ** 2)))
    return np.array([c[0], c[1], c2_fixed])


def fig3():
    R = json.load(open(config.OUT / "period_derivatives.json"))
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.35), gridspec_kw=dict(wspace=0.36))
    yrs = lambda mjd: 2005.0 + (np.asarray(mjd) - 53371.0) / 365.25   # noqa: E731
    for ax, (name, sc, unit) in zip(axes, (("Cen X-3", 1440.0, "min"), ("Cyg X-3", 1440.0, "min"),
                                           ("GX 301-2", 24.0, "h"))):
        r = R[name]
        t, y, s = (np.array(r["oc"][k]) for k in ("t", "oc", "soc"))
        s_plot = s * np.sqrt(max(r["oc"]["chi2r"], 1.0))
        c = _quad(t, y, s)
        tt = np.linspace(t.min() - 150, t.max() + 150, 300)
        tau = tt - config.TREF
        ax.errorbar(yrs(t), (y - c[0] - c[1] * (t - config.TREF)) * sc, yerr=s_plot * sc, fmt="o", ms=3.2,
                    color=C_BE, ecolor=C_INK2, elinewidth=0.6, mew=0, label="full-profile template", zorder=3)
        ax.plot(yrs(tt), c[2] * tau ** 2 * sc, color=C_BE, lw=1.0)
        c2l = r["lit_PdP_yr"] / 365.25 / 2.0
        cl = _quad(t, y, s_plot, c2_fixed=c2l)
        ax.plot(yrs(tt), (cl[0] - c[0] + (cl[1] - c[1]) * tau + c2l * tau ** 2) * sc, color=C_INK2, lw=0.9,
                ls="--", label=r"literature $\dot P/P$")
        if "eclipse" in r:
            e = r["eclipse"]
            te, ye, se = (np.array(e[k]) for k in ("t", "oc", "soc"))
            ce = _quad(te, ye, se)
            ax.errorbar(yrs(te) + 0.15, (ye - ce[0] - ce[1] * (te - config.TREF)) * sc, yerr=se * sc, fmt="s",
                        ms=3.0, color=C_SG, ecolor=C_INK2, elinewidth=0.6, mew=0, label="eclipse only", zorder=3)
            ax.plot(yrs(tt), ce[2] * tau ** 2 * sc, color=C_SG, lw=1.0)
        ax.set_title(name, fontsize=8)
        ax.set_xlabel("Year")
        ax.set_ylabel(f"O$-$C, linear term removed ({unit})")
        ax.legend(loc="upper center" if name == "Cyg X-3" else "lower center", fontsize=5.8, handlelength=1.8)
        ax.set_xlim(2004.5, 2026.5)
    save(fig, "Fig3_OC")


# ------------------------------------------------------------------------------------------------
def fig4():
    import bat_period_pipeline as bat
    import cenx3_doppler as cd
    D = json.load(open(config.OUT / "cenx3_doppler.json"))
    _, opath = bat.find_bat_files(common.star_dir("V779 Cen"))
    lc = bat.load_bat_lc(opath)
    dy, _ = bat.effective_errors(lc.rate, lc.err, "semi")
    w = 1.0 / dy ** 2
    cyc_f = cd.cycles(lc.t)
    ph, nb = cyc_f % 1.0, 40
    b = np.minimum((ph * nb).astype(int), nb - 1)
    uk, inv = np.unique(np.floor(cyc_f).astype(int) * nb + b, return_inverse=True)
    m_cb = np.bincount(inv, w * lc.rate) / np.bincount(inv, w)
    bb = (uk % nb).astype(int)
    n_b = np.bincount(bb, minlength=nb)
    mean = np.bincount(bb, m_cb, nb) / n_b
    err = np.sqrt(np.bincount(bb, (m_cb - mean[bb]) ** 2, nb) / (n_b - 1) / n_b)
    cen = (np.arange(nb) + 0.5) / nb
    fig = plt.figure(figsize=(FULL_W, 2.3))
    gs = fig.add_gridspec(1, 2, wspace=0.28)
    ax = fig.add_subplot(gs[0])
    x = np.concatenate([cen - 1, cen, cen + 1])
    y = np.concatenate([mean] * 3) * 1e3
    e = np.concatenate([err] * 3) * 1e3
    sel = (x >= -0.5) & (x <= 1.5)
    ax.errorbar(x[sel], y[sel], yerr=e[sel], fmt="o", ms=2.6, color=C_SG, ecolor=C_INK2, elinewidth=0.5, mew=0)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-1.5, 30)
    ax.set_xlabel("Orbital phase (0 = mid-eclipse)")
    ax.set_ylabel(r"BAT rate ($10^{-3}$ counts cm$^{-2}$ s$^{-1}$)")
    ax.text(0.02, 0.96, "(a)", transform=ax.transAxes, va="top", fontsize=8)
    ax.text(0.5, 0.96, "Swift/BAT orbit light curve, 2005–2026", transform=ax.transAxes, va="top",
            ha="center", fontsize=7, color=C_INK2)
    a2 = fig.add_subplot(gs[1])
    Porb_s = cd.P0 * 86400.0
    Ps, A = D["Ps_topo"], 2 * np.pi * D["asini"] / Porb_s
    dl = 2 * np.pi * D["dphase"]
    phx = np.linspace(0, 1, 400)
    lamx = 2 * np.pi * phx + np.pi / 2
    a2.plot(phx, Ps * (1 + A * np.cos(lamx + dl)), color=C_BE, lw=1.0,
            label=rf"Doppler fit: $a_x \sin i$ = {D['asini']:.2f} $\pm$ {D['asini_err']:.2f} lt-s")
    a2.plot(phx, Ps * (1 + 2 * np.pi * cd.ASINI / Porb_s * np.cos(lamx)), color=C_INK2, lw=0.8, ls="--",
            label="orbit of Falanga et al. (2015)")
    a2.axhline(Ps, color=C_GRID, lw=0.8, zorder=0)
    a2.errorbar(np.array(D["phases"]), np.array(D["periods"]), yerr=np.array(D["errors"]) * D["error_scale"],
                fmt="o", ms=3.4, color=C_SG, ecolor=C_INK2, mew=0, elinewidth=0.6, zorder=3,
                label="XMM-Newton 10-ks segments (2006)")
    a2.set_xlim(0, 1)
    a2.set_ylim(Ps * (1 - 1.9e-3), Ps * (1 + 1.6e-3))
    a2.set_xlabel("Orbital phase (0 = mid-eclipse)")
    a2.set_ylabel("Apparent spin period (s)")
    a2.legend(loc="lower right", fontsize=6.3, handlelength=2.0)
    a2.text(0.02, 0.96, "(b)", transform=a2.transAxes, va="top", fontsize=8)
    save(fig, "Fig4_CenX3")


# ------------------------------------------------------------------------------------------------
def halves_z():
    """z = (P2 - P1)/sigma_comb of the retained blind detections (split-data test of the pipeline)."""
    s = common.load_summary().set_index("Star (folder)")
    rows = []
    for d in common.all_bat():
        b = d.get("best") or {}
        if not b or b.get("kind") != "blind" or d["name"] in common.REJECTED:
            continue
        c = next(c for c in d["candidates"] if abs(c["P"] - b["P"]) < 1e-9 * b["P"])
        h = c["halves"]["halves"]
        rows.append(dict(name=d["name"], be=str(s.loc[d["name"], "Class"]).startswith("Be"),
                         evol=d["name"] in common.EVOLVING,
                         z=(h[1]["P"] - h[0]["P"]) / math.hypot(h[0]["sigma_P"], h[1]["sigma_P"])))
    return pd.DataFrame(rows)


def z_stats(z):
    z = np.asarray(z, float)
    return dict(N=int(z.size), median=float(np.median(z)), sd=float(np.std(z, ddof=1)), rms=float(np.sqrt(np.mean(z ** 2))),
                robust_sd=float(1.4826 * np.median(np.abs(z - np.median(z)))), n_gt3=int(np.sum(np.abs(z) > 3)),
                n_negative=int(np.sum(z < 0)))


def fig5():
    """Sect. 4.11: half-to-half differences of the 35 blind detections without documented period
    evolution (stacked: Be/X-ray binaries and other systems); the four systems with documented
    evolution are marked with arrows and are not in the histogram. Writes halves_stats.json."""
    from scipy.stats import binom
    df = halves_z()
    x = df[~df.evol]
    bins = np.arange(-7, 7.01, 1.0)
    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    ax.hist([np.clip(x[~x.be].z, -6.99, 6.99), np.clip(x[x.be].z, -6.99, 6.99)], bins=bins, stacked=True,
            color=[C_BE, C_SG], edgecolor="white", linewidth=0.8, label=["not Be", "Be/X-ray binaries"])
    xx = np.linspace(-7, 7, 300)
    for sd, ls, lab in ((1.0, "-", r"$\mathcal{N}(0,1)$"), (1.5, "--", r"$\mathcal{N}(0,1.5^2)$")):
        ax.plot(xx, len(x) * np.exp(-0.5 * (xx / sd) ** 2) / (sd * np.sqrt(2 * np.pi)),
                color=C_INK if ls == "-" else C_INK2, ls=ls, lw=0.9, label=lab)
    # documented period evolution: label positions (text x, text y, arrow tip y) in data units
    pos = {"V779 Cen": (-5.2, 3.4, 0.25), "BP Cru": (-4.0, 5.4, 2.25), "OAO 1657-415": (3.4, 7.6, 4.25),
           "V1521 Cyg": (5.1, 3.4, 0.25)}
    for _, r in df[df.evol].iterrows():
        tx, ty, tip = pos.get(r["name"], (np.clip(r.z, -6.8, 6.8), 3.0, 0.25))
        ax.annotate(f"{short(r['name'])} ({r.z:+.1f})".replace("-", "\u2212"), (np.clip(r.z, -6.8, 6.8), tip),
                    xytext=(tx, ty), textcoords="data", ha="center", va="bottom", fontsize=5.6,
                    arrowprops=dict(arrowstyle="-|>", lw=0.5, color=C_INK2, mutation_scale=5, shrinkA=1, shrinkB=0))
    ax.set_xlim(-7, 7)
    ax.set_xlabel(r"$(P_{\rm 2nd\,half} - P_{\rm 1st\,half})/\sigma_{\rm comb}$")
    ax.set_ylabel("Number of sources")
    ax.legend(loc="upper left", fontsize=5.8)
    save(fig, "Fig5_halves")
    out = {"all": z_stats(x.z), "not_Be": z_stats(x[~x.be].z), "Be": z_stats(x[x.be].z),
           "with_GX301_OAO1657": z_stats(df[~df.name.isin(["V779 Cen", "V1521 Cyg"])].z),
           "documented_evolution": {r["name"]: float(r.z) for _, r in df[df.evol].iterrows()},
           "gt3_sources": {r["name"]: float(r.z) for _, r in x[np.abs(x.z) > 3].iterrows()}}
    a = out["all"]
    k = min(a["n_negative"], a["N"] - a["n_negative"])
    out["all"]["sign_test_p_two_sided"] = float(min(1.0, 2 * binom.cdf(k, a["N"], 0.5)))
    common.dump(out, "halves_stats.json")
    for kk in ("all", "not_Be", "Be"):
        print(kk, {q: (round(v, 2) if isinstance(v, float) else v) for q, v in out[kk].items()})


# ------------------------------------------------------------------------------------------------
def fig6():
    s = summary_with_kinds()
    w = pd.read_excel(config.CATALOG)
    both = s[s["P_orb (d)"].notna() & s["P_spin (s)"].notna() & ~s["Star (folder)"].isin(common.REJECTED)].copy()
    cc = w[w["Period"].notna() & w["Spin_period"].notna() & ~w["Main_ID"].isin(set(both["Catalogue name"]))].copy()
    cc["grp"] = cc["Class"].map(cls_group)
    fig, ax = plt.subplots(figsize=(COL_W, 3.0))
    for g in ("Be", "sg", "SFXT", "other"):
        m = cc.grp == g
        ax.scatter(cc.loc[m, "Period"], cc.loc[m, "Spin_period"], s=12, marker=MARK[g], facecolors="none",
                   edgecolors=C_GRID, linewidths=0.7, zorder=1)
    for g in ("Be", "sg", "SFXT", "other"):
        m = both.grp == g
        ax.scatter(both.loc[m, "P_orb (d)"], both.loc[m, "P_spin (s)"], s=20, marker=MARK[g], color=COL[g],
                   edgecolors="white", linewidths=0.4, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Orbital period (d)")
    ax.set_ylabel("Spin period (s)")
    lab = {"V779 Cen": (3.2, 2.2), "V662 Cas": (20, 1.6e4), "IGR J19294+1816": (150, 5.0), "V2246 Cyg": (90, 22),
           "V635 Cas": (40, 2.2), "BP Cru": (22, 2500), "GP Vel": (2.2, 150), "IGR J18483-0311": (7, 9),
           "AX J1820.5-1434": (240, 150), "V725 Tau": (200, 62), "IGR J16418-4532": (1.6, 2500)}
    for _, r in both.iterrows():
        n = r["Star (folder)"]
        if n in lab:
            ax.annotate(short(n), (r["P_orb (d)"], r["P_spin (s)"]), xytext=lab[n], textcoords="data",
                        fontsize=5.8, ha="center", va="center",
                        arrowprops=dict(arrowstyle="-", lw=0.4, color=C_INK2, shrinkA=1, shrinkB=2.5))
    hand = [Line2D([], [], marker=MARK[g], ls="", color=COL[g], ms=4.5, label=lb) for g, lb in
            (("Be", "Be (this work)"), ("sg", "supergiant (this work)"), ("SFXT", "SFXT (this work)"))]
    hand.append(Line2D([], [], marker="o", ls="", mfc="none", mec=C_GRID, ms=4.5, label="catalogue values only"))
    ax.legend(handles=hand, loc="upper right", fontsize=5.8, handletextpad=0.2)
    ax.set_xlim(1, 400)
    ax.set_ylim(1, 3e4)
    save(fig, "Fig6_Corbet")
    print("both periods:", len(both))


if __name__ == "__main__":
    todo = [int(a) for a in sys.argv[1:]] or [1, 2, 3, 4, 5, 6]
    for k in todo:
        {1: fig1, 2: fig2, 3: fig3, 4: fig4, 5: fig5, 6: fig6}[k]()
