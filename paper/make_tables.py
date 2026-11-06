"""Appendix tables A1 (Swift/BAT orbital periods) and A2 (XMM-Newton apparent spin periods) as
LaTeX longtables (tab_bat_full.tex, tab_xmm_full.tex in config.OUT). Run xmm_timing_bounds.py first."""
import math
import re

import numpy as np
import pandas as pd
from astropy.time import Time

import common
import config

OUT = str(config.OUT) + "/"
s = common.load_summary()
w = common.load_catalogue()
tb = pd.read_csv(config.OUT / "xmm_timing_bounds.csv").set_index("source")

def fnum(v):
    return np.format_float_positional(float(v), trim="-")

def compact(val, err, sig=2):
    """value(err) with err in units of the last digits (err rounded to sig significant figures)."""
    if val is None or not np.isfinite(val):
        return "--"
    if err is None or not np.isfinite(err) or err <= 0:
        return fnum(val)
    e = int(math.floor(math.log10(err)))
    lead = int(f"{err:e}"[0])
    sg = sig if sig else (2 if lead <= 2 else 1)
    dec = sg - 1 - e
    er = round(err, dec)
    if er >= 10 ** (e + 1):
        e += 1; dec = sg - 1 - e; er = round(err, dec)
    if dec > 0:
        vs = f"{val:.{dec}f}"
        es = f"{er:.{dec}f}"
        if er < 1:
            es = es.replace(".", "").lstrip("0")
        return f"{vs}({es})"
    return f"{val:.0f}({er:.0f})"

def cat_compact(val, err):
    """Catalogue value as given, error in units of its last decimal (rounded to an integer)."""
    if not np.isfinite(val):
        return "--"
    vs = fnum(val)
    if not (np.isfinite(err) and err > 0):
        return vs
    dec = len(vs.split(".")[1]) if "." in vs else 0
    eu = err * 10 ** dec
    while eu < 0.95:                   # value quoted with fewer digits than its error: pad with zeros
        dec += 1
        eu *= 10
        vs = f"{val:.{dec}f}"
    if eu >= 99.5:                     # error coarser than the quoted digits: round the value instead
        lead = int(f"{err:e}"[0])
        return compact(val, err, sig=2 if lead == 1 else 1)
    return f"{vs}({max(int(round(eu)), 1)})"

def tex_name(n):
    n = str(n).replace("-", "$-$", 1) if re.search(r"\d-\d", str(n)) else str(n)
    n = re.sub(r"(\d)-(\d)", r"\1$-$\2", str(n).replace("$-$", "-"))
    return n.replace("gam Cas", r"$\gamma$~Cas").replace("+", "$+$")

def tex_class(c):
    c = str(c)
    if c == "nan":
        return "--"
    c = c.replace("$\\gamma$\\,Be", "$\\gamma$").replace("$\\gamma$ Be", "$\\gamma$").replace("Be ?", "Be?")
    return c

def alias(folder, catname):
    if not isinstance(catname, str):
        return ""
    a, b = folder.replace(" ", "").lower(), catname.replace(" ", "").lower()
    if a == b or a.startswith(b) or b.startswith(a) or a.replace("gam", "") == b.replace("gam", ""):
        return ""
    return folder

kind, bat = {}, {}
for d in common.all_bat():
    bat[d["name"]] = d
    b = d.get("best") or {}
    kind[d["name"]] = b.get("kind") if b else None
xmm = {d["name"]: d for d in common.all_xmm()}

def ra_of(r):
    n = r["Catalogue name"]
    if isinstance(n, str) and n in w.index:
        return float(w.loc[n, "RAdeg"])
    return 16 * 15 + 31.6 / 60 * 15   # IGR J16316-4028 (from its name)
s["ra"] = s.apply(ra_of, axis=1)
s = s.sort_values("ra").reset_index(drop=True)

# ---------------------- notes per source (BAT) ----------------------
NB = {
 "V779 Cen": "a,b", "V1521 Cyg": "a", "BP Cru": "a", "OAO 1657-415": "a",
 "IGR J16418-4532": "b,f", "AX J1631.9-4752": "b", "AX J1820.5-1434": "c", "PSR J2032+4127": "d",
 "4U 2206+54": "e", "4U 1909+07": "f", "IGR J16493-4348": "f", "V662 Cas": "f", "V1343 Aql": "f",
 "GRO J1008-57": "g", "1E 1740.7-2942": "h", "GRO J1750-27": "h", "GRS 1758-258": "h,i", "KS 1947+300": "h",
 "V1357 Cyg": "h", "V635 Cas": "h", "GRO J2058+42": "i", "4U 1954+319": "j",
 "XTE J1829-098": "j", "V2246 Cyg": "k", "MAXI J0903-531": "l", "SAX J2103.5+4545": "l",
}
rows = []
for _, r in s.iterrows():
    n = r["Star (folder)"]
    cn = r["Catalogue name"] if isinstance(r["Catalogue name"], str) else n
    alt = alias(n, r["Catalogue name"])
    name = tex_name(cn) + (f" ({tex_name(alt)})" if alt else "")
    k = kind.get(n)
    P, sP = r["P_orb (d)"], r["sigma P_orb (d)"]
    d = bat[n]
    lc = ""
    if k:
        lc = "o" if (d["best"].get("lc_used") == "orbit") else "d"
    S = r["Strength S"]
    Pcat = r["Literature P_orb (d)"]
    ecat = w.loc[cn, "Period_err"] if cn in w.index else np.nan
    if k:
        Ss = f"{S:.0f}" if S >= 100 else f"{S:.1f}"
        typ = ("R" if n in common.REJECTED else "B") if k == "blind" else "T"
        hv = d["best"].get("halves_consistent")
        H = ("\\checkmark" if hv else "$\\times$") if k == "blind" else ""
        Pstr = compact(P, sP)
    else:
        kn = d.get("known") or {}
        Sk = kn.get("strength")
        Ss = f"({Sk:.1f})" if (Sk is not None and np.isfinite(Sk)) else ""
        typ, H, Pstr = "--", "", "--"
    ob = (d.get("best_outburst") or {})
    Pob = compact(ob["P"], ob["sigma_P"]) if ob.get("P") else ""
    if n == "PSR J2032+4127":
        Pstr = "\\textit{" + Pstr + "}"
    rows.append([name, tex_class(r["Class"]), Pstr, lc, Ss, typ, H, Pob, cat_compact(Pcat, ecat), NB.get(n, "")])

hdr = (r"\begin{longtable}{@{}l l l c r c c l l l@{}}" "\n"
       r"\caption{Swift/BAT orbital periods for the 102 sources, ordered by right ascension. Columns: catalogue name "
       r"(alternative name used in our files in parentheses), class (Fortin et al.\ 2023), adopted period with its formal "
       r"1$\sigma$ error in units of the last digits, light curve used (d = daily, o = orbit), strength $S$ (for sources "
       r"without a detection, $S$ at the catalogue period in parentheses), detection type (B = blind detection retained, "
       r"R = blind detection rejected, T = targeted check of the catalogue period), agreement of the periods measured in "
       r"the two halves of the data within three times their combined formal error (\checkmark) or not ($\times$; all "
       r"blind detections except GX~304$-$1 have $S \geq 6.64$ in both halves), period from outburst timing, catalogue "
       r"period, and notes.}\label{tab:bat}\\" "\n"
       r"\hline\noalign{\smallskip}" "\n"
       r"Source & Class & $P_{\rm orb}$ (d) & LC & $S$ & Det. & Halves & $P_{\rm ob}$ (d) & $P_{\rm cat}$ (d) & Note\\" "\n"
       r"\noalign{\smallskip}\hline\noalign{\smallskip}" "\n" r"\endfirsthead" "\n"
       r"\multicolumn{10}{l}{\tablename\ \thetable{} -- continued}\\\noalign{\smallskip}\hline\noalign{\smallskip}" "\n"
       r"Source & Class & $P_{\rm orb}$ (d) & LC & $S$ & Det. & Halves & $P_{\rm ob}$ (d) & $P_{\rm cat}$ (d) & Note\\" "\n"
       r"\noalign{\smallskip}\hline\noalign{\smallskip}" "\n" r"\endhead" "\n"
       r"\noalign{\smallskip}\hline" "\n" r"\endfoot" "\n")
notes_txt = (r"\noalign{\smallskip}\hline\noalign{\smallskip}" "\n"
 r"\multicolumn{10}{p{0.97\textwidth}}{\textit{Notes.} "
 r"(a) Documented orbital period evolution; the BAT value is the mean over 2005--2026 (Sect.~\ref{sec:pdot}, Table~\ref{tab:pdot}). "
 r"(b) Catalogue value superseded: our period agrees with later measurements (Sect.~\ref{sec:revised}). "
 r"(c) Modulation period; the orbital period may be this value or half of it (Sect.~\ref{sec:ax1820}, Fig.~\ref{fig:ax1820}). "
 r"(d) Rejected: the source is not detected on average and a 4.46-h period is incompatible with the decades-long orbit (Sect.~\ref{sec:spurious}); listed in italics, not counted among the detections. "
 r"(e) The family check favours $2P = 19.1254(18)$\,d, but the 19.1-d structure is not significant in the second half of the data (Sect.~\ref{sec:revised}). "
 r"(f) Additional superorbital or combination periods, Table~\ref{tab:other}. "
 r"(g) The strongest blind peaks are the 5th, 8th and 9th harmonics of the orbital period. "
 r"(h) Only signals within 3\% of 1 or 1/2\,d (instrumental) exceed the blind threshold. "
 r"(i) The maximum lies just outside the window around the catalogue period (GRO~J2058$+$42: 55.81\,d, $S = 10.8$; GRS~1758$-$258: 18.10\,d, $S = 17.6$). "
 r"(j) Only 5--30 cycles in the light curve; the formal error is a lower limit. "
 r"(k) Periodogram period of a Be transient; outburst phases drift, so it can differ from the pulse-timing orbit. "
 r"(l) A further peak above the blind threshold is not present in both halves of the data or is a near-orbital side peak (MAXI~J0903$-$531: 59.79\,d; SAX~J2103.5$+$4545: 12.32 and 12.44\,d).}\\" "\n")
with open(OUT + "tab_bat_full.tex", "w") as fh:
    fh.write(hdr)
    for rw in rows:
        fh.write(" & ".join(rw) + r" \\" + "\n")
    fh.write(notes_txt)
    fh.write(r"\end{longtable}" + "\n")
print("BAT rows", len(rows))

# ---------------------- XMM table ----------------------
NX = {"V779 Cen": "a", "4U 2206+54": "b", "IGR J00370+6122": "c", "IGR J16465-4507": "d",
      "AX J1838.0-0655": "e", "XTE J1829-098": "e", "IGR J19140+0951": "f", "V1343 Aql": "f",
      "XTE J1855-026": "g", "XTE J1858+034": "g", "IGR J19294+1816": "h", "IGR J18483-0311": "h", "BQ Cam": "i",
      "AX J1631.9-4752": "j", "V635 Cas": "j", "V884 Sco": "j"}


def fap_str(fap):
    if not np.isfinite(fap):
        return ""
    if fap < 1e-10:
        return r"$<10^{-10}$"
    if fap >= 1e-3:
        return f"{fap:.1g}"
    m, e = f"{fap:.0e}".split("e")
    return rf"${m}\times10^{{{int(e)}}}$"


xr = []
for _, r in s.iterrows():
    n = r["Star (folder)"]
    if n not in xmm:
        continue
    d = xmm[n]
    obs = d.get("observations", [])
    cn = r["Catalogue name"] if isinstance(r["Catalogue name"], str) else n
    alt = alias(n, r["Catalogue name"])
    name = tex_name(cn) + (f" ({tex_name(alt)})" if alt else "")
    Pc = r["Literature P_spin (s)"]
    ec = w.loc[cn, "Spin_period_err"] if cn in w.index else np.nan
    if np.isfinite(r["P_spin (s)"]):
        b = d["best"]
        o = next(o for o in obs if o["obs_id"] == b["obs_id"])
        date = Time(o["mjd_start"], format="mjd").iso[:10]
        T = f"{o['T'] / 1e3:.1f}"
        P = compact(r["P_spin (s)"], r["sigma P_spin (s)"])
        fs = fap_str(r["FAP (spin)"])
        typ = "B" if b.get("kind") == "blind" else "T"
        seg = r"$^{\rm s}$" if b.get("segment") else ""
        m = str(b.get("m", ""))
        Z = f"{b['Z']:.0f}" if b.get("Z") is not None and np.isfinite(b["Z"]) else ""
        t = tb.loc[n]
        dps = f"{t.frac_diff_pct:+.2f}".replace("-", "$-$") + (r"$^\dagger$" if t.spin_change else "")
        dmax = f"{t.delta_max_pct:.2f}"
        xr.append([name, b["obs_id"] + seg, date, T, P, f"{Z} ({m})", fs, typ, cat_compact(Pc, ec), dps, dmax, NX.get(n, ""), str(len(obs))])
    else:
        if not obs:
            xr.append([name, "--", "--", "--", "--", "", "", "--", cat_compact(Pc, ec), "--", "", "k", "0"])
            continue
        Ts = sum(o.get("T", 0) for o in obs) / 1e3
        dates = sorted(Time(o["mjd_start"], format="mjd").iso[:4] for o in obs if o.get("mjd_start"))
        dr = dates[0] if dates[0] == dates[-1] else f"{dates[0]}--{dates[-1]}"
        xr.append([name, f"{len(obs)} obs.", dr, f"{Ts:.1f}", "n.d.", "", "", "--", cat_compact(Pc, ec), "--", "", NX.get(n, ""), str(len(obs))])
cols = r"Source & ObsID & Date & $T$ (ks) & $P_{\rm app}$ (s) & $Z_m$ ($m$) & FAP & Det. & $P_{\rm cat}$ (s) & $\Delta$ (\%) & $\delta_{\max}$ (\%) & Note\\"
hdr = (r"\begin{longtable}{@{}l@{\hspace{5pt}\extracolsep{\fill}}l l r l r l c l r r l@{}}" "\n"
       r"\caption{XMM-Newton apparent spin periods. For detections: observation (superscript s: detected in a 10-ks segment), start date, "
       r"length of the light curve, apparent period $P_{\rm app}$ (spacecraft frame; not corrected for the motion of the "
       r"Earth, of the spacecraft or of the neutron star in its orbit) with its formal 1$\sigma$ error, value $Z_m$ of "
       r"the best harmonic sum and its number of harmonics $m$, false-alarm probability for all trials of that observation (analytic; "
       r"values below $10^{-10}$ are given as upper bounds, Sect.~\ref{sec:spin}), detection type (B = blind, T = targeted "
       r"check of the catalogue value), catalogue spin period, fractional difference $\Delta$ and the bound $\delta_{\max}$ "
       r"on the uncorrected timing effects [Eq.~(\ref{eq:dmax})]; $^\dagger$ marks differences larger than "
       r"$\delta_{\max}$ plus three times the combined error [Eq.~(\ref{eq:spinchange}), Sect.~\ref{sec:spinres}]. For non-detections: number of observations, years and total "
       r"length.}\label{tab:xmm}\\" "\n"
       r"\hline\noalign{\smallskip}" "\n" + cols + "\n"
       r"\noalign{\smallskip}\hline\noalign{\smallskip}" "\n" r"\endfirsthead" "\n"
       r"\multicolumn{12}{l}{\tablename\ \thetable{} -- continued}\\\noalign{\smallskip}\hline\noalign{\smallskip}" "\n"
       + cols + "\n"
       r"\noalign{\smallskip}\hline\noalign{\smallskip}" "\n" r"\endhead" "\n"
       r"\noalign{\smallskip}\hline" "\n" r"\endfoot" "\n")
notes_x = (r"\noalign{\smallskip}\hline\noalign{\smallskip}" "\n"
 r"\multicolumn{12}{p{0.97\textwidth}}{\textit{Notes.} "
 r"(a) Doppler-shifted by the 2.09-d orbit; corrected for the orbit and the Earth's motion the 2006 period is "
 r"4.80507(1)\,s (Sect.~\ref{sec:cenx3}, Fig.~\ref{fig:cenx3}). "
 r"(b) The catalogue value (392\,s) is not confirmed; the strongest peak, at 5757\,s (FAP = 0.018), corresponds to the "
 r"5593(10)-s pulsation found in the same data by Reig et al.\ (2012). "
 r"(c) The 674-s period was reported with a false-alarm probability of a few per cent (Uchida et al.\ 2021). "
 r"(d) Not recovered in the pipeline light curves of the two 2004 observations (6.6 and 12.6\,ks). "
 r"(e) Catalogue period outside the searchable range: the 70-ms period of AX~J1838.0$-$0655 cannot be sampled with "
 r"its 0.12-s pipeline bins, and the only usable light curve of XTE~J1829$-$098 (EPIC-MOS2, 2.6\,ks) has 10-s bins "
 r"(Sect.~\ref{sec:spin}). "
 r"(f) A weak candidate ($10^{-3} < $ FAP $< 10^{-2}$) does not pass the per-source threshold. "
 r"(g) Faint or eclipsed during the observation. "
 r"(h) Agrees with a measurement made in the same observation (Sect.~\ref{sec:spinres}). "
 r"(i) The catalogue value is rounded (4.4\,s). "
 r"(j) Weak candidates in one or more observations do not pass the per-source threshold. "
 r"(k) The pipeline products contain no light curve of the target.}\\" "\n")
with open(OUT + "tab_xmm_full.tex", "w") as fh:
    fh.write(hdr)
    for rw in xr:
        fh.write(" & ".join(rw[:12]) + r" \\" + "\n")
    fh.write(notes_x)
    fh.write(r"\end{longtable}" + "\n")
print("XMM rows", len(xr), "total obs", sum(int(r[12]) for r in xr))
