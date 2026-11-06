"""
hmxb_common.py
==============
Shared helpers for the HMXB timing pipeline (Swift/BAT orbits and XMM-Newton spins).

Contents
--------
* significance helpers for periodograms normalised by a local noise level
* harmonic (Fourier-series) period fit with a delta-chi-square = 1 error
* rounding a value to the decimal place of its error
* loading literature/reference periods (HMXBwebcat + your "my results.xlsx")
* small utilities (name normalisation, JSON-safe conversion)

Only numpy, scipy and pandas are needed here.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------------------
# Significance of a peak in a periodogram that has been divided by its local noise level
# --------------------------------------------------------------------------------------
# Under the null hypothesis (noise only) the periodogram power at one frequency is
# exponentially distributed around the local noise level.
#   * "median-normalised" strength S = power / local median  ->  P(S > s) = 2**(-s)
#   * "mean-normalised"   strength z = power / local mean    ->  P(z > s) = exp(-s)
# For a blind search over n independent frequencies the global false-alarm probability
# is 1 - (1 - p1)**n.  For a check of ONE period known in advance, n = 1.


def global_fap(p_single, n_trials):
    """Global false-alarm probability from a single-trial probability and n trials."""
    p1 = np.clip(np.asarray(p_single, float), 0.0, 1.0 - 1e-16)
    return -np.expm1(n_trials * np.log1p(-p1))


def fap_median_norm(strength, n_trials=1.0):
    """(single-trial p, global FAP) for strength = power / local MEDIAN."""
    p1 = np.power(2.0, -np.asarray(strength, float))
    return p1, global_fap(p1, n_trials)


def threshold_median_norm(n_trials, fap=0.01):
    """Strength (power / local median) needed for a given global FAP."""
    return float(-np.log2(-np.expm1(np.log1p(-fap) / max(n_trials, 1.0))))


def fap_mean_norm(z, n_trials=1.0):
    """(single-trial p, global FAP) for z = power / local MEAN (Israel & Stella 1996)."""
    p1 = np.exp(-np.asarray(z, float))
    return p1, global_fap(p1, n_trials)


def threshold_mean_norm(n_trials, fap=0.01):
    """z (power / local mean) needed for a given global FAP."""
    return float(-np.log(-np.expm1(np.log1p(-fap) / max(n_trials, 1.0))))


# --------------------------------------------------------------------------------------
# Harmonic (Fourier-series) period fit:  y = c + sum_h [a_h sin(2 pi h f t) + b_h cos(...)]
# --------------------------------------------------------------------------------------

def _design(t, f, nharm):
    """Columns 1, sin(2 pi h f t), cos(2 pi h f t) for h = 1..nharm (angle-addition recurrence)."""
    ph = 2.0 * np.pi * f * t
    s1, c1 = np.sin(ph), np.cos(ph)
    X = np.empty((t.size, 2 * nharm + 1))
    X[:, 0] = 1.0
    s, c = s1, c1
    X[:, 1], X[:, 2] = s, c
    for h in range(2, nharm + 1):
        s, c = s * c1 + c * s1, c * c1 - s * s1
        X[:, 2 * h - 1], X[:, 2 * h] = s, c
    return X


def _chi2_at(t, y, w, f, nharm, ywy=None):
    X = _design(t, f, nharm)
    Xw = X * w[:, None]
    A = X.T @ Xw
    b = Xw.T @ y
    try:
        beta = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(A, b, rcond=None)[0]
    if ywy is None:
        ywy = float(np.sum(w * y * y))
    return max(ywy - float(b @ beta), 0.0), beta


def _scan(t, y, w, freqs, nharm):
    ywy = float(np.sum(w * y * y))
    chi = np.empty(len(freqs))
    for i, f in enumerate(freqs):
        chi[i] = _chi2_at(t, y, w, f, nharm, ywy)[0]
    return chi


def _crossings(freqs, dchi, i_min, level=1.0):
    """Frequencies left/right of the minimum where dchi crosses `level` (linear interp.)."""
    lo = hi = np.nan
    for j in range(i_min, 0, -1):                      # walk left
        if dchi[j - 1] >= level >= dchi[j]:
            x0, x1, y0, y1 = freqs[j - 1], freqs[j], dchi[j - 1], dchi[j]
            lo = x0 + (level - y0) * (x1 - x0) / (y1 - y0) if y1 != y0 else x0
            break
    for j in range(i_min, len(freqs) - 1):             # walk right
        if dchi[j + 1] >= level >= dchi[j]:
            x0, x1, y0, y1 = freqs[j], freqs[j + 1], dchi[j], dchi[j + 1]
            hi = x0 + (level - y0) * (x1 - x0) / (y1 - y0) if y1 != y0 else x1
            break
    return lo, hi


def harmonic_period_fit(t, y, dy, f0, nharm="auto", nharm_choices=(2, 3, 4),
                        halfwidth=None, n_coarse=121, n_fine=121, min_scale=1.0):
    """Best frequency near f0 and its 1-sigma error from delta-chi-square = 1.

    The errors are rescaled so that the reduced chi-square at the minimum is 1
    (never scaled DOWN unless min_scale < 1).  `nharm="auto"` picks the number of
    harmonics from `nharm_choices` with the Bayesian information criterion.

    Returns a dict with f, sigma_f, P, sigma_P, nharm, chi2_red (the rescale factor),
    amp1 (semi-amplitude of the fundamental), mean, frac_amp (model (max-min)/(max+min)),
    edge (True if the minimum sits at the edge of the search window) and notes.
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    tc = t - t.mean()                       # centre time for numerical stability
    if dy is None:
        dy = np.full_like(y, np.std(y) if np.std(y) > 0 else 1.0)
    w = 1.0 / np.asarray(dy, float) ** 2
    T = t.max() - t.min()
    hw = (1.0 / T) if halfwidth is None else float(halfwidth)
    n = len(t)

    # 1) choose the number of harmonics (BIC at the starting frequency)
    nh_list = [int(nharm)] if nharm != "auto" else list(nharm_choices)
    nh_max = max(nh_list)
    if len(nh_list) > 1:
        chis = {k: _chi2_at(tc, y, w, f0, k)[0] for k in nh_list}
        scale0 = max(chis[nh_max] / max(n - (2 * nh_max + 2), 1), 1e-300)
        bic = {k: chis[k] / scale0 + (2 * k + 2) * math.log(n) for k in nh_list}
        nh = min(bic, key=bic.get)
    else:
        nh = nh_list[0]

    # 2) coarse scan with the chosen nharm
    fr = np.linspace(f0 - hw, f0 + hw, n_coarse)
    fr = fr[fr > 0]
    chi_c = _scan(tc, y, w, fr, nh)
    i = int(np.argmin(chi_c))
    edge = i in (0, len(fr) - 1)
    dof = max(n - (2 * nh + 2), 1)
    scale = max(chi_c[i] / dof, min_scale)
    d_c = (chi_c - chi_c[i]) / scale

    # 3) fine scan over the region with delta-chi2 <= 16 (4 sigma)
    step = fr[1] - fr[0] if len(fr) > 1 else hw / 10
    inside = np.where(d_c <= 16.0)[0]
    lo_i, hi_i = inside.min(), inside.max()
    f_lo = fr[max(lo_i - 1, 0)]
    f_hi = fr[min(hi_i + 1, len(fr) - 1)]
    if (f_hi - f_lo) < 6 * step:
        f_lo, f_hi = fr[i] - 3 * step, fr[i] + 3 * step
    ff = np.linspace(max(f_lo, 1e-12), f_hi, n_fine)
    chi_f = _scan(tc, y, w, ff, nh)
    j = int(np.argmin(chi_f))
    scale = max(chi_f[j] / dof, min_scale)
    d_f = (chi_f - chi_f[j]) / scale
    lo, hi = _crossings(ff, d_f, j, 1.0)
    notes = []
    if np.isfinite(lo) and np.isfinite(hi):
        sig_f = 0.5 * (hi - lo)
    elif np.isfinite(lo):
        sig_f = ff[j] - lo
        notes.append("one-sided error")
    elif np.isfinite(hi):
        sig_f = hi - ff[j]
        notes.append("one-sided error")
    else:
        sig_f = hw
        notes.append("error unconstrained (delta-chi2 never reached 1)")
    if edge:
        notes.append("minimum at edge of search window")

    f_best = float(ff[j])
    _, beta = _chi2_at(tc, y, w, f_best, nh)
    amp1 = float(np.hypot(beta[1], beta[2]))
    phase_grid = np.linspace(0, 1, 200, endpoint=False) / f_best
    model = _design(phase_grid, f_best, nh) @ beta
    mx, mn = float(model.max()), float(model.min())
    frac = (mx - mn) / (mx + mn) if (mx + mn) > 0 else np.nan
    return dict(f=f_best, sigma_f=float(sig_f), P=1.0 / f_best,
                sigma_P=float(sig_f) / f_best ** 2,
                P_lo=1.0 / hi if np.isfinite(hi) else np.nan,
                P_hi=1.0 / lo if np.isfinite(lo) and lo > 0 else np.nan,
                nharm=int(nh), chi2_red=float(chi_f[j] / dof), scale=float(scale),
                amp1=amp1, mean=float(beta[0]), frac_amp=frac, edge=bool(edge),
                n=int(n), notes="; ".join(notes))


# --------------------------------------------------------------------------------------
# Rounding
# --------------------------------------------------------------------------------------

def round_with_error(value, err, sig=2):
    """Round `err` to `sig` significant figures and `value` to the same decimal place.

    Returns (value_str, err_str, "value ± err").
    """
    if value is None or not np.isfinite(value):
        return "", "", ""
    if err is None or not np.isfinite(err) or err <= 0:
        s = f"{value:.6g}"
        return s, "", s
    exp = int(math.floor(math.log10(err)))
    dec = sig - 1 - exp
    err_r = round(err, dec)
    if err_r >= 10 ** (exp + 1):          # e.g. 0.0996 -> 0.10 changes the exponent
        exp += 1
        dec = sig - 1 - exp
        err_r = round(err, dec)
    val_r = round(value, dec)
    if dec > 0:
        vs, es = f"{val_r:.{dec}f}", f"{err_r:.{dec}f}"
    else:
        vs, es = f"{val_r:.0f}", f"{err_r:.0f}"
    return vs, es, f"{vs} ± {es}"


# --------------------------------------------------------------------------------------
# Names and reference periods
# --------------------------------------------------------------------------------------

def normalise_name(s):
    """Lower-case name without spaces, '+', '-', '.', '*' (for matching folder names)."""
    return re.sub(r"[\s\-\+\.\*_]", "", str(s)).lower()


# folder name -> catalogue Main_ID, for folders that match neither column of my results.xlsx
EXTRA_ALIASES = {"IGR J18410-0535": "AX J1841.0-0536",   # folder name -> catalogue Main_ID
                 "4U 2206+54": "4U 2206+543",
                 "1A 1118-61": "1A 1118-615"}


def _num(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


# folder name -> catalogue name table distributed with the code (data/source_names.csv)
NAMES_CSV = Path(__file__).resolve().parent.parent / "data" / "source_names.csv"


def load_references(catalog_xlsx, results_xlsx=None, stars_dir=None, extra_aliases=None, names_csv=None):
    """Literature periods and positions for every star folder.

    catalog_xlsx : HMXBwebcat_latest.xlsx (columns Main_ID, RAdeg, DEdeg, Class, Period,
                   Period_err, Superorbital Period, Spin_period, Spin_period_err ...)
    results_xlsx : your "my results.xlsx" (header on row 2: star, other name, ...); used to
                   map folder names such as "V779 Cen" to catalogue names such as "Cen X-3",
                   and to keep your previous values for comparison.
    stars_dir    : the folder that contains one sub-folder per star.
    names_csv    : table folder,catalogue_name,other_name (default data/source_names.csv of the
                   package); maps folder names to catalogue names when results_xlsx is absent.

    Returns a pandas DataFrame indexed by folder name.
    """
    import pandas as pd

    cat = pd.read_excel(catalog_xlsx)
    cat.columns = [str(c).strip() for c in cat.columns]
    cat["_key"] = cat["Main_ID"].map(normalise_name)
    mine = None
    if results_xlsx is not None and Path(results_xlsx).exists():
        mine = pd.read_excel(results_xlsx, header=1)
        mine.columns = [str(c).strip() for c in mine.columns]
        mine["_k1"] = mine["star"].map(normalise_name)
        mine["_k2"] = mine["other name"].map(normalise_name) if "other name" in mine else ""
    aliases = dict(EXTRA_ALIASES)
    other = {}
    nc = Path(names_csv) if names_csv is not None else NAMES_CSV
    if nc.exists():
        tab = pd.read_csv(nc, dtype=str).fillna("")
        aliases.update({r.folder: r.catalogue_name for r in tab.itertuples() if r.catalogue_name})
        other = {r.folder: r.other_name for r in tab.itertuples() if r.other_name}
    aliases.update(extra_aliases or {})
    if stars_dir is not None:
        folders = sorted(p.name for p in Path(stars_dir).iterdir() if p.is_dir())
    elif mine is not None:
        folders = list(mine["star"].astype(str))
    else:
        folders = list(cat["Main_ID"].astype(str))

    rows = {}
    for folder in folders:
        key = normalise_name(folder)
        main_id, mrow = aliases.get(folder), None
        if mine is not None:
            if main_id is not None:
                hit = mine[mine["star"] == main_id]
            else:
                hit = mine[(mine["_k1"] == key) | (mine["_k2"] == key)]
            if len(hit):
                mrow = hit.iloc[0]
                main_id = main_id or str(mrow["star"])
        if main_id is None:
            hit = cat[cat["_key"] == key]
            if len(hit):
                main_id = str(hit.iloc[0]["Main_ID"])
        crow = None
        if main_id is not None and (cat["Main_ID"] == main_id).any():
            crow = cat[cat["Main_ID"] == main_id].iloc[0]
        g = (lambda r, c: r[c] if (r is not None and c in r.index) else np.nan)
        rows[folder] = dict(
            folder=folder, main_id=main_id,
            other_name=(mrow["other name"] if mrow is not None and "other name" in mrow.index
                        and isinstance(mrow["other name"], str) else other.get(folder, "")),
            ra=_num(g(crow, "RAdeg")), dec=_num(g(crow, "DEdeg")),
            cls=g(crow, "Class") if crow is not None else "",
            P_orb_lit=_num(g(crow, "Period")), P_orb_lit_err=_num(g(crow, "Period_err")),
            P_super_lit=_num(g(crow, "Superorbital Period")),
            P_spin_lit=_num(g(crow, "Spin_period")), P_spin_lit_err=_num(g(crow, "Spin_period_err")),
            my_P_orb=g(mrow, "Orbital period"), my_P_super=g(mrow, "Superorbital Period"),
            my_P_spin=g(mrow, "spin period"),
        )
    return pd.DataFrame.from_dict(rows, orient="index")


# --------------------------------------------------------------------------------------
# JSON-safe conversion
# --------------------------------------------------------------------------------------

def to_jsonable(obj):
    """Convert numpy scalars/arrays (and nested containers) to plain Python types."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj
