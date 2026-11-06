r"""
bat_period_pipeline.py
======================
Orbital-period analysis of Swift/BAT Hard X-ray Transient Monitor light curves.

The six steps:
  1. Pick the light curve by period: the daily file for P above ~5 d, the orbit-resolution
     file (*.orbit.lc.fits) below that. Keep DATA_FLAG == 0, drop only points with absurd
     errors (outside 0.3-3 x the median error). Bright points are never clipped.
  2. Lomb-Scargle periodogram of the raw rates with errors. For bright, variable sources the
     intrinsic scatter is added to each error ("semi-weighting", Corbet et al. 2007, ApJ 655, 458).
     Search from 1/T to 0.5 c/d (daily) with 10 samples per peak width.
  3. Significance: divide by the local red-noise level (running median in log frequency;
     cf. Israel & Stella 1996, Vaughan 2005). Analytic false-alarm probability for the number
     of independent frequencies, or calibration with simulated red-noise light curves.
  4. Family check for every significant peak: P/2, P, 2P ("one structure per cycle"), the
     1-day alias 1/(1-f) (resolved with the orbit light curve), and the two halves of the data.
  5. Be transients with few outbursts: fit each outburst's peak time and a straight-line
     ephemeris T_n = T0 + n P.
  6. Measure: least-squares fit of a sinusoid plus harmonics to the raw data; errors rescaled so
     that the reduced chi-square is 1; 1-sigma error where delta-chi-square = 1. The period is
     rounded to the decimal place of its error.

Adopted period: the blind detection that matches a confirmed literature period; otherwise the
most significant independent blind detection that is also seen in both halves of the data and is
not a known systematic (within 3% of 1, 1/2, 1/3, 1/4 d or 1, 1/2 yr); otherwise the targeted
measurement of a confirmed literature period. Harmonics, beats, 1-day aliases and window-function
side-lobes of a stronger peak are labelled and never adopted.

Quick start (Jupyter):
    import bat_period_pipeline as bat
    res = bat.analyse_star(r"D:\HMXBs\Stars\IGR J00370+6122", known_period=15.6649)
    bat.print_summary(res)

Requires numpy, scipy, astropy, matplotlib, pandas; and hmxb_common.py in the same folder.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.timeseries import LombScargle
from scipy.optimize import brentq, curve_fit
from scipy.stats import chi2 as chi2_dist

from hmxb_common import (fap_median_norm, harmonic_period_fit, round_with_error,
                         threshold_median_norm, to_jsonable)

DAILY_FMAX = 0.5          # c/d, Nyquist frequency of daily bins
BLIND_FMAX_DAILY = 0.45   # blind peaks in daily data are taken below this (P > 2.2 d)
P_SWITCH = 5.0            # d; below this the orbit light curve is used
ORBIT_FMIN = 1.0 / P_SWITCH
ORBIT_FMAX = 6.0          # c/d (P > 4 h); raise it for shorter periods


# ======================================================================================
# Step 1: light curves
# ======================================================================================

@dataclass
class LightCurve:
    t: np.ndarray          # MJD, bin centre
    rate: np.ndarray       # counts cm^-2 s^-1 (15-50 keV)
    err: np.ndarray
    kind: str              # "daily" or "orbit"
    path: str = ""
    source: str = ""
    n_raw: int = 0
    n_kept: int = 0
    ra: float = np.nan
    dec: float = np.nan

    @property
    def T(self):
        return float(self.t[-1] - self.t[0]) if len(self.t) > 1 else 0.0

    def subset(self, mask):
        return replace(self, t=self.t[mask], rate=self.rate[mask], err=self.err[mask],
                       n_kept=int(np.sum(mask)))


def _is_bat(path):
    try:
        with fits.open(path) as h:
            hh = h[1].header
            return (str(hh.get("TELESCOP", "")).strip().upper() == "SWIFT"
                    and str(hh.get("INSTRUME", "")).strip().upper() == "BAT")
    except Exception:
        return False


def find_bat_files(star_dir):
    """(daily_path, orbit_path) of the BAT light curves anywhere below `star_dir`."""
    daily, orbit = [], []
    for p in sorted(Path(star_dir).rglob("*.fits")):
        n = p.name.lower()
        if n.endswith(".orbit.lc.fits"):
            orbit.append(p)
        elif n.endswith(".lc.fits"):
            daily.append(p)
    daily = [p for p in daily if _is_bat(p)]
    orbit = [p for p in orbit if _is_bat(p)]
    return (daily[0] if daily else None), (orbit[0] if orbit else None)


def load_bat_lc(path, err_range=(0.3, 3.0), keep_snr=5.0, dither_flags=None, min_pcodefr=None):
    """Load and clean a BAT daily or orbit light curve.

    * times are converted to MJD at the bin centre (orbit files are in Swift MET seconds)
    * kept: DATA_FLAG == 0, finite values
    * dropped as "absurd errors": error < err_range[0] x median, or error > err_range[1] x median
      while the point is NOT a significant detection (rate/error < keep_snr). Bright outburst
      days have naturally larger errors and are therefore kept: bright points are never removed.
    * optional: keep only some DITHER_FLAG values, or PCODEFR >= min_pcodefr (orbit files)
    """
    path = Path(path)
    with fits.open(path) as hdul:
        hdu = hdul["RATE"] if "RATE" in hdul else hdul[1]
        d, H = hdu.data, hdu.header
        names = [c.upper() for c in d.columns.names]
        time = np.asarray(d["TIME"], float)
        rate = np.asarray(d["RATE"], float)
        err = np.asarray(d["ERROR"], float)
        pixr = float(H.get("TIMEPIXR", 0.5))
        tunit = str(H.get("TIMEUNIT", H.get("TUNIT1", "d"))).strip().lower()
        if "TIMEDEL" in names or tunit.startswith("s"):
            kind = "orbit"
            tdel = np.asarray(d["TIMEDEL"], float) if "TIMEDEL" in names else np.zeros_like(time)
            mjdref = float(H.get("MJDREFI", 51910)) + float(H.get("MJDREFF", 7.428703703703703e-4))
            t = mjdref + (time + (0.5 - pixr) * tdel) / 86400.0
        else:
            kind = "daily"
            t = time + (0.5 - pixr) * float(H.get("TIMEDEL", 1.0))
        good = np.isfinite(t) & np.isfinite(rate) & np.isfinite(err) & (err > 0)
        if "DATA_FLAG" in names:
            good &= np.asarray(d["DATA_FLAG"]) == 0
        if dither_flags is not None and "DITHER_FLAG" in names:
            good &= np.isin(np.asarray(d["DITHER_FLAG"]), list(dither_flags))
        if min_pcodefr is not None and "PCODEFR" in names:
            good &= np.asarray(d["PCODEFR"], float) >= min_pcodefr
        if err_range is not None and good.sum() > 0:
            med = np.median(err[good])
            with np.errstate(invalid="ignore", divide="ignore"):
                significant = (rate / err) >= keep_snr
            good &= err >= err_range[0] * med
            good &= (err <= err_range[1] * med) | significant
        src = str(H.get("OBJECT", H.get("SRC_NAME", path.stem))).strip()
        ra = float(H.get("RA_OBJ", np.nan))
        dec = float(H.get("DEC_OBJ", np.nan))
    o = np.argsort(t[good])
    return LightCurve(t=t[good][o], rate=rate[good][o], err=err[good][o], kind=kind,
                      path=str(path), source=src, n_raw=len(time), n_kept=int(good.sum()),
                      ra=ra, dec=dec)


def choose_lc(P, daily, orbit, p_switch=P_SWITCH):
    """Daily light curve for P >= p_switch, orbit light curve below (when available)."""
    if P < p_switch and orbit is not None:
        return orbit
    return daily if daily is not None else orbit


# ======================================================================================
# Step 2: periodogram
# ======================================================================================

def effective_errors(rate, err, weighting="semi"):
    """Errors used as weights. Returns (dy or None, V_S).

    "semi"  : sqrt(err^2 + V_S), V_S = robust variance of the rates minus median err^2
    "full"  : err (pure 1/err^2 weighting)
    "none"  : None (unweighted)
    """
    if weighting in (None, "none", "unweighted"):
        return None, 0.0
    err = np.asarray(err, float)
    if weighting == "full":
        return err.copy(), 0.0
    if weighting == "semi":
        r = np.asarray(rate, float)
        mad = np.median(np.abs(r - np.median(r)))
        v_s = max((1.4826 * mad) ** 2 - np.median(err ** 2), 0.0)
        return np.sqrt(err ** 2 + v_s), float(v_s)
    raise ValueError("weighting must be 'semi', 'full' or 'none'")


@dataclass
class Periodogram:
    freq: np.ndarray        # c/d
    power: np.ndarray       # Lomb-Scargle power, "psd" normalisation
    continuum: np.ndarray   # local red-noise level (running median)
    strength: np.ndarray    # power / continuum
    T: float
    kind: str
    weighting: str
    v_s: float
    oversample: float
    n_trials: float         # independent frequencies in the blind range
    blind_fmax: float

    @property
    def fmin(self):
        return float(self.freq[0])

    @property
    def fmax(self):
        return float(self.freq[-1])

    def covers(self, P):
        return self.fmin <= 1.0 / P <= self.fmax

    def strength_at(self, P, halfwidth=None):
        """Highest strength within +/- halfwidth (default 0.5/T) of 1/P: (P_peak, S, power)."""
        f0 = 1.0 / P
        hw = 0.5 / self.T if halfwidth is None else halfwidth
        if f0 + hw < self.freq[0] or f0 - hw > self.freq[-1]:
            return np.nan, np.nan, np.nan
        idx = np.where((self.freq >= f0 - hw) & (self.freq <= f0 + hw))[0]
        if idx.size == 0:
            idx = np.array([int(np.argmin(np.abs(self.freq - f0)))])
        k = idx[int(np.argmax(self.strength[idx]))]
        return float(1.0 / self.freq[k]), float(self.strength[k]), float(self.power[k])


def local_continuum(freq, power, half_width_dex=0.15, n_anchor=600):
    """Running median of the power within +/- half_width_dex (log10 frequency)."""
    lf = np.log10(freq)
    anchors = np.linspace(lf[0], lf[-1], n_anchor)
    lo = np.searchsorted(lf, anchors - half_width_dex)
    hi = np.searchsorted(lf, anchors + half_width_dex)
    med = np.array([np.median(power[a:b]) if b > a else np.nan for a, b in zip(lo, hi)])
    ok = np.isfinite(med) & (med > 0)
    return np.interp(lf, anchors[ok], med[ok])


def compute_periodogram(lc, fmin=None, fmax=None, oversample=10, weighting="semi",
                        continuum_dex=0.15, blind_fmax=None):
    """Lomb-Scargle of the raw rates with (semi-)weighted errors, plus local strength."""
    T = lc.T
    if fmin is None:
        fmin = 1.0 / T if lc.kind == "daily" else ORBIT_FMIN
    if fmax is None:
        fmax = DAILY_FMAX if lc.kind == "daily" else ORBIT_FMAX
    freq = np.arange(fmin, fmax, 1.0 / (oversample * T))
    dy, v_s = effective_errors(lc.rate, lc.err, weighting)
    ls = LombScargle(lc.t, lc.rate, dy)
    method = "fast" if len(freq) > 2000 else "auto"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        power = ls.power(freq, method=method, assume_regular_frequency=(method == "fast"),
                         normalization="psd")
    power = np.clip(np.asarray(power, float), 0.0, None)
    cont = local_continuum(freq, power, continuum_dex)
    if blind_fmax is None:
        blind_fmax = BLIND_FMAX_DAILY if lc.kind == "daily" else fmax
    blind_fmax = min(blind_fmax, fmax)
    n_trials = max(T * (blind_fmax - fmin), 1.0)
    return Periodogram(freq=freq, power=power, continuum=cont, strength=power / cont, T=T,
                       kind=lc.kind, weighting=str(weighting), v_s=v_s, oversample=oversample,
                       n_trials=n_trials, blind_fmax=blind_fmax)


# ======================================================================================
# Step 3: significance
# ======================================================================================

def find_peaks(pg, n=8, min_strength=0.0, fmax=None, min_sep=None):
    """The n strongest local maxima of the strength below fmax (default the blind range)."""
    f, s = pg.freq, pg.strength
    fmax = pg.blind_fmax if fmax is None else fmax
    idx = np.where((s[1:-1] > s[:-2]) & (s[1:-1] >= s[2:]))[0] + 1
    idx = idx[(f[idx] <= fmax) & (s[idx] >= min_strength)]
    idx = idx[np.argsort(s[idx])[::-1]]
    sep = 3.0 / pg.T if min_sep is None else min_sep
    keep = []
    for i in idx:
        if all(abs(f[i] - f[j]) > sep for j in keep):
            keep.append(i)
            if len(keep) >= n:
                break
    out = []
    for i in keep:
        p1, fap = fap_median_norm(s[i], pg.n_trials)
        out.append(dict(P=float(1.0 / f[i]), f=float(f[i]), strength=float(s[i]),
                        power=float(pg.power[i]), p_single=float(p1), fap=float(fap)))
    return out


def fit_red_noise(pg, n_pts=300):
    """Fit continuum(f) = A f^-alpha + C in log space. Returns dict(logA, alpha, logC)."""
    lf = np.log10(pg.freq)
    sel = np.unique(np.searchsorted(lf, np.linspace(lf[0], lf[-1], n_pts)).clip(0, len(lf) - 1))
    x, ylog = lf[sel], np.log10(pg.continuum[sel])

    def model(x, logA, alpha, logC):
        return np.log10(10.0 ** (logA - alpha * x) + 10.0 ** logC)

    p0 = [ylog[0] + x[0], 1.0, ylog[-1]]
    try:
        popt, _ = curve_fit(model, x, ylog, p0=p0, maxfev=20000,
                            bounds=([-np.inf, 0.0, -np.inf], [np.inf, 4.0, np.inf]))
    except Exception:
        popt = [np.nan, 0.0, np.nan]
    return dict(logA=float(popt[0]), alpha=float(popt[1]), logC=float(popt[2]))


def _tk95(n, dt, alpha, rng):
    """Timmer & Koenig (1995) red noise with PSD ~ f^-alpha on a regular grid."""
    f = np.fft.rfftfreq(n, dt)[1:]
    amp = f ** (-alpha / 2.0)
    spec = np.zeros(f.size + 1, complex)
    spec[1:] = amp * (rng.normal(size=f.size) + 1j * rng.normal(size=f.size))
    if n % 2 == 0:
        spec[-1] = spec[-1].real
    return np.fft.irfft(spec, n)


def simulate_max_strength(lc, pg, n_sim=100, seed=0, oversize=4, dt=None, verbose=False):
    """Calibrate the blind threshold with simulated red-noise light curves.

    Each simulation: red noise with the fitted continuum slope (Timmer & Koenig 1995) and the
    observed excess variance, sampled at the observed times, plus Gaussian noise with the
    observed errors; analysed exactly like the data. Returns the maximum strength of each
    simulation (blind range) and the 95/99% thresholds. Outbursts are not Gaussian red noise,
    so for transients treat this as an approximation.
    """
    rng = np.random.default_rng(seed)
    alpha = fit_red_noise(pg)["alpha"]
    alpha = alpha if np.isfinite(alpha) else 0.0
    var_red = max(float(np.var(lc.rate) - np.mean(lc.err ** 2)), 0.0)
    dt = (1.0 if lc.kind == "daily" else 0.02) if dt is None else dt
    n_obs = int(np.ceil(lc.T / dt)) + 2
    n = int(2 ** np.ceil(np.log2(oversize * n_obs)))
    maxs = np.empty(n_sim)
    for k in range(n_sim):
        if var_red > 0:
            x = _tk95(n, dt, alpha, rng)
            start = int(rng.integers(0, n - n_obs))
            tg = lc.t[0] + (np.arange(n) - start) * dt
            xs = np.interp(lc.t, tg, x)
            sd = xs.std()
            xs = (xs - xs.mean()) / (sd if sd > 0 else 1.0) * np.sqrt(var_red)
        else:
            xs = np.zeros_like(lc.t)
        sim = replace(lc, rate=np.mean(lc.rate) + xs + rng.normal(0.0, lc.err))
        ps = compute_periodogram(sim, fmin=pg.fmin, fmax=pg.fmax + 0.5 / (pg.oversample * pg.T),
                                 oversample=pg.oversample, weighting=pg.weighting,
                                 blind_fmax=pg.blind_fmax)
        maxs[k] = ps.strength[ps.freq <= pg.blind_fmax].max()
        if verbose and (k + 1) % 20 == 0:
            print(f"  simulation {k + 1}/{n_sim}")
    return dict(max_strength=maxs, alpha=float(alpha), var_red=var_red,
                threshold_95=float(np.percentile(maxs, 95)),
                threshold_99=float(np.percentile(maxs, 99)))


def simulated_fap(strength, sims):
    m = sims["max_strength"]
    return float((1 + np.sum(m >= strength)) / (1 + len(m)))


# ======================================================================================
# Step 4: folding, family check, alias, halves of the data
# ======================================================================================

def fold_profile(t, y, w, P, nbins=16, t0=None, by_cycle=True):
    """Folded profile. With by_cycle=True each bin is the mean of per-cycle bin means and its
    error is the cycle-to-cycle scatter / sqrt(N cycles), which is robust to red noise."""
    t0 = np.min(t) if t0 is None else t0
    x = (t - t0) / P
    b = np.minimum(((x % 1.0) * nbins).astype(int), nbins - 1)
    centres = (np.arange(nbins) + 0.5) / nbins
    if not by_cycle:
        sw = np.bincount(b, w, nbins)
        swy = np.bincount(b, w * y, nbins)
        n_b = np.bincount(b, minlength=nbins)
        with np.errstate(invalid="ignore", divide="ignore"):
            return dict(phase=centres, mean=swy / sw, err=1.0 / np.sqrt(sw), n=n_b)
    cyc = np.floor(x).astype(np.int64)
    uk, inv = np.unique(cyc * nbins + b, return_inverse=True)
    m_cb = np.bincount(inv, w * y) / np.bincount(inv, w)
    bb = (uk % nbins).astype(int)
    n_b = np.bincount(bb, minlength=nbins)
    mean = np.bincount(bb, m_cb, nbins) / np.maximum(n_b, 1)
    var = np.bincount(bb, (m_cb - mean[bb]) ** 2, nbins) / np.maximum(n_b - 1, 1)
    err = np.sqrt(var / np.maximum(n_b, 1))
    mean = np.where(n_b > 0, mean, np.nan)
    err = np.where(n_b > 1, err, np.nan)
    return dict(phase=centres, mean=mean, err=err, n=n_b)


def fold_chi2(t, y, w, P, nbins=16):
    """Chi-square of the folded profile against a constant (cycle-scatter errors)."""
    pr = fold_profile(t, y, w, P, nbins)
    ok = np.isfinite(pr["mean"]) & np.isfinite(pr["err"]) & (pr["err"] > 0)
    if ok.sum() < 3:
        return np.nan, 0
    wb = 1.0 / pr["err"][ok] ** 2
    mu = np.sum(wb * pr["mean"][ok]) / np.sum(wb)
    return float(np.sum(wb * (pr["mean"][ok] - mu) ** 2)), int(ok.sum() - 1)


def halves_test(t, y, w, P, nbins=16):
    """Fold at 2P and compare its two halves (each one cycle of P).

    Small p means the halves differ, i.e. the true period is 2P rather than P."""
    pr = fold_profile(t, y, w, 2.0 * P, 2 * nbins)
    m1, m2 = pr["mean"][:nbins], pr["mean"][nbins:]
    e1, e2 = pr["err"][:nbins], pr["err"][nbins:]
    ok = np.isfinite(m1) & np.isfinite(m2) & np.isfinite(e1) & np.isfinite(e2) & (e1 > 0) & (e2 > 0)
    dof = int(ok.sum())
    if dof < 3:
        return dict(chi2=np.nan, dof=dof, p=np.nan)
    chi = float(np.sum((m1[ok] - m2[ok]) ** 2 / (e1[ok] ** 2 + e2[ok] ** 2)))
    return dict(chi2=chi, dof=dof, p=float(chi2_dist.sf(chi, dof)))


def family_check(lc, P0, weighting="semi", nbins=16, p_diff=0.01, p_same=0.05, max_steps=2,
                 pg=None, s_min_half=None, min_cycles=4):
    """Decide between P0/2, P0 and 2*P0 ("one structure per cycle").

    * halve while the P cycle is made of two statistically identical halves (p > p_same)
      and (if a periodogram is given) P/2 itself has a real peak (strength >= s_min_half)
    * double while the two halves of a 2P cycle differ (p < p_diff)
    """
    dy, _ = effective_errors(lc.rate, lc.err, weighting)
    w = 1.0 / dy ** 2 if dy is not None else np.ones_like(lc.rate)
    t, y = lc.t, lc.rate
    s_min_half = threshold_median_norm(1, 0.01) if s_min_half is None else s_min_half
    steps, P = [], float(P0)
    for _ in range(max_steps):
        h = halves_test(t, y, w, P / 2.0, nbins)            # fold at P, compare halves
        s_half = pg.strength_at(P / 2.0)[1] if (pg is not None and pg.covers(P / 2.0)) else np.nan
        steps.append(dict(test=f"is {P / 2:.6g} d enough?", candidate=P / 2.0, strength=s_half, **h))
        supported = (pg is None) or (np.isfinite(s_half) and s_half >= s_min_half)
        if np.isfinite(h["p"]) and h["p"] > p_same and supported:
            P /= 2.0
        else:
            break
    for _ in range(max_steps):
        if lc.T / (2.0 * P) < min_cycles:
            steps.append(dict(test=f"is {2 * P:.6g} d needed?", candidate=2 * P, chi2=np.nan,
                              dof=0, p=np.nan, note="too few cycles"))
            break
        h = halves_test(t, y, w, P, nbins)                  # fold at 2P, compare halves
        steps.append(dict(test=f"is {2 * P:.6g} d needed?", candidate=2.0 * P, **h))
        if np.isfinite(h["p"]) and h["p"] < p_diff:
            P *= 2.0
        else:
            break
    changed = not np.isclose(P, P0, rtol=1e-9)
    return dict(P_start=float(P0), P_chosen=float(P), steps=steps,
                note=(f"{P0:.6g} d -> {P:.6g} d (one structure per cycle)" if changed else "kept"))


def alias_check(P, orbit_lc, weighting="semi", npts=201):
    """Compare power at f = 1/P and at the 1-day alias |1 - f| in the orbit light curve,
    whose sampling is not tied to whole days. Daily data cannot separate the two."""
    f = 1.0 / P
    fa = abs(1.0 - f)
    Pa = 1.0 / fa if fa > 0 else np.inf
    if orbit_lc is None or fa <= 0:
        return dict(P=P, P_alias=Pa, preferred=P, ratio=np.nan, resolved=False,
                    note="no orbit light curve: 1-day alias not tested")
    dy, _ = effective_errors(orbit_lc.rate, orbit_lc.err, weighting)
    ls = LombScargle(orbit_lc.t, orbit_lc.rate, dy)
    hw = 0.5 / orbit_lc.T

    def pmax(fc):
        fr = np.linspace(fc - hw, fc + hw, npts)
        fr = fr[fr > 0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(np.max(ls.power(fr, normalization="psd")))

    p_true, p_alias = pmax(f), pmax(fa)
    ratio = p_true / p_alias if p_alias > 0 else np.inf
    preferred = P if ratio >= 1 else Pa
    resolved = ratio > 3 or ratio < 1 / 3
    note = (f"1-day alias {Pa:.6g} d rejected (power ratio {ratio:.3g})" if ratio >= 1
            else f"1-day alias {Pa:.6g} d preferred (power ratio {1 / ratio:.3g})")
    if not resolved:
        note += "; weak preference"
    return dict(P=P, P_alias=Pa, preferred=preferred, ratio=ratio, resolved=bool(resolved),
                power_true=p_true, power_alias=p_alias, note=note)


def measure_period(lc, P, weighting="semi", nharm="auto", halfwidth=None):
    """Step 6: harmonic fit around P with a delta-chi2 = 1 error, rounded."""
    dy, _ = effective_errors(lc.rate, lc.err, weighting)
    if dy is None:
        dy = lc.err
    res = harmonic_period_fit(lc.t, lc.rate, dy, 1.0 / P, nharm=nharm, halfwidth=halfwidth)
    res["P_str"] = round_with_error(res["P"], res["sigma_P"])[2]
    res["lc_kind"] = lc.kind
    return res


def data_halves_check(lc, P, weighting="semi", nharm="auto", fap=0.01):
    """Is the period present, with the same value, in both halves of the data?"""
    s_min = threshold_median_norm(1, fap)
    tm = np.median(lc.t)
    out = []
    for m in (lc.t < tm, lc.t >= tm):
        sub = lc.subset(m)
        if sub.n_kept < 50 or sub.T < 3.0 * P:
            out.append(dict(P=np.nan, sigma_P=np.nan, strength=np.nan, n=sub.n_kept,
                            note="too short"))
            continue
        f0 = 1.0 / P
        fmin = max(1.0 / sub.T, f0 / 2.5)
        fmax = f0 * 2.5 if lc.kind != "daily" else min(f0 * 2.5, DAILY_FMAX)
        pg = compute_periodogram(sub, fmin=fmin, fmax=fmax, weighting=weighting, blind_fmax=fmax)
        Pp, S, _ = pg.strength_at(P, halfwidth=1.0 / sub.T)
        meas = measure_period(sub, Pp if np.isfinite(Pp) else P, weighting=weighting,
                              nharm=nharm, halfwidth=1.0 / sub.T)
        out.append(dict(P=meas["P"], sigma_P=meas["sigma_P"], strength=S, n=sub.n_kept,
                        t_range=(float(sub.t[0]), float(sub.t[-1]))))
    ok = all(np.isfinite(o["P"]) for o in out)
    consistent = detected = False
    if ok:
        diff = abs(out[0]["P"] - out[1]["P"])
        sig = float(np.hypot(out[0]["sigma_P"], out[1]["sigma_P"]))
        consistent = diff <= 3.0 * sig
        detected = all(np.isfinite(o["strength"]) and o["strength"] >= s_min for o in out)
    return dict(halves=out, consistent=bool(consistent), detected_both=bool(detected),
                threshold=s_min)


# ======================================================================================
# Step 5: outburst timing (Be transients)
# ======================================================================================

def _gauss(x, c, A, mu, s):
    return c + A * np.exp(-0.5 * ((x - mu) / s) ** 2)


def fit_gaussian_peak(t, y, e, t_guess, width_guess, baseline=0.0):
    """Gaussian + constant fit to one outburst: peak time and its error."""
    span = max(float(t.max() - t.min()), 1.0)
    width_guess = float(np.clip(width_guess, 0.3, span))
    p0 = [baseline, max(float(np.max(y) - baseline), 1e-6), float(np.clip(t_guess, t.min(), t.max())),
          width_guess]
    lo = [-np.inf, 0.0, float(t.min()), 0.2]
    hi = [np.inf, np.inf, float(t.max()), span]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, pcov = curve_fit(_gauss, t, y, p0=p0, sigma=e, absolute_sigma=False,
                                   bounds=(lo, hi), maxfev=20000)
        perr = np.sqrt(np.diag(pcov))
        ok = bool(np.isfinite(perr[2]) and 0 < perr[2] < 0.5 * span)
        return dict(t_peak=float(popt[2]), t_peak_err=float(perr[2]), amp=float(popt[1]),
                    width=float(popt[3]), fit_ok=ok)
    except Exception:
        ww = np.clip(y - baseline, 0, None) / e ** 2
        tp = float(np.sum(ww * t) / np.sum(ww)) if ww.sum() > 0 else float(t_guess)
        return dict(t_peak=tp, t_peak_err=width_guess, amp=np.nan, width=np.nan, fit_ok=False)


def detect_outbursts(lc, smooth_days=3.0, n_sigma=5.0, merge_gap=5.0, min_points=3, min_rate=0.0):
    """Find outbursts in a daily light curve and fit each one's peak time.

    A day is "active" when its running weighted mean over `smooth_days` exceeds the median
    rate by n_sigma; active days closer than merge_gap are one outburst.
    Returns (list of outbursts, dict(duty=fraction of active days, baseline)).
    """
    t, y, e = lc.t, lc.rate, lc.err
    w = 1.0 / e ** 2
    h = smooth_days / 2.0
    lo = np.searchsorted(t, t - h, "left")
    hi = np.searchsorted(t, t + h, "right")
    cw = np.concatenate([[0.0], np.cumsum(w)])
    cwy = np.concatenate([[0.0], np.cumsum(w * y)])
    sw = cw[hi] - cw[lo]
    sm = (cwy[hi] - cwy[lo]) / sw
    se = 1.0 / np.sqrt(sw)
    base = float(np.median(y))
    active = ((sm - base) / se > n_sigma) & ((sm - base) > min_rate)
    idx = np.where(active)[0]
    info = dict(duty=float(active.mean()), baseline=base)
    if idx.size == 0:
        return [], info
    groups = np.split(idx, np.where(np.diff(t[idx]) > merge_gap)[0] + 1)
    out = []
    for g in groups:
        if g.size < min_points:
            continue
        t1, t2 = float(t[g[0]]), float(t[g[-1]])
        dur = t2 - t1
        pad = max(5.0, 0.5 * dur)
        m = (t >= t1 - pad) & (t <= t2 + pad)
        fit = fit_gaussian_peak(t[m], y[m], e[m], t_guess=float(t[g][np.argmax(sm[g])]),
                                width_guess=max(dur / 2.355, 1.0), baseline=base)
        out.append(dict(t_start=t1, t_end=t2, duration=dur, n_points=int(g.size),
                        peak_rate=float(np.max(y[g])), **fit))
    return out, info


def rayleigh_scan(times, Pmin, Pmax, oversample=10):
    """Rayleigh Z^2_1 of outburst peak times versus trial period."""
    times = np.asarray(times, float)
    span = max(times.max() - times.min(), Pmax)
    f = np.arange(1.0 / Pmax, 1.0 / Pmin, 1.0 / (oversample * span))
    ph = 2.0 * np.pi * np.outer(f, times)
    z2 = (2.0 / len(times)) * (np.cos(ph).sum(1) ** 2 + np.sin(ph).sum(1) ** 2)
    k = int(np.argmax(z2))
    return dict(freq=f, Z2=z2, P_best=float(1.0 / f[k]), Z2_best=float(z2[k]))


def _line_fit_with_jitter(tp, te, n):
    """Weighted fit tp = T0 + n P; jitter added in quadrature so that chi2/dof = 1."""
    A = np.column_stack([np.ones_like(tp), n.astype(float)])
    dof = len(tp) - 2

    def solve(jit):
        W = 1.0 / (te ** 2 + jit ** 2)
        cov = np.linalg.inv(A.T @ (A * W[:, None]))
        beta = cov @ (A.T @ (W * tp))
        r = tp - A @ beta
        return beta, cov, r, float(np.sum(W * r ** 2))

    jit = 0.0
    beta, cov, r, chi = solve(0.0)
    if dof > 0 and chi / dof > 1.0:
        fn = (lambda j: solve(j)[3] / dof - 1.0)
        top = max(np.ptp(tp), 1.0)
        while fn(top) > 0:
            top *= 2.0
        jit = brentq(fn, 0.0, top)
        beta, cov, r, chi = solve(jit)
    return beta, cov, r, chi, jit, dof


def fit_ephemeris(t_peak, t_err, P_guess, peak_rate=None, t_ref=None, clip_frac=0.2):
    """Weighted straight-line fit T_n = T0 + n P to outburst peak times.

    * cycle numbers come from P_guess; if two outbursts fall in one cycle the brighter is kept
    * an extra timing jitter is added in quadrature until the reduced chi-square is 1
      (outburst peaks scatter by days around periastron)
    * outbursts whose O-C exceeds clip_frac x P are removed one at a time (not orbit-locked)
    """
    tp = np.asarray(t_peak, float)
    te = np.asarray(t_err, float)
    pr = np.ones_like(tp) if peak_rate is None else np.asarray(peak_rate, float)
    o = np.argsort(tp)
    tp, te, pr = tp[o], te[o], pr[o]
    if t_ref is None:
        t_ref = tp[int(np.argmin(np.abs(tp - np.median(tp))))]
    n = np.round((tp - t_ref) / P_guess).astype(int)
    keep = np.ones(len(tp), bool)
    for c in np.unique(n):                         # one outburst per cycle: the brightest
        idx = np.where(n == c)[0]
        if idx.size > 1:
            keep[idx] = False
            keep[idx[np.argmax(pr[idx])]] = True
    n_same_cycle = int((~keep).sum())
    rejected = []
    while keep.sum() >= 3:
        beta, cov, r, chi, jit, dof = _line_fit_with_jitter(tp[keep], te[keep], n[keep])
        if dof < 1:
            break
        worst = int(np.argmax(np.abs(r)))
        if abs(r[worst]) > clip_frac * beta[1] and keep.sum() > 3:
            ii = np.where(keep)[0][worst]
            keep[ii] = False
            rejected.append(float(tp[ii]))
            continue
        break
    if keep.sum() < 3:
        return None
    beta, cov, r, chi, jit, dof = _line_fit_with_jitter(tp[keep], te[keep], n[keep])
    if dof < 1:
        return None
    T0, P = beta
    sT0, sP = np.sqrt(np.diag(cov))
    notes = []
    if n_same_cycle:
        notes.append(f"{n_same_cycle} weaker outburst(s) sharing a cycle dropped")
    if rejected:
        notes.append(f"{len(rejected)} outburst(s) off the ephemeris dropped")
    if np.max(np.abs(r)) > 0.3 * P:
        notes.append("O-C exceeds 0.3 P: cycle count may be wrong")
    return dict(P=float(P), sigma_P=float(sP), T0=float(T0), sigma_T0=float(sT0),
                jitter=float(jit), n_used=int(keep.sum()), cycles=n[keep], t_peak=tp[keep],
                oc=r, chi2_red=float(chi / dof), P_str=round_with_error(P, sP)[2],
                rejected=rejected, notes="; ".join(notes))


def outburst_timing(lc_daily, P_guess=None, Pmin=5.0, Pmax=None, exclude_long_frac=0.6,
                    outbursts=None, **detect_kw):
    """Detect outbursts, drop failed fits and very long (Type II) outbursts, fit an ephemeris."""
    if outbursts is None:
        obs, info = detect_outbursts(lc_daily, **detect_kw)
    else:
        obs, info = outbursts
    good = [o for o in obs if o["fit_ok"]]
    if P_guess is not None and exclude_long_frac:
        good = [o for o in good if o["duration"] <= exclude_long_frac * P_guess]
    res = dict(outbursts=obs, n_detected=len(obs), n_used=len(good), duty=info["duty"],
               ephemeris=None, scan=None, P_guess=P_guess, note="")
    if len(good) < 3:
        res["note"] = "fewer than 3 usable outbursts"
        return res
    tp = np.array([o["t_peak"] for o in good])
    te = np.array([o["t_peak_err"] for o in good])
    pr = np.array([o["peak_rate"] for o in good])
    if P_guess is None:
        if len(good) < 6:
            res["note"] = "fewer than 6 usable outbursts: no blind period scan"
            return res
        Pmax = Pmax or 0.25 * (tp.max() - tp.min())     # at least 4 cycles (activity epochs cluster)
        if Pmax <= Pmin:
            res["note"] = "outbursts span too short for a period scan"
            return res
        sc = rayleigh_scan(tp, Pmin, Pmax)
        # blind scan: Z^2_1 is chi2(2) under the null (p = exp(-Z2/2)); trials = independent
        # frequencies between 1/Pmax and 1/Pmin over the span of the outbursts
        n_tr = max(1.0, (tp.max() - tp.min()) * (1.0 / Pmin - 1.0 / Pmax))
        p1 = float(np.exp(-0.5 * sc["Z2_best"]))
        sc["fap"] = float(-np.expm1(n_tr * np.log1p(-min(p1, 1 - 1e-16))))
        sc["n_trials"] = n_tr
        res["scan"] = sc
        res["blind"] = True
        P_guess = sc["P_best"]
        res["P_guess"] = P_guess
    res["ephemeris"] = fit_ephemeris(tp, te, P_guess, peak_rate=pr)
    if res["ephemeris"] is None:
        res["note"] = "ephemeris fit failed (fewer than 3 outbursts on the ephemeris)"
    return res


# ======================================================================================
# Relations between periods
# ======================================================================================

def mark_harmonics(entries, T, kmax=10, near_res=5.0, lobe_res=25.0, lobe_ratio=0.1):
    """Label entries that are not independent of a MORE significant entry.

    Entries must be ordered from most to least significant. Labels (e["harmonic_of"]):
      ("P/k" or "k*P")      harmonic or sub-harmonic, k = 2..kmax (default 10)
      ("near")              within near_res frequency-resolution elements of a stronger peak
                            (drifting or changing period)
      ("1-yr side-lobe")    offset by 1/yr or 2/yr from a stronger peak (yearly window function)
      ("side-lobe")         within lobe_res resolution elements of a peak at least 1/lobe_ratio
                            times stronger (amplitude modulation, window function)
      ("f1+f2", "f1-f2", "2f1+f2", ...)  combination (beat) of two stronger periods
    """
    def indep(upto):
        return [b for b in entries[:upto] if not b.get("harmonic_of")]

    for i, e in enumerate(entries):
        fe, sfe = 1.0 / e["P"], e["sigma_P"] / e["P"] ** 2
        for b in indep(i):
            fb, sfb = 1.0 / b["P"], b["sigma_P"] / b["P"] ** 2
            df = abs(fe - fb)
            if df <= near_res / T:
                e["harmonic_of"] = (float(b["P"]), "near")
                break
            Se, Sb = e.get("strength", np.nan), b.get("strength", np.nan)
            weaker = np.isfinite(Se) and np.isfinite(Sb) and Se < Sb
            if weaker and min(abs(df - 1.0 / 365.25), abs(df - 2.0 / 365.25)) <= 1.5 / T:
                e["harmonic_of"] = (float(b["P"]), "1-yr side-lobe")
                break
            if weaker and df <= lobe_res / T and Se < lobe_ratio * Sb:
                e["harmonic_of"] = (float(b["P"]), "side-lobe")
                break
            for k in range(2, kmax + 1):
                if abs(fe - k * fb) <= max(3.0 * np.hypot(sfe, k * sfb), 1.0 / T):
                    e["harmonic_of"] = (float(b["P"]), f"P/{k}")
                    break
                if abs(k * fe - fb) <= max(3.0 * np.hypot(k * sfe, sfb), 1.0 / T):
                    e["harmonic_of"] = (float(b["P"]), f"{k}*P")
                    break
            if e.get("harmonic_of"):
                break
            for k in range(1, 4):                       # 1-day aliases of the period and its harmonics
                if abs(fe - abs(1.0 - k * fb)) <= max(3.0 * np.hypot(sfe, k * sfb), 1.0 / T):
                    e["harmonic_of"] = (float(b["P"]), "1-day alias of " + ("P" if k == 1 else f"P/{k}"))
                    break
            if e.get("harmonic_of"):
                break
        if e.get("harmonic_of"):
            continue
        base = indep(i)
        for a_i, a in enumerate(base):
            for b in base[a_i + 1:]:
                fa, fb = 1.0 / a["P"], 1.0 / b["P"]
                for m, n_, lab in ((1, 1, "f1+f2"), (1, -1, "f1-f2"), (2, 1, "2f1+f2"), (2, -1, "2f1-f2"),
                                   (1, 2, "f1+2f2"), (1, -2, "f1-2f2")):
                    fc = abs(m * fa + n_ * fb)
                    if fc > 0 and abs(fe - fc) <= 1.0 / T:
                        e["harmonic_of"] = (float(a["P"]), f"{lab} with f2 = 1/{b['P']:.5g} d")
                        break
                if e.get("harmonic_of"):
                    break
            if e.get("harmonic_of"):
                break
    return entries


INSTRUMENTAL = ((1.0, "1 day"), (0.5, "1/2 day"), (1.0 / 3.0, "1/3 day"), (0.25, "1/4 day"),
                (365.25, "1 year"), (182.625, "1/2 year"))


def instrumental_flag(P, tol=0.03):
    """Name of a known systematic period within tol (fractional) of P, else ''.

    BAT survey light curves carry signals tied to the daily observing pattern (strongest for
    sources near the Galactic centre) and to the yearly Sun constraint."""
    for Pi, lab in INSTRUMENTAL:
        if abs(P - Pi) <= tol * Pi:
            return f"within {tol:.0%} of {lab}: probably instrumental"
    return ""


def relation_to(P, sP, P_ref, sP_ref, T):
    """Describe how P relates to a reference period (same, 2x, 1/2x, 3x, 1/3x, 1-day alias)."""
    if not (np.isfinite(P) and np.isfinite(P_ref) and P_ref > 0):
        return ""
    sP = sP if np.isfinite(sP) else 0.0
    sP_ref = sP_ref if np.isfinite(sP_ref) else 0.0
    for k, lab in ((1.0, "same"), (2.0, "2 x literature"), (0.5, "1/2 x literature"),
                   (3.0, "3 x literature"), (1.0 / 3.0, "1/3 x literature")):
        tol = max(3.0 * np.hypot(sP, k * sP_ref), P ** 2 / T)
        if abs(P - k * P_ref) <= tol:
            return lab
    fa = abs(1.0 - 1.0 / P_ref)
    if fa > 0 and abs(P - 1.0 / fa) <= max(3.0 * sP, P ** 2 / T):
        return "1-day alias of literature"
    return "different"


# ======================================================================================
# The whole analysis for one star
# ======================================================================================

def analyse_star(star_dir, name=None, known_period=None, known_period_err=None,
                 weighting="semi", use_orbit=True, search_orbit=True, orbit_fmax=ORBIT_FMAX,
                 fap_level=0.01, n_sim=0, n_sim_orbit=0, transient="auto", nharm="auto",
                 max_candidates=5, out_dir=None, show=True, verbose=True):
    """Run steps 1-6 for one star folder.

    known_period : literature orbital period (d) to test as a targeted check (optional)
    n_sim        : number of red-noise simulations to calibrate the daily blind threshold
                   (0 = analytic false-alarm probability only)
    n_sim_orbit  : the same for the orbit-light-curve search (slow: ~4 s per simulation)
    transient    : "auto" (decide from the light curve), True or False
    out_dir      : if given, a summary PNG and a JSON file are saved there
    """
    star_dir = Path(star_dir)
    name = name or star_dir.name
    daily_path, orbit_path = find_bat_files(star_dir)
    if daily_path is None and orbit_path is None:
        raise FileNotFoundError(f"No Swift/BAT light curve below {star_dir}")
    daily = load_bat_lc(daily_path) if daily_path else None
    orbit = load_bat_lc(orbit_path) if (orbit_path and use_orbit) else None
    T = (daily or orbit).T
    res = dict(name=name, folder=str(star_dir), source=(daily or orbit).source,
               daily_file=str(daily_path or ""), orbit_file=str(orbit_path or ""),
               n_daily=daily.n_kept if daily else 0, n_orbit=orbit.n_kept if orbit else 0,
               T_days=T, weighting=weighting, fap_level=fap_level, notes=[])

    # ---- steps 2-3: blind searches -------------------------------------------------------
    pg_d = compute_periodogram(daily, weighting=weighting) if daily else None
    pg_o = None
    if orbit is not None and search_orbit:
        pg_o = compute_periodogram(orbit, fmin=ORBIT_FMIN, fmax=orbit_fmax, weighting=weighting)
    res["v_s"] = pg_d.v_s if pg_d else (pg_o.v_s if pg_o else np.nan)
    thr_d = threshold_median_norm(pg_d.n_trials, fap_level) if pg_d else np.nan
    thr_o = threshold_median_norm(pg_o.n_trials, fap_level) if pg_o else np.nan
    peaks_d = find_peaks(pg_d, n=8) if pg_d else []
    peaks_o = find_peaks(pg_o, n=8) if pg_o else []
    sims = sims_o = None
    if n_sim and pg_d is not None:
        sims = simulate_max_strength(daily, pg_d, n_sim=n_sim, verbose=verbose)
        for p in peaks_d:
            p["fap_sim"] = simulated_fap(p["strength"], sims)
    if n_sim_orbit and pg_o is not None:
        sims_o = simulate_max_strength(orbit, pg_o, n_sim=n_sim_orbit, verbose=verbose)
        for p in peaks_o:
            p["fap_sim"] = simulated_fap(p["strength"], sims_o)
    res.update(threshold_daily=thr_d, threshold_orbit=thr_o, peaks_daily=peaks_d,
               peaks_orbit=peaks_o, threshold_targeted=threshold_median_norm(1, fap_level),
               n_trials_daily=pg_d.n_trials if pg_d else np.nan,
               n_trials_orbit=pg_o.n_trials if pg_o else np.nan)
    if sims is not None:
        res["simulation"] = dict(n_sim=n_sim, alpha=sims["alpha"], threshold_95=sims["threshold_95"],
                                 threshold_99=sims["threshold_99"])
    if sims_o is not None:
        res["simulation_orbit"] = dict(n_sim=n_sim_orbit, alpha=sims_o["alpha"],
                                       threshold_95=sims_o["threshold_95"], threshold_99=sims_o["threshold_99"])

    cands, failed = [], []
    for search, pk, thr in (("daily", peaks_d, thr_d), ("orbit", peaks_o, thr_o)):
        for p in pk:
            if "fap_sim" in p:
                if p["fap_sim"] < fap_level:
                    cands.append(dict(p, search=search))
                elif p["strength"] >= thr:
                    failed.append(dict(P=p["P"], strength=p["strength"], fap_sim=p["fap_sim"], search=search))
            elif p["strength"] >= thr:
                cands.append(dict(p, search=search))
    res["failed_simulation"] = failed
    cands.sort(key=lambda c: (c.get("fap_sim", c["fap"]), -c["strength"]))

    # ---- steps 4 and 6 for each significant peak -----------------------------------------
    found = []
    for c in cands[:max_candidates]:
        P0 = c["P"]
        lc_c = choose_lc(P0, daily, orbit)
        pg_c = pg_o if (lc_c is orbit and pg_o is not None) else pg_d
        fam = family_check(lc_c, P0, weighting, pg=pg_c)
        P1 = fam["P_chosen"]
        al = None
        if lc_c.kind == "daily" and orbit is not None:
            al = alias_check(P1, orbit, weighting)
            if al["preferred"] != P1:
                P1 = al["preferred"]
        lc_m = choose_lc(P1, daily, orbit)
        meas = measure_period(lc_m, P1, weighting, nharm)
        halves = data_halves_check(lc_m, meas["P"], weighting, nharm, fap_level)
        pg_m = pg_o if (pg_o is not None and lc_m is orbit and pg_o.covers(meas["P"])) else pg_d
        S_m = pg_m.strength_at(meas["P"])[1] if (pg_m is not None and pg_m.covers(meas["P"])) else np.nan
        fap_m = float(fap_median_norm(S_m, pg_m.n_trials)[1]) if np.isfinite(S_m) else np.nan
        entry = dict(P_peak=P0, search=c["search"], strength_peak=c["strength"], fap_peak=c["fap"],
                     fap_sim=c.get("fap_sim", np.nan), family=fam, alias=al, measure=meas,
                     halves=halves, P=meas["P"], sigma_P=meas["sigma_P"], P_str=meas["P_str"],
                     strength=S_m, fap=fap_m, lc_used=lc_m.kind)
        tol = lambda e1, e2: max(3.0 * np.hypot(e1["sigma_P"], e2["sigma_P"]), 0.5 * e2["P"] ** 2 / T)
        dup = next((e for e in found if abs(e["P"] - entry["P"]) <= tol(e, entry)), None)
        if dup is not None:
            dup.setdefault("also_from_peaks", []).append(float(P0))
        else:
            found.append(entry)
    mark_harmonics(found, T)
    for e in found:
        e["instrumental"] = instrumental_flag(e["P"])
    res["candidates"] = found

    # ---- targeted check of the literature period -----------------------------------------
    known = None
    if known_period is not None and np.isfinite(known_period) and known_period > T:
        known = dict(P_lit=float(known_period), P_lit_err=known_period_err, P_peak=np.nan,
                     strength=np.nan, n_trials=np.nan, fap=np.nan, confirmed=False, lc_used="",
                     window=np.nan, note="longer than the light curve: cannot be tested")
    elif known_period is not None and np.isfinite(known_period) and known_period > 0:
        lc_k = choose_lc(known_period, daily, orbit)
        pg_k = None
        if lc_k is orbit and pg_o is not None and pg_o.covers(known_period):
            pg_k = pg_o
        elif lc_k is daily and pg_d is not None and pg_d.covers(known_period):
            pg_k = pg_d
        if pg_k is None:
            f0 = 1.0 / known_period
            fmax_k = f0 * 2.5 if lc_k.kind != "daily" else min(f0 * 2.5, DAILY_FMAX)
            pg_k = compute_periodogram(lc_k, fmin=max(1.0 / lc_k.T, f0 / 2.5), fmax=fmax_k,
                                       weighting=weighting, blind_fmax=fmax_k)
        sf_lit = (known_period_err / known_period ** 2
                  if known_period_err is not None and np.isfinite(known_period_err) else 0.0)
        hw = max(0.5 / pg_k.T, 3.0 * sf_lit)
        Pk, Sk, _ = pg_k.strength_at(known_period, halfwidth=hw)
        n_k = max(1.0, 2.0 * hw * pg_k.T)
        fap_k = float(fap_median_norm(Sk, n_k)[1]) if np.isfinite(Sk) else np.nan
        confirmed = bool(np.isfinite(fap_k) and fap_k < fap_level)
        note = ""
        # a maximum on the edge of the window may be the wing of a stronger peak just outside it
        Pk2, Sk2, _ = pg_k.strength_at(known_period, halfwidth=hw + 2.0 / pg_k.T)
        if (confirmed and np.isfinite(Sk2) and Sk2 > 1.05 * Sk
                and abs(1.0 / Pk2 - 1.0 / known_period) > hw + 0.5 / pg_k.T):
            confirmed = False
            note = (f"the peak is at {Pk2:.6g} d (S = {Sk2:.1f}), just outside the literature window "
                    f"(+/- {'3 sigma' if 3.0 * sf_lit > 0.5 / pg_k.T else 'half a resolution element'})")
        known = dict(P_lit=float(known_period), P_lit_err=known_period_err, P_peak=Pk,
                     strength=Sk, n_trials=n_k, fap=fap_k, confirmed=confirmed, lc_used=lc_k.kind,
                     window=hw, note=note)
        if confirmed:
            known["family"] = family_check(lc_k, Pk, weighting, pg=pg_k)
            known["measure"] = measure_period(lc_k, Pk, weighting, nharm)
    res["known"] = known

    # ---- choose the headline result --------------------------------------------------------
    # Adopted: the blind detection that matches a confirmed literature period; otherwise the most
    # significant independent, non-instrumental blind detection seen in BOTH halves of the data;
    # otherwise the targeted measurement of a confirmed literature period.
    best = None
    independent = [e for e in found if not e.get("harmonic_of")]
    lit_err = (known["P_lit_err"] if known and known["P_lit_err"] is not None else np.nan)

    def matches_lit(e):
        if not (known and known["confirmed"]):
            return False
        df = abs(1.0 / e["P"] - 1.0 / known["P_lit"])
        return (df <= known["window"] + 1.0 / T
                or relation_to(e["P"], e["sigma_P"], known["P_lit"], lit_err, T) == "same")

    usable = [e for e in independent if not e["instrumental"] or matches_lit(e)]
    qualified = [e for e in usable if e["halves"]["detected_both"]]
    strongest = (min(qualified, key=lambda x: (x["fap"] if np.isfinite(x["fap"]) else 1.0, -x["strength"]))
                 if qualified else None)
    pick = strongest
    if known and known["confirmed"]:
        match = next((e for e in usable if matches_lit(e)), None)
        pick = match
        if strongest is not None and match is not None and strongest is not match:
            res["notes"].append(f"strongest blind period {strongest['P_str']} d is not the literature orbit")
    if pick is not None:
        e = pick
        method = f"BAT {e['lc_used']} LC, weighted LS + {e['measure']['nharm']}-harmonic fit"
        if np.isfinite(e.get("fap_sim", np.nan)):
            method += f" (red-noise simulations: FAP {e['fap_sim']:.2g})"
        best = dict(P=e["P"], sigma_P=e["sigma_P"], P_str=e["P_str"], strength=e["strength"],
                    fap=e["fap"], fap_sim=e.get("fap_sim", np.nan), method=method, lc_used=e["lc_used"],
                    kind="blind", family_note=e["family"]["note"],
                    alias_note=e["alias"]["note"] if e["alias"] else "",
                    halves_consistent=e["halves"]["consistent"],
                    halves_detected=e["halves"]["detected_both"])
    elif known and known["confirmed"]:
        m = known["measure"]
        best = dict(P=m["P"], sigma_P=m["sigma_P"], P_str=m["P_str"], strength=known["strength"],
                    fap=known["fap"], fap_sim=np.nan, kind="targeted",
                    method=f"BAT {known['lc_used']} LC, targeted check of literature period + "
                           f"{m['nharm']}-harmonic fit",
                    lc_used=known["lc_used"], family_note=known["family"]["note"], alias_note="",
                    halves_consistent=np.nan, halves_detected=np.nan)
    res["best"] = best
    for e in independent:
        if e is pick:
            continue
        why = []
        if e["instrumental"] and not matches_lit(e):
            why.append(e["instrumental"])
        if not e["halves"]["detected_both"]:
            why.append("not detected in both halves of the data")
        if why:
            fam = (f"; family check of the {e['P_peak']:.5g} d peak" if e["family"]["note"] != "kept" else "")
            res["notes"].append(f"{e['P_str']} d (S = {e['strength']:.1f}{fam}) not adopted: " + "; ".join(why))
    for p in res.get("failed_simulation", []):
        res["notes"].append(f"{p['P']:.6g} d ({p['search']}, S = {p['strength']:.1f}) passed the analytic "
                            f"threshold but not the red-noise simulations (FAP_sim = {p['fap_sim']:.2g})")

    # ---- step 5: outburst timing (after the headline period, which anchors the cycle count) ----
    ob = None
    if daily is not None and transient is not False:
        obs, info = detect_outbursts(daily)
        is_tr = (transient is True) or (len(obs) >= 3 and info["duty"] < 0.25)
        res["outburst_duty"] = info["duty"]
        res["n_outbursts"] = len(obs)
        if is_tr:
            P_guess = best["P"] if (best is not None and best["P"] >= 10.0) else None
            ob = outburst_timing(daily, P_guess=P_guess, outbursts=(obs, info))
    res["outburst_timing"] = ob
    eph = ob["ephemeris"] if ob else None
    if eph is not None and ob.get("blind") and ob["scan"]["fap"] >= fap_level:
        # no period to anchor the cycle count and the outburst times alone are not periodic
        res["notes"].append(f"outburst times show no significant period (blind Rayleigh scan: best "
                            f"{ob['scan']['P_best']:.4g} d, FAP {ob['scan']['fap']:.2g})")
        eph = None
    if eph is not None and eph["n_used"] < 4:
        res["notes"].append(f"outburst timing not used: only {eph['n_used']} outbursts on the ephemeris")
        eph = None
    if eph is not None:
        res["best_outburst"] = dict(P=eph["P"], sigma_P=eph["sigma_P"], P_str=eph["P_str"],
                                    n_used=eph["n_used"], jitter=eph["jitter"], notes=eph["notes"],
                                    blind=bool(ob.get("blind")),
                                    method=f"BAT outburst timing ({eph['n_used']} outbursts)")

    # ---- notes -----------------------------------------------------------------------------
    notes = res["notes"]
    if best is None:
        notes.append("no significant periodicity")
    else:
        if best["kind"] == "blind" and best["family_note"] != "kept":
            notes.append("family: " + best["family_note"])
        if best.get("alias_note"):
            notes.append(best["alias_note"])
        if best["kind"] == "blind" and not best["halves_detected"]:
            notes.append("not detected in both halves of the data")
        elif best["kind"] == "blind" and not best["halves_consistent"]:
            notes.append("halves give different periods")
    if known:
        if best is not None:
            rel = relation_to(best["P"], best["sigma_P"], known["P_lit"],
                              known["P_lit_err"] if known["P_lit_err"] is not None else np.nan, T)
            if rel == "different" and best["kind"] == "blind" and matches_lit(pick):
                rel = "same (within the literature error)"
            res["relation_to_literature"] = rel
            if rel and not rel.startswith("same"):
                notes.append(f"best period is {rel} ({known['P_lit']:g} d)")
            ratio = best["P"] / known["P_lit"]
            if ratio < 0.1 or ratio > 10:
                notes.append("far from the literature orbital period: probably not the orbit (check for "
                             "systematics or a nearby source in the BAT beam)")
        if np.isfinite(known["strength"]):
            notes.append(f"literature {known['P_lit']:g} d "
                         + ("confirmed" if known["confirmed"] else "not confirmed")
                         + f" (S = {known['strength']:.1f})"
                         + (f": {known['note']}" if known.get("note") else ""))
        else:
            notes.append(f"literature {known['P_lit']:g} d: {known.get('note', 'not tested')}")
    others = [e for e in qualified if best is None or e["P"] != best["P"]]
    if others:
        notes.append("other independent periods: " + ", ".join(e["P_str"] + " d" for e in others))
    harms = [e for e in found if e.get("harmonic_of")]
    if harms:
        def _lab(e):
            P_par, kind = e["harmonic_of"]
            if kind == "near":
                return f"{e['P']:.5g} d near {P_par:.5g} d"
            if kind.startswith("f1") or kind.startswith("2f1"):
                return f"{e['P']:.5g} d = {kind} (f1 = 1/{P_par:.5g} d)"
            return f"{e['P']:.5g} d = {kind} of {P_par:.5g} d"
        notes.append("related peaks: " + ", ".join(_lab(e) for e in harms))
    if eph is not None:
        notes.append(f"outburst timing: {eph['P_str']} d from {eph['n_used']} outbursts"
                     + (f" ({eph['notes']})" if eph["notes"] else ""))

    res["_daily"], res["_orbit"], res["_pg_d"], res["_pg_o"] = daily, orbit, pg_d, pg_o
    if verbose:
        print_summary(res)
    if out_dir is not None or show:
        fig = plot_star(res, show=show)
        if out_dir is not None:
            out = Path(out_dir)
            out.mkdir(parents=True, exist_ok=True)
            stem = name.replace(" ", "_").replace("/", "_")
            fig.savefig(out / f"{stem}_BAT.png", dpi=110, bbox_inches="tight")
            with open(out / f"{stem}_BAT.json", "w") as fh:
                json.dump(to_jsonable(public_result(res)), fh, indent=1)
        if not show:
            import matplotlib.pyplot as plt
            plt.close(fig)
    return res


def public_result(res):
    """The result without the private light-curve/periodogram objects (for saving)."""
    out = {k: v for k, v in res.items() if not k.startswith("_")}
    ob = out.get("outburst_timing")
    if ob and ob.get("scan"):
        ob = dict(ob)
        ob["scan"] = {"P_best": ob["scan"]["P_best"], "Z2_best": ob["scan"]["Z2_best"]}
        out["outburst_timing"] = ob
    return out


def summary_row(res, ref=None):
    """One flat row for the results sheet."""
    b = res.get("best") or {}
    k = res.get("known") or {}
    o = res.get("best_outburst") or {}
    row = dict(folder=Path(res["folder"]).name, source=res.get("source", ""),
               n_daily=res.get("n_daily"), n_orbit=res.get("n_orbit"), T_days=res.get("T_days"),
               P_orb=b.get("P", np.nan), sigma_P=b.get("sigma_P", np.nan), P_orb_str=b.get("P_str", ""),
               strength=b.get("strength", np.nan), FAP=b.get("fap", np.nan),
               method=b.get("method", "no significant periodicity"),
               lc_used=b.get("lc_used", ""), halves_consistent=b.get("halves_consistent", np.nan),
               P_lit_tested=k.get("P_lit", np.nan), strength_at_lit=k.get("strength", np.nan),
               FAP_at_lit=k.get("fap", np.nan), lit_confirmed=k.get("confirmed", np.nan),
               relation_to_lit=res.get("relation_to_literature", ""),
               P_outburst=o.get("P", np.nan), sigma_P_outburst=o.get("sigma_P", np.nan),
               P_outburst_str=o.get("P_str", ""), n_outbursts_used=o.get("n_used", np.nan),
               threshold_blind_daily=res.get("threshold_daily", np.nan),
               threshold_blind_orbit=res.get("threshold_orbit", np.nan),
               top_peaks_daily=", ".join(f"{p['P']:.4g} ({p['strength']:.0f})"
                                         for p in res.get("peaks_daily", [])[:5]),
               top_peaks_orbit=", ".join(f"{p['P']:.4g} ({p['strength']:.0f})"
                                         for p in res.get("peaks_orbit", [])[:5]),
               notes="; ".join(res.get("notes", [])))
    if ref is not None:
        for c in ("main_id", "other_name", "cls", "P_orb_lit", "P_orb_lit_err", "P_super_lit", "my_P_orb"):
            if c in ref:
                row[c] = ref[c]
    return row


def print_summary(res):
    """Human-readable summary of analyse_star()."""
    print(f"\n=== {res['name']} ({res.get('source', '')}) ===")
    print(f"daily points {res['n_daily']}, orbit points {res['n_orbit']}, baseline {res['T_days']:.0f} d, "
          f"weighting {res['weighting']} (V_S = {res.get('v_s', np.nan):.3g})")
    print(f"blind thresholds (FAP {res['fap_level']:g}): daily S >= {res['threshold_daily']:.1f}, "
          f"orbit S >= {res['threshold_orbit']:.1f}; known-period check S >= {res['threshold_targeted']:.1f}")
    for lab, key in (("daily", "peaks_daily"), ("orbit", "peaks_orbit")):
        pk = res.get(key) or []
        if pk:
            print(f"  top {lab} peaks: " + ", ".join(f"{p['P']:.5g} d (S {p['strength']:.1f})" for p in pk[:5]))
    for e in res.get("candidates", []):
        print(f"  candidate from {e['P_peak']:.5g} d ({e['search']}): -> {e['P_str']} d "
              f"[{e['lc_used']}], S = {e['strength']:.1f}, FAP = {e['fap']:.2g}; family: {e['family']['note']}"
              + (f"; {e['alias']['note']}" if e.get("alias") else "")
              + f"; halves consistent: {e['halves']['consistent']}, detected in both: {e['halves']['detected_both']}"
              + (f"; HARMONIC ({e['harmonic_of'][1]} of {e['harmonic_of'][0]:.5g} d)" if e.get("harmonic_of") else "")
              + (f"; {e['instrumental']}" if e.get("instrumental") else ""))
    if res.get("known"):
        k = res["known"]
        print(f"  literature {k['P_lit']:g} d: S = {k['strength']:.1f}, FAP = {k['fap']:.2g} -> "
              + ("confirmed" + (f", measured {k['measure']['P_str']} d" if k.get('measure') else "")
                 if k["confirmed"] else "not confirmed"))
    if res.get("best_outburst"):
        o = res["best_outburst"]
        print(f"  outburst timing: {o['P_str']} d from {o['n_used']} outbursts (jitter {o['jitter']:.2g} d) {o['notes']}")
    b = res.get("best")
    print("  RESULT: " + (f"P = {b['P_str']} d ({b['method']})" if b else "no significant periodicity"))
    if res.get("notes"):
        print("  notes: " + "; ".join(res["notes"]))


# ======================================================================================
# Plot
# ======================================================================================

def plot_star(res, show=True):
    """Light curve, strength spectrum, folded profiles at P/2, P, 2P, and O-C or halves."""
    import matplotlib.pyplot as plt
    daily, orbit = res.get("_daily"), res.get("_orbit")
    pg_d, pg_o = res.get("_pg_d"), res.get("_pg_o")
    b = res.get("best")
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.45, wspace=0.25)
    ax_lc = fig.add_subplot(gs[0, 0])
    lc0 = daily or orbit
    ax_lc.plot(lc0.t, lc0.rate, ".", ms=1.5, color="0.3")
    ob = res.get("outburst_timing")
    if ob:
        for o in ob["outbursts"]:
            ax_lc.axvline(o["t_peak"], color="tab:red", lw=0.6, alpha=0.6)
    ax_lc.set_xlabel("MJD")
    ax_lc.set_ylabel("rate (cts cm$^{-2}$ s$^{-1}$)")
    ax_lc.set_title(f"{res['name']}: BAT {lc0.kind} light curve")

    ax_pg = fig.add_subplot(gs[0, 1])
    if pg_d is not None:
        ax_pg.plot(1 / pg_d.freq, pg_d.strength, color="k", lw=0.5, label="daily")
        ax_pg.axhline(res["threshold_daily"], color="k", ls="--", lw=0.8, label="blind 1% (daily)")
    if pg_o is not None:
        ax_pg.plot(1 / pg_o.freq, pg_o.strength, color="tab:blue", lw=0.5, alpha=0.7, label="orbit")
        ax_pg.axhline(res["threshold_orbit"], color="tab:blue", ls="--", lw=0.8, label="blind 1% (orbit)")
    ax_pg.axhline(res["threshold_targeted"], color="0.5", ls=":", lw=0.8, label="known-period 1%")
    if res.get("known"):
        ax_pg.axvline(res["known"]["P_lit"], color="tab:green", ls="--", lw=0.8, label="literature")
    if b:
        ax_pg.axvline(b["P"], color="tab:red", lw=0.8, alpha=0.7, label="result")
    ax_pg.set_xscale("log")
    ax_pg.set_xlabel("period (d)")
    ax_pg.set_ylabel("strength = power / local noise")
    ax_pg.legend(fontsize=7, ncol=2)

    if b:
        lc_b = orbit if (b["lc_used"] == "orbit" and orbit is not None) else daily
        dy, _ = effective_errors(lc_b.rate, lc_b.err, res["weighting"])
        w = 1 / dy ** 2 if dy is not None else np.ones_like(lc_b.rate)
        for i, (fac, lab) in enumerate(((0.5, "P/2"), (1.0, "P"), (2.0, "2P"))):
            ax = fig.add_subplot(gs[1 + i // 2, i % 2]) if i < 2 else fig.add_subplot(gs[2, 0])
            Pf = b["P"] * fac
            pr = fold_profile(lc_b.t, lc_b.rate, w, Pf, 16 if fac <= 1 else 32)
            ph = np.concatenate([pr["phase"], pr["phase"] + 1])
            ax.errorbar(ph, np.tile(pr["mean"], 2), np.tile(pr["err"], 2), fmt="o", ms=3, color="k")
            chi, dof = fold_chi2(lc_b.t, lc_b.rate, w, Pf, 16 if fac <= 1 else 32)
            ax.set_title(f"folded at {lab} = {Pf:.6g} d  (chi2/dof = {chi:.0f}/{dof})", fontsize=9)
            ax.set_xlabel("phase")
            ax.set_ylabel("rate")
        ax4 = fig.add_subplot(gs[2, 1])
        eph = ob["ephemeris"] if (ob and res.get("best_outburst")) else None
        if eph is not None:
            ax4.plot(eph["cycles"], eph["oc"], "o", color="tab:red")
            ax4.axhline(0, color="0.5", lw=0.8)
            ax4.set_xlabel("cycle")
            ax4.set_ylabel("O - C (d)")
            ax4.set_title(f"outburst timing: P = {eph['P_str']} d", fontsize=9)
        else:
            tm = np.median(lc_b.t)
            for m, col, lab in ((lc_b.t < tm, "tab:blue", "first half"), (lc_b.t >= tm, "tab:orange", "second half")):
                pr = fold_profile(lc_b.t[m], lc_b.rate[m], w[m], b["P"], 16)
                ax4.errorbar(pr["phase"], pr["mean"], pr["err"], fmt="o-", ms=3, color=col, label=lab)
            ax4.legend(fontsize=7)
            ax4.set_xlabel("phase")
            ax4.set_title(f"both halves folded at P = {b['P_str']} d", fontsize=9)
    fig.suptitle(f"{res['name']}: " + (f"P = {b['P_str']} d ({b['method']})" if b else "no significant periodicity"),
                 fontsize=11)
    if show:
        plt.show()
    return fig


# ======================================================================================
# All stars
# ======================================================================================

def run_all(stars_dir, references=None, out_dir=None, only=None, skip=(), **kwargs):
    """Run analyse_star() on every sub-folder of stars_dir that holds a BAT light curve.

    references : DataFrame from hmxb_common.load_references() (index = folder name); its
                 P_orb_lit is used as the targeted known period.
    Returns (DataFrame of summary rows, dict folder -> full result).
    """
    import pandas as pd
    stars_dir = Path(stars_dir)
    rows, results = [], {}
    folders = sorted(p for p in stars_dir.iterdir() if p.is_dir())
    for i, folder in enumerate(folders):
        if (only and folder.name not in only) or folder.name in skip:
            continue
        d, o = find_bat_files(folder)
        if d is None and o is None:
            continue
        ref = references.loc[folder.name] if (references is not None and folder.name in references.index) else None
        kp = ref["P_orb_lit"] if ref is not None else None
        kpe = ref["P_orb_lit_err"] if ref is not None else None
        try:
            r = analyse_star(folder, known_period=kp, known_period_err=kpe, out_dir=out_dir,
                             show=False, verbose=False, **kwargs)
            results[folder.name] = r
            rows.append(summary_row(r, ref))
            b = r.get("best")
            print(f"[{i + 1}/{len(folders)}] {folder.name}: "
                  + (f"{b['P_str']} d (S {b['strength']:.0f})" if b else "no significant period"))
        except Exception as ex:
            rows.append(dict(folder=folder.name, notes=f"ERROR {type(ex).__name__}: {ex}"))
            print(f"[{i + 1}/{len(folders)}] {folder.name}: ERROR {ex}")
    df = pd.DataFrame(rows)
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        df.to_csv(Path(out_dir) / "BAT_orbital_periods.csv", index=False)
    return df, results
