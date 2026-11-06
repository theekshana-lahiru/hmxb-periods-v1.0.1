r"""
xmm_spin_pipeline.py
====================
Spin (pulse) period search in XMM-Newton EPIC data, with the red-noise-aware significance
of Israel & Stella (1996).

Two kinds of input:
  A. pipeline (PPS) source light curves  P<obsid><inst><expo>SRCTSR<src>.FTZ  (what is in
     D:\HMXBs\Stars\<star>\XMM ...\<obsid>\pps now). The "X000" files also carry the five
     standard bands (RATE1..RATE5 = 0.2-0.5, 0.5-1, 1-2, 2-4.5, 4.5-12 keV).
  B. EPIC event lists (PPS PIEVLI/MIEVLI products or your own cleaned, barycentred files):
     load_events() selects a circular source region, PI range, PATTERN and FLAG, the source
     CCD's GTIs and (optionally) removes soft-proton flares.

Method (per observation):
  1. pick the target's light curve by position (PN preferred, then finest bins, longest exposure;
     PN timing-mode light curves, source number 0000, are accepted within 90")
  2. screen: inside source GTIs, FRACEXP >= 0.5, optional background-flare cut; zero-count bins
     are kept; rebinning uses a whole number of pipeline bins (otherwise the pattern of 2-or-3
     bins per new bin repeats at a beat period and fakes a signal)
  3. power spectrum at the independent Fourier frequencies k/T (Lomb-Scargle "psd" for the
     binned rates, which tolerates gaps)
  4. Israel & Stella normalisation: z = power / mean power of the neighbouring frequencies
     (15 on each side, the bin itself excluded); under the null z is exponential, so the
     single-trial probability is exp(-z) and the global FAP accounts for the number of trials
  5. harmonic sums z(f) + z(2f) + ... for peaked pulse profiles (Gamma distribution)
  6. fundamental check: harmonic sums also respond to P/h and k P, so the power at j/(kP)
     decides whether the profile repeats after P (Vela X-1's strongest peak is P/4)
  7. refine: harmonic fit with a delta-chi2 = 1 error; folded profile and pulsed fraction;
     presence in both halves of the observation
  8. optional search of shorter segments (orbital Doppler shifts smear fast pulsars such as
     Cen X-3 over a whole observation)
Per star, a blind detection is adopted only if FAP < 0.01 / N_observations (a further factor 10
for a signal seen in one segment only); otherwise a confirmed literature spin is used.

Quick start (Jupyter):
    import xmm_spin_pipeline as xmm
    res = xmm.analyse_xmm_star(r"D:\HMXBs\Stars\IGR J00370+6122", ra=9.29038, dec=61.36005,
                               known_spin=674.0)
    xmm.print_xmm_summary(res)

Requires numpy, scipy, astropy, matplotlib; and hmxb_common.py in the same folder.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.timeseries import LombScargle
from scipy.special import gammaincc

from hmxb_common import (harmonic_period_fit, round_with_error, threshold_mean_norm,
                         to_jsonable)

XMM_MJDREF = 50814.0
# imaging-mode X000 files: RATE1..RATE5 = 0.2-0.5, 0.5-1, 1-2, 2-4.5, 4.5-12 keV
BAND_COLUMNS = {"total": ["RATE"], "soft": ["RATE1", "RATE2", "RATE3"],   # 0.2-2 keV
                "hard": ["RATE4", "RATE5"],                               # 2-12 keV
                "1-12": ["RATE3", "RATE4", "RATE5"]}
# PN timing-mode files (source number 0000): RATE2, RATE3, RATE4, RATE100 = 0.5-1, 1-2, 2-4.5, 4.5-7 keV
BAND_COLUMNS_TIMING = {"total": ["RATE"], "soft": ["RATE2", "RATE3"],    # 0.5-2 keV
                       "hard": ["RATE4", "RATE100"],                      # 2-7 keV
                       "1-12": ["RATE3", "RATE4", "RATE100"]}


# ======================================================================================
# Finding the target's light curves
# ======================================================================================

def _ang_sep_arcsec(ra1, dec1, ra2, dec2):
    r1, d1, r2, d2 = map(np.radians, (ra1, dec1, ra2, dec2))
    c = np.sin(d1) * np.sin(d2) + np.cos(d1) * np.cos(d2) * np.cos(r1 - r2)
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))) * 3600.0)


def find_xmm_lightcurves(star_dir):
    """Header summary of every PPS source light curve (*SRCTSR*.FTZ / .FIT) below star_dir."""
    out = []
    for p in sorted(Path(star_dir).rglob("*")):
        n = p.name.upper()
        if "SRCTSR" not in n or not (n.endswith(".FTZ") or n.endswith(".FIT") or n.endswith(".FITS")):
            continue
        try:
            with fits.open(p) as h:
                H = h[1].header
                P0 = h[0].header
                cols = [c.upper() for c in h[1].columns.names]
                # imaging mode: SRC_RA/SRC_DEC; PN timing mode (source 0000): SRC_CRA/SRC_CDEC
                ra = H.get("SRC_RA", H.get("SRC_CRA", P0.get("SRC_RA", np.nan)))
                dec = H.get("SRC_DEC", H.get("SRC_CDEC", P0.get("SRC_DEC", np.nan)))
                timing = "RATE100" in cols or n.split("SRCTSR")[-1][:4] == "0000"
                out.append(dict(
                    path=str(p), obs_id=str(P0.get("OBS_ID", H.get("OBS_ID", ""))).strip(),
                    instrument=str(H.get("INSTRUME", P0.get("INSTRUME", ""))).strip(),
                    expid=str(H.get("EXPIDSTR", P0.get("EXPIDSTR", ""))).strip(),
                    object=str(H.get("OBJECT", P0.get("OBJECT", ""))).strip(),
                    timedel=float(H.get("TIMEDEL", np.nan)) if "TIMEDEL" in H else np.nan,
                    src_ra=float(ra) if ra is not None else np.nan,
                    src_dec=float(dec) if dec is not None else np.nan,
                    exposure=float(H.get("EXPOSURE", np.nan)), tstart=float(H.get("TSTART", np.nan)),
                    tstop=float(H.get("TSTOP", np.nan)), srcnum=H.get("SRCNUM", None), timing=timing,
                    multiband=("RATE4" in cols and ("RATE5" in cols or "RATE100" in cols)),
                    n_rows=int(H.get("NAXIS2", 0))))
        except Exception as ex:
            out.append(dict(path=str(p), error=repr(ex)))
    for d in out:                                  # TIMEDEL may be a column, not a keyword
        if "error" in d or np.isfinite(d.get("timedel", np.nan)):
            continue
        try:
            with fits.open(d["path"]) as h:
                t = np.asarray(h[1].data["TIME"], float)
                d["timedel"] = float(np.median(np.diff(t))) if len(t) > 1 else np.nan
        except Exception:
            pass
    return out


def select_target_lightcurves(lcs, ra, dec, radius_arcsec=15.0, timing_radius_arcsec=90.0):
    """Light curves of the source at (ra, dec), best first within each observation.

    PN timing-mode light curves (source number 0000) are one-dimensional: their SRC_CRA/SRC_CDEC
    is only approximate, so they are accepted within timing_radius_arcsec (they always belong to
    the prime target of the observation).
    Order: PN before MOS, finer bins first, longer exposure first, multi-band files first.
    Returns dict obs_id -> list of header dicts."""
    good = []
    for d in lcs:
        if "error" in d:
            continue
        if not np.isfinite(d.get("src_ra", np.nan)):
            if d.get("timing"):
                good.append(dict(d, sep_arcsec=np.nan))
            continue
        sep = _ang_sep_arcsec(ra, dec, d["src_ra"], d["src_dec"])
        if sep <= (timing_radius_arcsec if d.get("timing") else radius_arcsec):
            good.append(dict(d, sep_arcsec=sep))
    by_obs = {}
    for d in good:
        by_obs.setdefault(d["obs_id"], []).append(d)
    for k in by_obs:
        by_obs[k].sort(key=lambda d: (0 if d["instrument"].upper() == "EPN" else 1,
                                      round(d["timedel"], 3) if np.isfinite(d["timedel"]) else 1e9,
                                      -round(d["exposure"] if np.isfinite(d["exposure"]) else 0, -1),
                                      0 if d["multiband"] else 1))
    return by_obs


# ======================================================================================
# Light curves
# ======================================================================================

@dataclass
class XLightCurve:
    t: np.ndarray        # s since the first bin (bin centres)
    rate: np.ndarray     # net count rate (counts/s)
    err: np.ndarray
    dt: float            # bin size (s)
    t0_met: float        # XMM MET (s) of t = 0
    mjdref: float = XMM_MJDREF
    instrument: str = ""
    obs_id: str = ""
    path: str = ""
    band: str = "total"

    @property
    def T(self):
        return float(self.t[-1] - self.t[0] + self.dt) if len(self.t) else 0.0

    @property
    def mjd_start(self):
        return self.mjdref + self.t0_met / 86400.0

    def subset(self, mask):
        return replace(self, t=self.t[mask], rate=self.rate[mask], err=self.err[mask])


def load_xmm_lc(path, band="total", min_fracexp=0.5, back_sigma=None, bin_size=None):
    """Load a PPS source light curve, screen it, and optionally rebin.

    band       : "total" (RATE), "hard" (2-12 keV), "1-12", "soft" (needs an X000 multi-band file)
    min_fracexp: minimum fractional exposure of a bin
    back_sigma : if set, drop bins whose background rate BACKV exceeds median + back_sigma*MAD
    bin_size   : new bin size in s (an integer multiple of the original is best)
    """
    with fits.open(path) as h:
        d, H = h[1].data, h[1].header
        P0 = h[0].header
        cols = [c.upper() for c in d.columns.names]
        time = np.asarray(d["TIME"], float)
        names = (BAND_COLUMNS_TIMING if "RATE100" in cols else BAND_COLUMNS)[band]
        if not all(n in cols for n in names):
            if band != "total":
                warnings.warn(f"band '{band}' not available in {Path(path).name}; using total RATE")
            names, band = ["RATE"], "total"
        rate = np.zeros_like(time)
        var = np.zeros_like(time)
        for n in names:
            rate += np.asarray(d[n], float)
            ecol = "ERROR" if n == "RATE" else f"{n}_ERR"
            var += np.asarray(d[ecol], float) ** 2 if ecol in cols else 0.0
        err = np.sqrt(var)
        fracexp = np.asarray(d["FRACEXP"], float) if "FRACEXP" in cols else np.ones_like(time)
        backv = np.asarray(d["BACKV"], float) if "BACKV" in cols else None
        if "TIMEDEL" in H:
            dt = float(H["TIMEDEL"])
        elif "TIMEDEL" in cols:
            dt = float(np.median(np.asarray(d["TIMEDEL"], float)))
        else:
            dt = float(np.median(np.diff(time)))
        pixr = float(H.get("TIMEPIXR", 0.5))
        mjdref = float(H.get("MJDREF", XMM_MJDREF))
        gti = None
        for ext in ("SRC_GTIS", "STDGTI", "GTI"):
            if ext in h:
                g = h[ext].data
                gti = np.column_stack([np.asarray(g["START"], float), np.asarray(g["STOP"], float)])
                break
        inst = str(H.get("INSTRUME", P0.get("INSTRUME", ""))).strip()
        obs = str(P0.get("OBS_ID", H.get("OBS_ID", ""))).strip()
    tc = time + (0.5 - pixr) * dt                      # bin centres (MET)
    fe_pos = fracexp[np.isfinite(fracexp) & (fracexp > 0)]
    if fe_pos.size and np.median(fe_pos) < min_fracexp:
        # some early PN timing-mode products have FRACEXP ~0.03 in every bin: use a relative cut
        min_fracexp = 0.5 * float(np.median(fe_pos))
    good = np.isfinite(tc) & np.isfinite(rate) & np.isfinite(err) & (fracexp >= min_fracexp)
    if gti is not None and len(gti):
        in_gti = np.zeros_like(good)
        for a, b in gti:
            in_gti |= (tc >= a) & (tc <= b)
        good &= in_gti
    if back_sigma is not None and backv is not None:
        bv = backv[good]
        med = np.median(bv)
        mad = 1.4826 * np.median(np.abs(bv - med))
        good &= backv <= med + back_sigma * max(mad, 1e-12)
    # Bins with zero source counts have ERROR = 0 in the pipeline files. They must be KEPT
    # (dropping them biases the light curve and creates fake periods); give them a
    # one-count error instead.
    floor = 1.0 / (dt * np.clip(fracexp, 0.1, None))
    err = np.where(err > 0, err, floor)
    t0 = tc[good][0] if good.any() else tc[0]
    lc = XLightCurve(t=tc[good] - t0, rate=rate[good], err=err[good], dt=dt, t0_met=t0,
                     mjdref=mjdref, instrument=inst, obs_id=obs, path=str(path), band=band)
    if bin_size is not None and bin_size > dt * 1.5:
        lc = rebin(lc, bin_size)
    return lc


def rebin(lc, bin_size):
    """Average a light curve into bins of bin_size seconds.

    Plain means (NOT inverse-variance weights: with Poisson errors those favour bins with few
    counts and bias the rate); error = sqrt(sum err^2) / n. Bins less than half filled are dropped."""
    k = np.floor((lc.t + 0.5 * lc.dt) / bin_size).astype(np.int64)   # half-bin offset: no rounding ties
    uk, inv = np.unique(k, return_inverse=True)
    n = np.bincount(inv)
    r = np.bincount(inv, lc.rate) / n
    e = np.sqrt(np.bincount(inv, lc.err ** 2)) / n
    full = n >= 0.5 * bin_size / lc.dt
    return replace(lc, t=(uk[full] + 0.5) * bin_size - 0.5 * lc.dt, rate=r[full], err=e[full],
                   dt=float(bin_size))


# ======================================================================================
# Event files (optional branch)
# ======================================================================================

def load_events(evt_path, ra, dec, radius_arcsec=20.0, pi_range=(500, 10000), pattern_max=4,
                flag_zero=True, flare_threshold=None, flare_bin=100.0, barycentre=False):
    """Source photon arrival times from an EPIC event list.

    Selection: circle of radius_arcsec around (ra, dec) using the X/Y sky-pixel WCS, PI in
    pi_range (eV), PATTERN <= pattern_max, FLAG == 0; times inside the source CCD's GTIs.
    flare_threshold : e.g. 0.4 (PN) or 0.35 (MOS) counts/s in 10-12 keV, single-pixel events,
                      over the whole field in flare_bin s bins; bins above it are removed.
    barycentre      : approximate barycentric correction with astropy (geocentre; the
                      < 0.5 s XMM orbit term is ignored, fine for P > ~10 s). For precise work,
                      run SAS barycen on the event file instead.
    Returns dict(times (s, MET or TDB-corrected), gti (N x 2), mjdref, n_events, ccd).
    """
    from astropy.wcs import WCS
    with fits.open(evt_path) as h:
        ev = h["EVENTS"]
        d, H = ev.data, ev.header
        cols = [c.upper() for c in d.columns.names]
        ix, iy = cols.index("X") + 1, cols.index("Y") + 1
        w = WCS(naxis=2)
        w.wcs.ctype = [H[f"TCTYP{ix}"], H[f"TCTYP{iy}"]]
        w.wcs.crpix = [H[f"TCRPX{ix}"], H[f"TCRPX{iy}"]]
        w.wcs.crval = [H[f"TCRVL{ix}"], H[f"TCRVL{iy}"]]
        w.wcs.cdelt = [H[f"TCDLT{ix}"], H[f"TCDLT{iy}"]]
        x0, y0 = w.all_world2pix([[ra, dec]], 1)[0]
        pix_arcsec = abs(H[f"TCDLT{ix}"]) * 3600.0
        x, y = np.asarray(d["X"], float), np.asarray(d["Y"], float)
        t = np.asarray(d["TIME"], float)
        sel = (x - x0) ** 2 + (y - y0) ** 2 <= (radius_arcsec / pix_arcsec) ** 2
        if "PI" in cols:
            pi = np.asarray(d["PI"], float)
            sel &= (pi >= pi_range[0]) & (pi <= pi_range[1])
        if "PATTERN" in cols:
            sel &= np.asarray(d["PATTERN"]) <= pattern_max
        if flag_zero and "FLAG" in cols:
            sel &= np.asarray(d["FLAG"]) == 0
        ccd = None
        if "CCDNR" in cols and sel.any():
            vals, cnt = np.unique(np.asarray(d["CCDNR"])[sel], return_counts=True)
            ccd = int(vals[np.argmax(cnt)])
        gti = None
        names = [hd.name for hd in h]
        cand = [f"STDGTI{ccd:02d}"] if ccd is not None else []
        cand += [n for n in names if n.startswith("STDGTI")] + ["GTI"]
        for n in cand:
            if n in names:
                g = h[n].data
                gti = np.column_stack([np.asarray(g["START"], float), np.asarray(g["STOP"], float)])
                break
        if flare_threshold is not None and "PI" in cols:
            hi_e = (np.asarray(d["PI"]) > 10000) & (np.asarray(d["PI"]) < 12000)
            if "PATTERN" in cols:
                hi_e &= np.asarray(d["PATTERN"]) == 0
            edges = np.arange(t.min(), t.max() + flare_bin, flare_bin)
            cnt, _ = np.histogram(t[hi_e], edges)
            bad = cnt / flare_bin > flare_threshold
            good_iv = np.column_stack([edges[:-1][~bad], edges[1:][~bad]])
            gti = good_iv if gti is None else _intersect_gti(gti, good_iv)
        mjdref = float(H.get("MJDREF", XMM_MJDREF))
    times = t[sel]
    if gti is not None:
        inside = np.zeros(times.size, bool)
        for a, b in gti:
            inside |= (times >= a) & (times <= b)
        times = times[inside]
    times = np.sort(times)
    if barycentre and times.size:
        times = barycentre_times(times, mjdref, ra, dec)
        if gti is not None:
            gti = np.column_stack([barycentre_times(gti[:, 0], mjdref, ra, dec),
                                   barycentre_times(gti[:, 1], mjdref, ra, dec)])
    return dict(times=times, gti=gti, mjdref=mjdref, n_events=int(times.size), ccd=ccd)


def _intersect_gti(g1, g2):
    out = []
    for a1, b1 in g1:
        for a2, b2 in g2:
            a, b = max(a1, a2), min(b1, b2)
            if b > a:
                out.append((a, b))
    return np.array(out) if out else np.zeros((0, 2))


def barycentre_times(t_met, mjdref, ra, dec):
    """Approximate barycentric correction (geocentric observer, astropy built-in ephemeris)."""
    from astropy.coordinates import EarthLocation, SkyCoord
    from astropy.time import Time
    import astropy.units as u
    tt = Time(mjdref + np.asarray(t_met, float) / 86400.0, format="mjd", scale="tt",
              location=EarthLocation.from_geocentric(0, 0, 0, unit=u.m))
    ltt = tt.light_travel_time(SkyCoord(ra, dec, unit="deg"), kind="barycentric")
    return np.asarray(t_met, float) + ltt.to_value(u.s)


def events_to_lightcurve(ev, bin_size=1.0, min_frac=0.5):
    """Bin event times into a rate light curve (bins mostly inside GTIs only)."""
    t = ev["times"]
    t0 = t.min()
    gti = ev["gti"] if ev["gti"] is not None else np.array([[t.min(), t.max()]])
    edges = np.arange(t0, t.max() + bin_size, bin_size)
    cnt, _ = np.histogram(t, edges)
    cover = np.zeros(len(edges) - 1)
    for a, b in gti:                                   # exposure fraction of each bin
        lo = np.clip(a, edges[:-1], edges[1:])
        hi = np.clip(b, edges[:-1], edges[1:])
        cover += np.clip(hi - lo, 0, None)
    frac = cover / bin_size
    ok = frac >= min_frac
    rate = cnt[ok] / (bin_size * frac[ok])
    err = np.sqrt(np.maximum(cnt[ok], 1)) / (bin_size * frac[ok])
    tc = 0.5 * (edges[:-1] + edges[1:])[ok]
    return XLightCurve(t=tc - tc[0], rate=rate, err=err, dt=bin_size, t0_met=tc[0],
                       mjdref=ev["mjdref"], band="events")


# ======================================================================================
# Power spectrum with Israel & Stella normalisation
# ======================================================================================

def local_mean_power(power, k=15, exclude=1):
    """Mean of the k neighbouring powers on each side of every bin, excluding +/-exclude bins."""
    n = len(power)
    cs = np.concatenate([[0.0], np.cumsum(power)])
    idx = np.arange(n)
    lo1, hi1 = np.clip(idx - exclude - k, 0, n), np.clip(idx - exclude, 0, n)
    lo2, hi2 = np.clip(idx + exclude + 1, 0, n), np.clip(idx + exclude + 1 + k, 0, n)
    s = (cs[hi1] - cs[lo1]) + (cs[hi2] - cs[lo2])
    m = (hi1 - lo1) + (hi2 - lo2)
    return s / np.maximum(m, 1)


@dataclass
class SpinSpectrum:
    freq: np.ndarray      # Hz, independent Fourier frequencies k/T
    power: np.ndarray
    local_mean: np.ndarray
    z: np.ndarray         # power / local mean (Israel & Stella)
    T: float
    n_trials: float

    def z_at(self, P, halfwidth=None):
        """Highest z within +/- halfwidth (default 1/T) of 1/P: (P_peak, z)."""
        f0 = 1.0 / P
        hw = 1.0 / self.T if halfwidth is None else halfwidth
        idx = np.where((self.freq >= f0 - hw) & (self.freq <= f0 + hw))[0]
        if idx.size == 0:
            return np.nan, np.nan
        k = idx[int(np.argmax(self.z[idx]))]
        return float(1.0 / self.freq[k]), float(self.z[k])

    def z_nearest(self, f):
        """z at the independent frequency nearest to f (NaN outside the searched range).
        Unlike z_at() there is no maximisation, so under the null it is exponential."""
        if len(self.freq) < 2:
            return np.nan
        df = self.freq[1] - self.freq[0]
        i = int(np.rint((f - self.freq[0]) / df))
        if i < 0 or i >= len(self.freq):
            return np.nan
        return float(self.z[i])


def spin_spectrum(lc, Pmin=None, Pmax=None, k=15, exclude=1):
    """Power spectrum over independent frequencies with Israel & Stella normalisation.

    Pmin defaults to 2 bins (Nyquist), Pmax to T/4 (at least four cycles)."""
    T = lc.T
    fmin = 1.0 / (Pmax if Pmax is not None else T / 4.0)
    fmax = 1.0 / (Pmin if Pmin is not None else 2.0 * lc.dt)
    fmin = max(fmin, 1.0 / T)
    freq = np.arange(np.ceil(fmin * T), np.floor(fmax * T) + 1) / T
    # extend by k bins on both sides so that the edge bins also get a two-sided local mean
    lo_ext = max(1.0, freq[0] * T - k - 1)
    full = np.arange(lo_ext, freq[-1] * T + k + 2) / T
    full = full[full < 0.5 / lc.dt]
    ls = LombScargle(lc.t, lc.rate, None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pw = ls.power(full, method="fast" if len(full) > 2000 else "auto",
                      assume_regular_frequency=len(full) > 2000, normalization="psd")
    pw = np.clip(np.asarray(pw, float), 0, None)
    lm = local_mean_power(pw, k, exclude)
    sel = (full >= freq[0] - 0.5 / T) & (full <= freq[-1] + 0.5 / T)
    f, p, m = full[sel], pw[sel], lm[sel]
    z = p / np.where(m > 0, m, np.nan)
    return SpinSpectrum(freq=f, power=p, local_mean=m, z=z, T=T, n_trials=float(len(f)))


def harmonic_sum(spec, nharm=2):
    """Z_m(f) = sum of z at f, 2f, ..., m f (nearest bins). Null: Gamma(m, 1)."""
    f = spec.freq
    zs = np.zeros_like(spec.z)
    for h in range(1, nharm + 1):
        j = np.searchsorted(spec.freq, h * f)
        ok = j < len(f)
        zz = np.zeros_like(zs)
        zz[ok] = np.nan_to_num(spec.z[j[ok]], nan=0.0)
        zs += zz
    return zs


def gamma_fap(Z, m, n_trials):
    """Single-trial and global FAP for a sum of m exponential variables."""
    p1 = np.clip(gammaincc(m, np.asarray(Z, float)), 0, 1)
    return p1, -np.expm1(n_trials * np.log1p(-np.minimum(p1, 1 - 1e-16)))


def search_spin(lc, Pmin=None, Pmax=None, nharm_max=4, k=15, fap_level=0.01, n_peaks=6):
    """Blind search: top peaks by global FAP over 1..nharm_max harmonic sums."""
    spec = spin_spectrum(lc, Pmin, Pmax, k)
    rows = []
    for m in range(1, nharm_max + 1):
        Z = spec.z if m == 1 else harmonic_sum(spec, m)
        Zc = np.nan_to_num(Z, nan=0.0)
        idx = np.where((Zc[1:-1] > Zc[:-2]) & (Zc[1:-1] >= Zc[2:]))[0] + 1
        idx = idx[np.argsort(Zc[idx])[::-1][:n_peaks * 3]]
        for i in idx:
            p1, fap = gamma_fap(Zc[i], m, spec.n_trials * nharm_max)
            rows.append(dict(P=float(1 / spec.freq[i]), f=float(spec.freq[i]), m=m, Z=float(Zc[i]),
                             z1=float(np.nan_to_num(spec.z[i])), p_single=float(p1), fap=float(fap)))
    rows.sort(key=lambda r: (r["fap"], -r["Z"]))
    peaks = []
    for r in rows:                                   # one entry per frequency
        if all(abs(r["f"] - q["f"]) > 1.5 / spec.T for q in peaks):
            peaks.append(r)
        if len(peaks) >= n_peaks:
            break
    thr1 = threshold_mean_norm(spec.n_trials * nharm_max, fap_level)
    return dict(spectrum=spec, peaks=peaks, threshold_z1=thr1, n_trials=spec.n_trials * nharm_max,
                significant=[p for p in peaks if p["fap"] < fap_level])


def fold_rate(lc, P, nbins=16):
    """Folded profile (inverse-variance weighted) and pulsed fraction (max-min)/(max+min)."""
    ph = (lc.t / P) % 1.0
    b = np.minimum((ph * nbins).astype(int), nbins - 1)
    w = 1.0 / lc.err ** 2
    sw = np.bincount(b, w, nbins)
    m = np.bincount(b, w * lc.rate, nbins) / sw
    e = 1.0 / np.sqrt(sw)
    pf = (np.nanmax(m) - np.nanmin(m)) / (np.nanmax(m) + np.nanmin(m)) if np.nanmax(m) + np.nanmin(m) > 0 else np.nan
    return dict(phase=(np.arange(nbins) + 0.5) / nbins, mean=m, err=e, pulsed_fraction=float(pf))


def refine_spin(lc, P0, nharm="auto"):
    """Harmonic fit around P0 (delta-chi2 = 1 error, rescaled), pulsed fraction."""
    res = harmonic_period_fit(lc.t, lc.rate, lc.err, 1.0 / P0, nharm=nharm, halfwidth=1.0 / lc.T,
                              n_coarse=61, n_fine=61)
    res["P_str"] = round_with_error(res["P"], res["sigma_P"])[2]
    res["fold"] = fold_rate(lc, res["P"])
    res["pulsed_fraction"] = res["fold"]["pulsed_fraction"]
    return res


def halves_check(lc, P, k=15, fap_level=0.01):
    """z at P (single trial) in each half of the observation."""
    tm = 0.5 * (lc.t[0] + lc.t[-1])
    out = []
    for m in (lc.t < tm, lc.t >= tm):
        sub = lc.subset(m)
        if len(sub.t) < 50 or sub.T < 3 * P:
            out.append(dict(z=np.nan, P_peak=np.nan, note="too short"))
            continue
        spec = spin_spectrum(sub, Pmin=P / 1.5, Pmax=min(P * 1.5, sub.T / 3), k=k)
        Pp, z = spec.z_at(P, halfwidth=1.0 / sub.T)
        out.append(dict(z=z, P_peak=Pp, p_single=float(np.exp(-z)) if np.isfinite(z) else np.nan))
    thr = threshold_mean_norm(1, fap_level)
    return dict(halves=out, detected_both=all(np.isfinite(o["z"]) and o["z"] >= thr for o in out),
                threshold=thr)


# ======================================================================================
# Fundamental period: harmonics and sub-harmonics
# ======================================================================================

def spin_family_check(spec, P0, T, nharm=4, p_diff=1e-3, p_same=0.05, k_down=(2, 3),
                      k_up=(2, 3, 4, 5), min_cycles=3, max_steps=6):
    """Fundamental spin period behind a peak, decided in the power spectrum.

    Harmonic sums also respond to harmonics (P/h) and sub-harmonics (k P) of the true period,
    so the strongest peak is not always the spin. A signal with period P has power only at
    j/P (j = 1, 2, ...). Using the Israel & Stella z (exponential under the null, so a sum of n
    values follows a Gamma(n) distribution):
      up:   P -> k P   if the frequencies j/(kP) with j not a multiple of k carry significant
                       power (p < p_diff): the profile does not repeat after P
      down: P -> P/k   if those frequencies of P itself are consistent with noise (p > p_same)
                       while k/P has power: the profile repeats k times per P
    Frequency-domain tests are not biased by the time binning, by flares (local normalisation)
    or by slow phase drifts, unlike folded-profile comparisons."""
    thr1 = threshold_mean_norm(1, 0.01)

    def zsum(f, k):
        js = [j for j in range(1, k * nharm + 1) if j % k != 0]
        zs = np.array([spec.z_nearest(j * f) for j in js])
        zs = zs[np.isfinite(zs)]
        if zs.size == 0:
            return np.nan, 0, np.nan
        Z = float(zs.sum())
        return Z, int(zs.size), float(gammaincc(zs.size, Z))

    steps, P = [], float(P0)
    for _ in range(max_steps):                      # is P a sub-harmonic of the spin?
        moved = False
        for k in k_down:
            Z, n, p = zsum(1.0 / P, k)
            z_k = spec.z_nearest(k / P)
            steps.append(dict(test=f"P/{k} = {P / k:.6g} s enough?", Z_other=Z, n_terms=n, p=p, z_P_over_k=z_k))
            if n > 0 and np.isfinite(p) and p > p_same and np.isfinite(z_k) and z_k >= thr1:
                P /= k
                moved = True
                break
        if not moved:
            break
    for _ in range(max_steps):                      # is P a harmonic of the spin?
        moved = False
        for k in k_up:
            if T / (k * P) < min_cycles:
                continue
            Z, n, p = zsum(1.0 / (k * P), k)
            steps.append(dict(test=f"{k} x P = {k * P:.6g} s needed?", Z_sub=Z, n_terms=n, p=p))
            if n >= 2 and np.isfinite(p) and p < p_diff:
                P *= k
                moved = True
                break
        if not moved:
            break
    changed = not np.isclose(P, P0, rtol=1e-9)
    return dict(P_start=float(P0), P_chosen=float(P), steps=steps,
                note=(f"{P0:.6g} s -> {P:.6g} s (fundamental)" if changed else "kept"))


def harmonic_fap_at(spec, P, nharm_max, n_trials, halfwidth_bins=1):
    """Best harmonic sum Z_m (m = 1..nharm_max) within +/- halfwidth_bins of 1/P, and its FAP."""
    if not (spec.freq[0] - 1.0 / spec.T <= 1.0 / P <= spec.freq[-1] + 1.0 / spec.T):
        return dict(Z=np.nan, m=np.nan, p_single=np.nan, fap=np.nan)
    i0 = int(np.argmin(np.abs(spec.freq - 1.0 / P)))
    best = None
    for m in range(1, nharm_max + 1):
        Z = spec.z if m == 1 else harmonic_sum(spec, m)
        lo, hi = max(0, i0 - halfwidth_bins), min(len(Z), i0 + halfwidth_bins + 1)
        j = lo + int(np.nanargmax(np.nan_to_num(Z[lo:hi])))
        p1, fap = gamma_fap(np.nan_to_num(Z[j]), m, n_trials)
        if best is None or fap < best["fap"]:
            best = dict(Z=float(np.nan_to_num(Z[j])), m=m, p_single=float(p1), fap=float(fap))
    return best


def mark_spin_relations(entries, T, hmax=10, kmax=4):
    """Label detections that are harmonics (P/h), sub-harmonics (k P) or rational multiples
    (k P / h) of a MORE significant independent detection (entries sorted by significance)."""
    for i, e in enumerate(entries):
        fe, sfe = 1.0 / e["P_fit"], e["sigma_P"] / e["P_fit"] ** 2
        for b in entries[:i]:
            if b.get("related_to"):
                continue
            fb, sfb = 1.0 / b["P_fit"], b["sigma_P"] / b["P_fit"] ** 2
            for kk in range(1, kmax + 1):
                for h in range(1, hmax + 1):
                    if kk == h or (kk > 1 and h > 1 and np.gcd(kk, h) > 1):
                        continue
                    if abs(kk * fe - h * fb) <= max(3.0 * np.hypot(kk * sfe, h * sfb), 1.0 / T):
                        lab = (f"P/{h}" if kk == 1 else (f"{kk}P" if h == 1 else f"{kk}P/{h}"))
                        e["related_to"] = (float(b["P_fit"]), lab)
                        break
                if e.get("related_to"):
                    break
            if e.get("related_to"):
                break
    return entries


# ======================================================================================
# One observation / one star
# ======================================================================================

def analyse_observation(lc, known_spin=None, known_spin_err=None, Pmin=None, Pmax=None,
                        nharm_max=4, k=15, fap_level=0.01, trials_factor=1.0, max_candidates=4):
    """Search, family check (fundamental vs harmonics), refinement, halves check and the
    targeted check of a known spin for one light curve.

    trials_factor multiplies the number of trials (e.g. the number of segments searched)."""
    s = search_spin(lc, Pmin=Pmin, Pmax=Pmax, nharm_max=nharm_max, k=k, fap_level=fap_level)
    if trials_factor != 1.0:
        n_tr = s["n_trials"] * trials_factor
        for p in s["peaks"]:
            p["fap"] = float(gamma_fap(p["Z"], p["m"], n_tr)[1])
        s["n_trials"] = n_tr
        s["threshold_z1"] = threshold_mean_norm(n_tr, fap_level)
        s["significant"] = [p for p in s["peaks"] if p["fap"] < fap_level]
    spec = s["spectrum"]
    out = dict(obs_id=lc.obs_id, instrument=lc.instrument, band=lc.band, dt=lc.dt, T=lc.T,
               n_bins=len(lc.t), mean_rate=float(np.mean(lc.rate)), mjd_start=lc.mjd_start,
               path=lc.path, peaks=s["peaks"], threshold_z1=s["threshold_z1"],
               n_trials=s["n_trials"], significant=[], known=None)
    entries = []
    for p in s["significant"][:max_candidates]:
        ref0 = refine_spin(lc, p["P"])                       # precise period of the peak
        fam = spin_family_check(spec, ref0["P"], lc.T, nharm=nharm_max)   # harmonic/sub-harmonic?
        P1 = fam["P_chosen"]
        if fam["note"] == "kept":
            ref = ref0
        else:
            ref = refine_spin(lc, P1)
        at = harmonic_fap_at(spec, ref["P"], nharm_max, s["n_trials"])
        hv = halves_check(lc, ref["P"], k, fap_level)
        e = dict(P_peak=p["P"], m_peak=p["m"], Z_peak=p["Z"], fap=p["fap"], family=fam,
                 Z_at_P=at["Z"], m=at["m"], fap_at_P=at["fap"], refine=ref, halves=hv,
                 P=ref["P"], P_fit=ref["P"], sigma_P=ref["sigma_P"], P_str=ref["P_str"], Z=p["Z"])
        tol = max(3.0 * ref["sigma_P"], ref["P"] ** 2 / lc.T)
        dup = next((x for x in entries if abs(x["P_fit"] - e["P_fit"]) <= max(tol, 3.0 * x["sigma_P"])), None)
        if dup is not None:
            dup.setdefault("also_from_peaks", []).append(float(p["P"]))
        else:
            entries.append(e)
    out["significant"] = mark_spin_relations(entries, lc.T)
    if known_spin is not None and np.isfinite(known_spin) and known_spin > 0:
        inside = spec.freq[0] <= 1.0 / known_spin <= spec.freq[-1]
        if inside:
            sf = (known_spin_err / known_spin ** 2) if (known_spin_err is not None and np.isfinite(known_spin_err)) else 0.0
            hw = max(1.0 / spec.T, 3 * sf)
            n_k = max(1.0, 2 * hw * spec.T)
            best = None
            for m in range(1, nharm_max + 1):
                Z = spec.z if m == 1 else harmonic_sum(spec, m)
                idx = np.where((spec.freq >= 1 / known_spin - hw) & (spec.freq <= 1 / known_spin + hw))[0]
                if idx.size == 0:
                    continue
                i = idx[int(np.nanargmax(np.nan_to_num(Z[idx])))]
                p1, fap = gamma_fap(np.nan_to_num(Z[i]), m, n_k * nharm_max * trials_factor)
                if best is None or fap < best["fap"]:
                    best = dict(P_peak=float(1 / spec.freq[i]), m=m, Z=float(np.nan_to_num(Z[i])),
                                p_single=float(p1), fap=float(fap))
            if best is not None:
                best.update(P_lit=float(known_spin), confirmed=bool(best["fap"] < fap_level), n_trials=n_k)
                if best["confirmed"]:
                    best["refine"] = refine_spin(lc, best["P_peak"])
                out["known"] = best
        else:
            out["known"] = dict(P_lit=float(known_spin), confirmed=False,
                                note="outside the searchable range (bin size or length)")
    out["_lc"], out["_spec"] = lc, spec
    return out


def analyse_xmm_star(star_dir, ra, dec, name=None, known_spin=None, known_spin_err=None,
                     radius_arcsec=15.0, band="total", bin_size=None, min_bin=None,
                     back_sigma=None, Pmin=None, Pmax=None, nharm_max=4, k=15, fap_level=0.01,
                     segment_length=None, max_obs=None, out_dir=None, show=True, verbose=True):
    """Spin search in every XMM observation of one star (pipeline light curves).

    ra, dec        : target position (deg); the light curve whose SRC_RA/SRC_DEC lies within
                     radius_arcsec is used (PN preferred)
    band           : "total" or "hard"/"1-12"/"soft" (multi-band X000 files only)
    bin_size       : rebin to this many seconds (default: keep the pipeline binning, but at least
                     min_bin s if given); always rounded up to a whole number of pipeline bins
    segment_length : also search non-overlapping segments of this length (s); their FAPs are
                     corrected for the extra trials (pulsations sometimes appear only during flares)
    """
    star_dir = Path(star_dir)
    name = name or star_dir.name
    lcs = find_xmm_lightcurves(star_dir)
    by_obs = select_target_lightcurves(lcs, ra, dec, radius_arcsec)
    res = dict(name=name, ra=ra, dec=dec, n_lightcurves=len(lcs), observations=[],
               known_spin=known_spin, band=band, notes=[])
    if not by_obs:
        res["notes"].append("no XMM pipeline light curve of the target found "
                            f"(searched {len(lcs)} files within {radius_arcsec}\")")
        res["obs_without_target"] = sorted({d["obs_id"] for d in lcs if "error" not in d})
        res["best"] = None
        if out_dir is not None:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            with open(Path(out_dir) / f"{name.replace(' ', '_')}_XMM.json", "w") as fh:
                json.dump(to_jsonable(public_xmm_result(res)), fh, indent=1)
        return res
    for i, (obs_id, cand) in enumerate(sorted(by_obs.items())):
        if max_obs is not None and i >= max_obs:
            break
        use = cand[0]
        if band != "total":                       # same instrument and binning, multi-band file
            mb = [d for d in cand if d["multiband"] and d["instrument"] == use["instrument"]
                  and round(d["timedel"], 3) == round(use["timedel"], 3)]
            if mb:
                use = mb[0]
        want_band = band if use["multiband"] else "total"
        bs = bin_size
        if bs is None and min_bin is not None and np.isfinite(use["timedel"]) and use["timedel"] < min_bin:
            bs = min_bin
        if bs is not None and np.isfinite(use["timedel"]) and use["timedel"] > 0:
            # an integer number of original bins per new bin: otherwise the 2-or-3 (8-or-9 ...)
            # pattern of original bins per new bin repeats at a beat period and can fake a signal
            bs = use["timedel"] * max(1, int(np.ceil(bs / use["timedel"] - 1e-6)))
        try:
            lc = load_xmm_lc(use["path"], band=want_band, back_sigma=back_sigma, bin_size=bs)
            if len(lc.t) < 100:
                res["observations"].append(dict(obs_id=obs_id, path=use["path"], note="too few good bins"))
                continue
            ob = analyse_observation(lc, known_spin, known_spin_err, Pmin, Pmax, nharm_max, k, fap_level)
            ob["sep_arcsec"] = use["sep_arcsec"]
            if segment_length is not None and lc.T > 1.5 * segment_length:
                n_seg = int(lc.T // segment_length)
                ob["segments"] = []
                for j in range(n_seg):
                    m = (lc.t >= j * segment_length) & (lc.t < (j + 1) * segment_length)
                    if m.sum() < 100:
                        continue
                    sg = analyse_observation(lc.subset(m), known_spin, known_spin_err, Pmin, Pmax,
                                             nharm_max, k, fap_level, trials_factor=n_seg + 1,
                                             max_candidates=2)
                    sg["segment"] = (j * segment_length, (j + 1) * segment_length)
                    ob["segments"].append({kk: v for kk, v in sg.items() if not kk.startswith("_")})
            res["observations"].append(ob)
        except Exception as ex:
            res["observations"].append(dict(obs_id=obs_id, path=use["path"], note=f"ERROR {type(ex).__name__}: {ex}"))
    res["obs_without_target"] = sorted({d["obs_id"] for d in lcs if "error" not in d} - set(by_obs))
    res["segment_length"] = segment_length
    res["fap_level"] = fap_level
    choose_best_spin(res, fap_level)
    if verbose:
        print_xmm_summary(res)
    if out_dir is not None or show:
        import matplotlib.pyplot as plt
        for ob in res["observations"]:
            if "_lc" not in ob:
                continue
            fig = plot_observation(ob, name, show=show)
            if out_dir is not None:
                Path(out_dir).mkdir(parents=True, exist_ok=True)
                stem = f"{name.replace(' ', '_')}_{ob['obs_id']}_XMM"
                fig.savefig(Path(out_dir) / f"{stem}.png", dpi=110, bbox_inches="tight")
            if not show:
                plt.close(fig)
        if out_dir is not None:
            with open(Path(out_dir) / f"{name.replace(' ', '_')}_XMM.json", "w") as fh:
                json.dump(to_jsonable(public_xmm_result(res)), fh, indent=1)
    return res


def choose_best_spin(res, fap_level=0.01):
    """Adopted spin of a star (sets res["best"] and res["notes"]); works on live or JSON results.

    Each observation's FAP covers the trials of that observation (and its segments). Over a
    star's N observations the chance of a false alarm grows, so a blind detection is adopted only
    if FAP < fap_level / N (Bonferroni); a signal seen only in one segment needs a further
    factor 10 (single-segment signals are where false alarms concentrate). Otherwise a confirmed
    literature spin (targeted check) is adopted."""
    obs = [ob for ob in res.get("observations", []) if "peaks" in ob]
    n_obs = max(1, len(obs))
    lev_whole = fap_level / n_obs
    lev_seg = lev_whole / 10.0
    seg_len = res.get("segment_length")
    best, weak = None, []
    for ob in obs:
        pools = [(ob, "", lev_whole)] + [(sg, f", segment {sg['segment'][0]:.0f}-{sg['segment'][1]:.0f} s", lev_seg)
                                         for sg in ob.get("segments", [])]
        for src, seg_lab, lev in pools:
            for p in src.get("significant", []):
                if p.get("related_to"):
                    continue
                if p["fap"] >= lev:
                    weak.append((p, ob, seg_lab))
                    continue
                better = (best is None or (best["segment"] and not seg_lab)
                          or (bool(best["segment"]) == bool(seg_lab) and p["fap"] < best["fap"]))
                if better:
                    fam_lab = ("; fundamental of the " + f"{p['P_peak']:.5g} s peak"
                               if p["family"]["note"] != "kept" else "")
                    best = dict(P=p["P_fit"], sigma_P=p["sigma_P"], P_str=p["P_str"], fap=p["fap"], Z=p["Z_peak"],
                                m=p["m_peak"], obs_id=ob["obs_id"], instrument=ob["instrument"], kind="blind",
                                pulsed_fraction=p["refine"]["pulsed_fraction"],
                                halves=p["halves"]["detected_both"], segment=seg_lab.strip(", "),
                                method=f"XMM {ob['instrument']} pipeline LC ({ob['band']}, {ob['dt']:.3g} s bins"
                                       f"{seg_lab}), Israel & Stella normalised power, {p['m_peak']}-harmonic sum"
                                       f"{fam_lab}")
    if best is None:
        for ob in obs:
            for src, seg_lab in [(ob, "")] + [(sg, f", segment {sg['segment'][0]:.0f}-{sg['segment'][1]:.0f} s")
                                              for sg in ob.get("segments", [])]:
                kn = src.get("known")
                if kn and kn.get("confirmed") and (best is None or kn["fap"] < best["fap"]):
                    r = kn["refine"]
                    best = dict(P=r["P"], sigma_P=r["sigma_P"], P_str=r["P_str"], fap=kn["fap"], Z=kn["Z"],
                                m=kn["m"], obs_id=ob["obs_id"], instrument=ob["instrument"], kind="targeted",
                                pulsed_fraction=r["pulsed_fraction"], halves=np.nan, segment=seg_lab.strip(", "),
                                method=f"XMM {ob['instrument']} pipeline LC{seg_lab}, targeted check of "
                                       f"literature spin (Israel & Stella normalised power)")
    res["best"] = best
    notes = []
    if not res.get("observations"):
        notes.append("no XMM pipeline light curve of the target found")
    elif best is None:
        notes.append("no significant pulsation in any observation")
        tops = [(p, ob) for ob in obs for p in ob.get("peaks", [])[:1]]
        if tops:
            p, ob = min(tops, key=lambda t: t[0]["fap"])
            notes.append(f"strongest peak {p['P']:.5g} s (m = {p['m']}, FAP {p['fap']:.2g}) in {ob['obs_id']}")
    for p, ob, seg_lab in weak:
        if best is None or not np.isclose(p["P_fit"], best["P"], rtol=1e-3):
            notes.append(f"weak candidate {p['P_str']} s in {ob['obs_id']}{seg_lab} (FAP {p['fap']:.2g}) not "
                         f"adopted: needs FAP < {fap_level:g}/{n_obs} observation(s)"
                         + (" /10 for a single segment" if seg_lab else ""))
    ks = res.get("known_spin")
    kn_obs = [ob.get("known") or {} for ob in obs]
    if ks is not None and np.isfinite(ks) and kn_obs and all("outside" in (k.get("note") or "") for k in kn_obs):
        notes.append(f"literature spin {ks:g} s is outside the period range these light curves can search "
                     f"(2 bins to 1/4 of the observation)")
    elif ks is not None and np.isfinite(ks):
        n_conf = sum(1 for ob in obs if (ob.get("known") or {}).get("confirmed"))
        n_seg = sum(1 for ob in obs if not (ob.get("known") or {}).get("confirmed")
                    and any((sg.get("known") or {}).get("confirmed") for sg in ob.get("segments", [])))
        notes.append(f"literature spin {ks:g} s confirmed in {n_conf} of {len(obs)} observation(s)"
                     + (f" (+{n_seg} only in {(seg_len or 10000) / 1000:g} ks segments)" if n_seg else ""))
    bad = [ob["obs_id"] for ob in res.get("observations", []) if "peaks" not in ob]
    if bad:
        notes.append("not analysed (too few good bins or read error): " + ", ".join(bad))
    if res.get("obs_without_target"):
        notes.append("no target light curve (source not in the pipeline source list) in "
                     + ", ".join(res["obs_without_target"]))
    elif "obs_without_target" not in res:           # results saved by an older version
        notes += [n for n in res.get("notes", []) if n.startswith("no target light curve")]
    res["notes"] = notes
    return best


def public_xmm_result(res):
    """The result without light-curve/spectrum objects and folded profiles (for saving)."""
    def clean(ob):
        ob = {k: v for k, v in ob.items() if not k.startswith("_")}
        ob["significant"] = [dict(p, refine={k: v for k, v in p["refine"].items() if k != "fold"})
                             if "refine" in p else p for p in ob.get("significant", [])]
        if ob.get("known") and "refine" in ob["known"]:
            ob["known"] = dict(ob["known"])
            ob["known"]["refine"] = {k: v for k, v in ob["known"]["refine"].items() if k != "fold"}
        if "segments" in ob:
            ob["segments"] = [clean(sg) for sg in ob["segments"]]
        return ob
    out = dict(res)
    out["observations"] = [clean(ob) for ob in res["observations"]]
    return out


def print_xmm_summary(res):
    print(f"\n=== {res['name']}: XMM spin search ({len(res['observations'])} observation(s)) ===")
    for ob in res["observations"]:
        if "note" in ob and "peaks" not in ob:
            print(f"  {ob['obs_id']}: {ob['note']}")
            continue
        print(f"  {ob['obs_id']} {ob['instrument']} {ob['band']} dt={ob['dt']:g}s T={ob['T']:.0f}s "
              f"<rate>={ob['mean_rate']:.3g}/s; blind z1 threshold {ob['threshold_z1']:.1f}")
        print("    top peaks: " + ", ".join(f"{p['P']:.5g} s (m={p['m']}, Z={p['Z']:.1f}, FAP={p['fap']:.2g})"
                                          for p in ob["peaks"][:4]))
        for p in ob["significant"]:
            rel = p.get("related_to")
            print(f"    {'related' if rel else 'DETECTION'} {p['P_str']} s (from peak {p['P_peak']:.5g} s, "
                  f"family: {p['family']['note']}), FAP {p['fap']:.2g}, pulsed fraction "
                  f"{p['refine']['pulsed_fraction']:.2f}, in both halves: {p['halves']['detected_both']}"
                  + (f"  [= {rel[1]} of {rel[0]:.6g} s]" if rel else ""))
        for sg in ob.get("segments", []):
            for p in sg.get("significant", []):
                if not p.get("related_to"):
                    print(f"    segment {sg['segment'][0]:.0f}-{sg['segment'][1]:.0f} s: {p['P_str']} s, "
                          f"FAP {p['fap']:.2g} (trials include all segments)")
        kn = ob.get("known")
        if kn:
            if "Z" in kn:
                print(f"    literature {kn['P_lit']:g} s: peak {kn['P_peak']:.5g} s, m={kn['m']}, Z={kn['Z']:.1f}, "
                      f"FAP={kn['fap']:.2g} -> " + ("confirmed " + kn["refine"]["P_str"] + " s" if kn["confirmed"] else "not confirmed"))
            else:
                print(f"    literature {kn['P_lit']:g} s: {kn.get('note', '')}")
    b = res.get("best")
    print("  RESULT: " + (f"P_spin = {b['P_str']} s ({b['method']}; obs {b['obs_id']})" if b else "no significant pulsation"))
    if res.get("notes"):
        print("  notes: " + "; ".join(res["notes"]))


def plot_observation(ob, name="", show=True):
    import matplotlib.pyplot as plt
    lc, spec = ob["_lc"], ob["_spec"]
    fig, ax = plt.subplots(3, 1, figsize=(11, 9))
    ax[0].plot(lc.t, lc.rate, lw=0.5, color="0.3")
    ax[0].set_xlabel("time (s)")
    ax[0].set_ylabel("rate (cts/s)")
    ax[0].set_title(f"{name} XMM {ob['obs_id']} {ob['instrument']} ({ob['band']}, {ob['dt']:g} s bins)")
    ax[1].plot(1 / spec.freq, spec.z, lw=0.5, color="k")
    ax[1].axhline(ob["threshold_z1"], ls="--", color="k", lw=0.8, label="blind 1%")
    ax[1].axhline(threshold_mean_norm(1, 0.01), ls=":", color="0.5", lw=0.8, label="known-period 1%")
    kn = ob.get("known")
    if kn:
        ax[1].axvline(kn["P_lit"], color="tab:green", ls="--", lw=0.8, label="literature")
    indep = [p for p in ob["significant"] if not p.get("related_to")]
    for p in ob["significant"]:
        ax[1].axvline(p["P_fit"], color="tab:red" if not p.get("related_to") else "tab:orange", lw=0.8)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("period (s)")
    ax[1].set_ylabel("z = power / local mean")
    ax[1].legend(fontsize=7)
    P_show = (indep[0]["P_fit"] if indep
              else (kn["P_peak"] if kn and "P_peak" in kn else ob["peaks"][0]["P"]))
    fo = fold_rate(lc, P_show)
    ph = np.concatenate([fo["phase"], fo["phase"] + 1])
    ax[2].errorbar(ph, np.tile(fo["mean"], 2), np.tile(fo["err"], 2), fmt="o", ms=3, color="k")
    ax[2].set_xlabel("phase")
    ax[2].set_ylabel("rate")
    ax[2].set_title(f"folded at {P_show:.5g} s (pulsed fraction {fo['pulsed_fraction']:.2f})", fontsize=9)
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def xmm_summary_row(res, ref=None):
    b = res.get("best") or {}
    row = dict(folder=res["name"], P_spin=b.get("P", np.nan), sigma_P_spin=b.get("sigma_P", np.nan),
               P_spin_str=b.get("P_str", ""), spin_FAP=b.get("fap", np.nan),
               spin_method=b.get("method", "no significant pulsation" if res.get("observations") else "no XMM light curve"),
               spin_obs=b.get("obs_id", ""), pulsed_fraction=b.get("pulsed_fraction", np.nan),
               spin_both_halves=b.get("halves", np.nan), n_xmm_obs=len(res.get("observations", [])),
               spin_notes="; ".join(res.get("notes", [])))
    kn_all = []
    for ob in res.get("observations", []):                 # whole observations and their segments
        for src in [ob] + list(ob.get("segments", [])):
            kn = src.get("known")
            if kn and "Z" in kn:
                kn_all.append(kn)
    if kn_all:
        kb = min(kn_all, key=lambda k: k["fap"])
        row.update(spin_lit_tested=kb["P_lit"], spin_lit_FAP=kb["fap"], spin_lit_confirmed=kb["confirmed"])
    if ref is not None:
        row.update(P_spin_lit=ref.get("P_spin_lit", np.nan), my_P_spin=ref.get("my_P_spin", np.nan))
    return row
