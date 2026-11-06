r"""
build_results_sheet.py
======================
Runs the corrected period pipelines on every star and writes a NEW results workbook

    <root>\HMXB pipeline results <YYYY-MM-DD>.xlsx

"my results.xlsx" is only read (for your previous values), never written.

Sheets
  Summary        one row per star: orbital period (Swift/BAT), spin period (XMM-Newton), the
                 literature values, your previous values and notes
  Orbital (BAT)  every column of the BAT analysis (thresholds, top peaks, outburst timing ...)
  Spin (XMM)     one row per XMM observation (light curve used, peaks, detection, literature check)
  Peaks          every significant BAT candidate and XMM detection, with harmonics/aliases marked
  README         what the columns mean

Usage (Anaconda Prompt, in the folder that holds the four .py files):
    python build_results_sheet.py --root "D:\HMXBs"
    python build_results_sheet.py --root "D:\HMXBs" --only "IGR J00370+6122" "V779 Cen"
    python build_results_sheet.py --root "D:\HMXBs" --workers 4          (parallel, faster)
Options: --no-bat, --no-xmm, --sim N (red-noise simulations for borderline BAT detections,
default 200; 0 = off), --out <folder> for the figures and JSON files (default
<root>\pipeline_output), --xlsx <file> to choose the workbook name.

From Jupyter:
    import build_results_sheet as brs
    brs.main(["--root", r"D:\HMXBs", "--only", "IGR J00370+6122"])
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import hmxb_common as hc  # noqa: E402

# Positions to use instead of the catalogue (folder name -> ra, dec, reason)
COORD_OVERRIDES = {
    "AX J1838.0-0655": (279.51304, -6.92594,
                        "catalogue gives the old ASCA position (87\" off); the XMM source is PSR J1838-0655"),
}
# Known problems in catalogue entries, shown in the notes (folder name -> text)
CATALOGUE_NOTES = {
    "V779 Cen": "catalogue P_orb is the low-precision 2.033 +/- 0.029 d; the orbit of Cen X-3 is 2.087 d",
    "V1521 Cyg": "the orbital period of Cyg X-3 increases with time (Pdot/P ~ 1e-6 per year), so a catalogue "
                 "value from an earlier epoch differs slightly from the BAT-era period",
}

BAT_DEFAULTS = dict(weighting="semi", fap_level=0.01)
XMM_DEFAULTS = dict(band="total", min_bin=0.25, segment_length=10000.0, nharm_max=4, fap_level=0.01)
BORDERLINE_S = 30.0          # BAT blind detections below this strength get red-noise simulations


# ======================================================================================
# Workers (run in separate processes)
# ======================================================================================

def _fmt_rel(rel):
    if not rel:
        return ""
    P_par, kind = rel[0], rel[1]
    return f"{kind} of {float(P_par):.6g}"


def _f(x):
    """float, with None (JSON null) and non-numbers as NaN."""
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _clean(d):
    return {k: (np.nan if v is None else v) for k, v in d.items()}


def _stem(name):
    return name.replace(" ", "_").replace("/", "_")


def bat_post(name, r):
    """Rows for the workbook from a BAT result (live, or loaded from its JSON file)."""
    import bat_period_pipeline as bat
    row = _clean(bat.summary_row(r))
    b = r.get("best") or {}
    cands = []
    for e in r.get("candidates", []):
        cands.append(dict(
            star=name, data=f"Swift/BAT {e['lc_used']} LC", search=e["search"] + " search",
            peak_period=_f(e["P_peak"]), period=_f(e["P"]), sigma=_f(e["sigma_P"]), period_str=e["P_str"],
            unit="d", strength_or_Z=_f(e["strength"]), FAP=_f(e["fap"]), FAP_simulated=_f(e.get("fap_sim")),
            fundamental_check=e["family"]["note"], alias=(e.get("alias") or {}).get("note", ""),
            consistent_in_halves=e["halves"]["consistent"], detected_in_both_halves=e["halves"]["detected_both"],
            related_to=_fmt_rel(e.get("harmonic_of")), instrumental=e.get("instrumental", ""),
            adopted=bool(b and b.get("kind") == "blind" and np.isclose(_f(e["P"]), _f(b.get("P")), rtol=1e-12))))
    k = r.get("known")
    if k:
        m = k.get("measure") or {}
        cands.append(dict(
            star=name, data=f"Swift/BAT {k.get('lc_used', '')} LC", search="targeted (literature)",
            peak_period=_f(k.get("P_peak")), period=_f(m.get("P")), sigma=_f(m.get("sigma_P")),
            period_str=m.get("P_str", ""), unit="d", strength_or_Z=_f(k.get("strength")), FAP=_f(k.get("fap")),
            FAP_simulated=np.nan, fundamental_check=(k.get("family") or {}).get("note", ""), alias="",
            consistent_in_halves=np.nan, detected_in_both_halves=np.nan,
            related_to=(f"literature {k['P_lit']:g} d " + ("confirmed" if k["confirmed"] else "not confirmed")
                        + (f" ({k['note']})" if k.get("note") else "")),
            instrumental="", adopted=bool(b and b.get("kind") == "targeted")))
    return dict(folder=name, ok=True, row=row, cands=cands, sim=r.get("simulation") or r.get("simulation_orbit"),
                best_kind=b.get("kind"), best_S=_f(b.get("strength")), best_lc=b.get("lc_used"))


def bat_job(folder, known_P, known_P_err, fig_dir, kwargs):
    import matplotlib
    matplotlib.use("Agg")
    warnings.filterwarnings("ignore")
    import bat_period_pipeline as bat
    t0 = time.time()
    name = Path(folder).name
    try:
        r = bat.analyse_star(folder, known_period=known_P, known_period_err=known_P_err,
                             out_dir=fig_dir, show=False, verbose=False, **kwargs)
        out = bat_post(name, hc.to_jsonable(bat.public_result(r)))
        out["seconds"] = time.time() - t0
        return out
    except Exception as ex:
        return dict(folder=name, ok=False, error=f"{type(ex).__name__}: {ex}",
                    tb=traceback.format_exc(), seconds=time.time() - t0)


def xmm_post(name, r):
    """Rows for the workbook from an XMM result (live, or loaded from its JSON file)."""
    import xmm_spin_pipeline as xmm
    if "observations" in r and r["observations"] is not None:
        xmm.choose_best_spin(r, r.get("fap_level") or 0.01)     # current adoption rule
    row = _clean(xmm.xmm_summary_row(r))
    obs_rows, peaks = [], []
    best = r.get("best") or {}
    for ob in r["observations"]:
        if "peaks" not in ob:
            obs_rows.append(dict(star=name, obs_id=ob.get("obs_id", ""), file=Path(ob.get("path", "") or "").name,
                                 note=ob.get("note", "")))
            continue
        kn = ob.get("known") or {}
        indep = [p for p in ob["significant"] if not p.get("related_to")]
        seg_det = [(sg, p) for sg in ob.get("segments", []) for p in sg["significant"] if not p.get("related_to")]
        obs_rows.append(dict(
            star=name, obs_id=ob["obs_id"], instrument=ob["instrument"], file=Path(ob["path"]).name,
            band=ob["band"], bin_s=_f(ob["dt"]), length_s=_f(ob["T"]), n_bins=ob["n_bins"],
            mean_rate=_f(ob["mean_rate"]), offset_arcsec=_f(ob.get("sep_arcsec")),
            blind_z1_threshold=_f(ob["threshold_z1"]),
            top_peaks=", ".join(f"{_f(p['P']):.5g} s (m={p['m']}, Z={_f(p['Z']):.1f}, FAP={_f(p['fap']):.2g})"
                                for p in ob["peaks"][:3]),
            detection=", ".join(f"{p['P_str']} s (FAP {_f(p['fap']):.2g})" for p in indep),
            detection_FAP=(_f(indep[0]["fap"]) if indep else np.nan),
            pulsed_fraction=(_f(indep[0]["refine"]["pulsed_fraction"]) if indep else np.nan),
            detected_in_both_halves=(indep[0]["halves"]["detected_both"] if indep else np.nan),
            segment_detections="; ".join(f"{_f(sg['segment'][0]):.0f}-{_f(sg['segment'][1]):.0f} s: {p['P_str']} s "
                                         f"(FAP {_f(p['fap']):.2g})" for sg, p in seg_det[:6]),
            literature_spin=_f(kn.get("P_lit")), literature_peak=_f(kn.get("P_peak")),
            literature_Z=_f(kn.get("Z")), literature_FAP=_f(kn.get("fap")),
            literature_confirmed=kn.get("confirmed", np.nan) if kn else np.nan,
            literature_note=kn.get("note", ""),
            literature_measured=((kn.get("refine") or {}).get("P_str", "") if kn else "")))
        pools = [(ob, "")] + [(sg, f" segment {_f(sg['segment'][0]):.0f}-{_f(sg['segment'][1]):.0f} s")
                              for sg in ob.get("segments", [])]
        for src, lab in pools:
            for p in src["significant"]:
                peaks.append(dict(
                    star=name, data=f"XMM {ob['instrument']} {ob['obs_id']}{lab}", search="blind",
                    peak_period=_f(p["P_peak"]), period=_f(p["P_fit"]), sigma=_f(p["sigma_P"]),
                    period_str=p["P_str"], unit="s", strength_or_Z=_f(p["Z_peak"]), FAP=_f(p["fap"]),
                    FAP_simulated=np.nan, fundamental_check=p["family"]["note"], alias="",
                    consistent_in_halves=np.nan, detected_in_both_halves=p["halves"]["detected_both"],
                    related_to=_fmt_rel(p.get("related_to")), instrumental="",
                    adopted=bool(best and best.get("obs_id") == ob["obs_id"]
                                 and np.isclose(_f(p["P_fit"]), _f(best.get("P")), rtol=1e-12))))
    return dict(folder=name, ok=True, row=row, obs_rows=obs_rows, peaks=peaks)


def xmm_job(folder, ra, dec, known_spin, known_spin_err, fig_dir, kwargs):
    import matplotlib
    matplotlib.use("Agg")
    warnings.filterwarnings("ignore")
    import xmm_spin_pipeline as xmm
    t0 = time.time()
    name = Path(folder).name
    try:
        r = xmm.analyse_xmm_star(folder, ra=ra, dec=dec, known_spin=known_spin,
                                 known_spin_err=known_spin_err, out_dir=fig_dir, show=False,
                                 verbose=False, **kwargs)
        out = xmm_post(name, hc.to_jsonable(xmm.public_xmm_result(r)))
        out["seconds"] = time.time() - t0
        return out
    except Exception as ex:
        return dict(folder=name, ok=False, error=f"{type(ex).__name__}: {ex}",
                    tb=traceback.format_exc(), seconds=time.time() - t0)


def load_saved(kind, name, fig_dir):
    """Rebuild a job result from the JSON file a previous run saved (for --reuse)."""
    p = Path(fig_dir) / f"{_stem(name)}_{'BAT' if kind == 'bat' else 'XMM'}.json"
    if not p.exists():
        return None
    with open(p) as fh:
        r = json.load(fh)
    out = bat_post(name, r) if kind == "bat" else xmm_post(name, r)
    out["seconds"] = 0.0
    out["reused"] = True
    return out


# ======================================================================================
# Helpers
# ======================================================================================

def _to_float(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _decimals(x):
    """Number of decimals in a value as typed (8 -> 0, 41.03 -> 2)."""
    if isinstance(x, str):
        s = x.strip()
    else:
        v = _to_float(x)
        if not np.isfinite(v):
            return None
        s = repr(v)
        if v.is_integer():
            return 0
    m = re.match(r"^-?\d+(?:\.(\d+))?$", s)
    return len(m.group(1)) if (m and m.group(1)) else (0 if m else None)


def compare(P, sP, P_ref, T, daily=False, others=()):
    """How a new period P (+/- sP) relates to a reference value (e.g. your previous value).

    others: (P, sigma, label) of other detected but not adopted periods of the same star."""
    v = _to_float(P_ref)
    if not (np.isfinite(v) and v > 0):
        return ""
    if np.isfinite(P) and np.isfinite(T):
        sP = sP if np.isfinite(sP) else 0.0
        res = max(3.0 * sP, P ** 2 / T)                # 3 sigma or one frequency resolution element
        if abs(P - v) <= res:
            return "same"
        d = _decimals(P_ref)
        prec = 0.5 * 10.0 ** (-d) if d is not None else 0.0     # rounding of the value as you typed it
        if abs(P - v) <= prec:
            return "same, but your value is less precise"
        for k in (2, 3, 4, 5):
            if abs(P * k - v) <= max(k * res, prec):
                return f"your value = {k} x new (sub-harmonic)"
            if abs(P - k * v) <= max(res, k * prec):
                return f"your value = new / {k} (harmonic)"
        if daily:
            fa = abs(1.0 - 1.0 / v)
            if fa > 0 and abs(P - 1.0 / fa) <= res:
                return "your value is the 1-day alias"
    for Po, so, lab in others:
        if np.isfinite(Po) and np.isfinite(T) and abs(Po - v) <= max(3.0 * (so if np.isfinite(so) else 0.0), Po ** 2 / T):
            return f"your value matches another detected period ({lab}), not the adopted one"
    if not np.isfinite(P):
        return "not confirmed (no significant period found)"
    return f"different ({(v - P) / P:+.1%})"


def load_refs(root):
    stars = Path(root) / "Stars"
    ref = hc.load_references(Path(root) / "HMXBwebcat_latest.xlsx", stars / "my results.xlsx", stars)
    ref["coord_note"] = ""
    for f, (ra, dec, why) in COORD_OVERRIDES.items():
        if f in ref.index:
            ref.loc[f, ["ra", "dec"]] = [ra, dec]
            ref.loc[f, "coord_note"] = why
    return ref


def has_xmm(folder):
    return any("SRCTSR" in p.name.upper() for p in Path(folder).rglob("*") if p.is_file())


# ======================================================================================
# Main
# ======================================================================================

def run(root, out_dir=None, xlsx=None, only=None, workers=1, do_bat=True, do_xmm=True, n_sim=200,
        reuse=(), log=print):
    """reuse: ("bat", "xmm") parts to take from the JSON files of a previous run in out_dir
    instead of recomputing (stars without a saved file are computed)."""
    root = Path(root)
    stars_dir = root / "Stars"
    out_dir = Path(out_dir) if out_dir else root / "pipeline_output"
    fig_bat, fig_xmm = out_dir / "BAT", out_dir / "XMM"
    for d in (fig_bat, fig_xmm):
        d.mkdir(parents=True, exist_ok=True)
    ref = load_refs(root)
    folders = sorted(p for p in stars_dir.iterdir() if p.is_dir())
    if only:
        folders = [p for p in folders if p.name in set(only)]
    t_start = time.time()

    def pool_run(jobs, fn):
        out = []
        if workers <= 1:
            for j in jobs:
                r = fn(*j)
                out.append(r)
                log(f"  {r['folder']}: " + ("ok" if r["ok"] else "ERROR " + r["error"]) + f" ({r['seconds']:.0f} s)")
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(fn, *j): j for j in jobs}
                for fu in as_completed(futs):
                    r = fu.result()
                    out.append(r)
                    log(f"  {r['folder']}: " + ("ok" if r["ok"] else "ERROR " + r["error"]) + f" ({r['seconds']:.0f} s)")
        return {r["folder"]: r for r in out}

    # ---- Swift/BAT orbital periods -------------------------------------------------------
    bat_res = {}
    if do_bat:
        import bat_period_pipeline as bat
        jobs = []
        for f in folders:
            d, o = bat.find_bat_files(f)
            if d is None and o is None:
                continue
            kp = ref.loc[f.name, "P_orb_lit"] if f.name in ref.index else None
            kpe = ref.loc[f.name, "P_orb_lit_err"] if f.name in ref.index else None
            jobs.append((str(f), kp, kpe, str(fig_bat), dict(BAT_DEFAULTS)))
        if "bat" in reuse:
            for j in list(jobs):
                got = load_saved("bat", Path(j[0]).name, fig_bat)
                if got is not None:
                    bat_res[got["folder"]] = got
                    jobs.remove(j)
            log(f"Swift/BAT: {len(bat_res)} stars reused from saved results")
        log(f"Swift/BAT: {len(jobs)} stars to analyse")
        bat_res.update(pool_run(jobs, bat_job))
        if n_sim:
            def borderline(name, lc):
                r = bat_res.get(name, {})
                return (r.get("ok") and not r.get("sim") and r.get("best_kind") == "blind"
                        and r.get("best_lc") == lc and r.get("best_S", np.inf) < BORDERLINE_S)
            redo = [(a, b, c, d, dict(e, n_sim=n_sim)) for a, b, c, d, e in jobs
                    if borderline(Path(a).name, "daily")]
            redo += [(a, b, c, d, dict(e, n_sim_orbit=max(100, n_sim // 2))) for a, b, c, d, e in jobs
                     if borderline(Path(a).name, "orbit")]
            if redo:
                log(f"Swift/BAT: {len(redo)} borderline detections (S < {BORDERLINE_S:g}) -> red-noise simulations")
                bat_res.update(pool_run(redo, bat_job))

    # ---- XMM-Newton spin periods -------------------------------------------------------------
    xmm_res = {}
    if do_xmm:
        jobs = []
        for f in folders:
            if not has_xmm(f) or f.name not in ref.index:
                continue
            r0 = ref.loc[f.name]
            if not (np.isfinite(r0["ra"]) and np.isfinite(r0["dec"])):
                log(f"  {f.name}: no position, XMM skipped")
                continue
            jobs.append((str(f), float(r0["ra"]), float(r0["dec"]), r0["P_spin_lit"], r0["P_spin_lit_err"],
                         str(fig_xmm), dict(XMM_DEFAULTS)))
        if "xmm" in reuse:
            for j in list(jobs):
                got = load_saved("xmm", Path(j[0]).name, fig_xmm)
                if got is not None:
                    xmm_res[got["folder"]] = got
                    jobs.remove(j)
            log(f"XMM-Newton: {len(xmm_res)} stars reused from saved results")
        log(f"XMM-Newton: {len(jobs)} stars to analyse")
        xmm_res.update(pool_run(jobs, xmm_job))

    # ---- assemble the sheets ---------------------------------------------------------------
    sheets = assemble(folders, ref, bat_res, xmm_res)
    date = _dt.date.today().isoformat()
    xlsx = Path(xlsx) if xlsx else root / f"HMXB pipeline results {date}.xlsx"
    if xlsx.name.lower() == "my results.xlsx":
        raise ValueError("refusing to overwrite 'my results.xlsx'")
    write_workbook(xlsx, sheets, date)
    for name, df in sheets.items():
        df.to_csv(out_dir / f"{name.split(' ')[0]}.csv", index=False)
    errors = {k: v["error"] for k, v in {**bat_res, **xmm_res}.items() if not v.get("ok")}
    with open(out_dir / "errors.json", "w") as fh:
        json.dump(errors, fh, indent=1)
    log(f"done in {(time.time() - t_start) / 60:.1f} min -> {xlsx}")
    return xlsx, sheets


def assemble(folders, ref, bat_res, xmm_res):
    summary, bat_rows, xmm_obs, peaks = [], [], [], []
    for f in folders:
        name = f.name
        r0 = ref.loc[name] if name in ref.index else pd.Series(dtype=object)
        b = bat_res.get(name, {})
        x = xmm_res.get(name, {})
        br = b.get("row", {}) if b.get("ok") else {}
        xr = x.get("row", {}) if x.get("ok") else {}
        if b.get("ok"):
            row = dict(br)
            for c in ("main_id", "other_name", "cls", "P_orb_lit", "P_orb_lit_err", "my_P_orb"):
                row[c] = r0.get(c, np.nan)
            bat_rows.append(row)
            peaks += b["cands"]
        elif b:
            bat_rows.append(dict(folder=name, notes="ERROR " + b["error"]))
        if x.get("ok"):
            xmm_obs += x["obs_rows"]
            peaks += x["peaks"]
        T = br.get("T_days", np.nan)
        P, sP = br.get("P_orb", np.nan), br.get("sigma_P", np.nan)
        Ps, sPs = xr.get("P_spin", np.nan), xr.get("sigma_P_spin", np.nan)
        T_x = np.nan
        if x.get("ok") and xr.get("spin_obs"):
            Ls = [o.get("length_s", np.nan) for o in x["obs_rows"] if o.get("obs_id") == xr["spin_obs"]]
            T_x = Ls[0] if Ls else np.nan
        notes = []
        if name in CATALOGUE_NOTES:
            notes.append(CATALOGUE_NOTES[name])
        if isinstance(r0.get("coord_note", ""), str) and r0.get("coord_note", ""):
            notes.append("position: " + r0["coord_note"])
        if br.get("notes"):
            notes.append("BAT: " + br["notes"])
        if xr.get("spin_notes"):
            notes.append("XMM: " + xr["spin_notes"])
        if b and not b.get("ok"):
            notes.append("BAT ERROR: " + b["error"])
        if x and not x.get("ok"):
            notes.append("XMM ERROR: " + x["error"])
        summary.append({
            "Star (folder)": name,
            "Catalogue name": r0.get("main_id", ""),
            "Other name": r0.get("other_name", ""),
            "Class": r0.get("cls", ""),
            "P_orb (d)": P,
            "sigma P_orb (d)": sP,
            "P_orb (rounded)": br.get("P_orb_str", ""),
            "Strength S": br.get("strength", np.nan),
            "FAP (orbital)": br.get("FAP", np.nan),
            "Orbital method": br.get("method", "no BAT light curve" if not b else ""),
            "Halves consistent": br.get("halves_consistent", np.nan),
            "P_orb from outburst timing (d)": br.get("P_outburst_str", ""),
            "Literature P_orb (d)": r0.get("P_orb_lit", np.nan),
            "Literature check (S, FAP)": (f"S = {br['strength_at_lit']:.1f}, FAP = {br['FAP_at_lit']:.2g} -> "
                                          + ("confirmed" if br.get("lit_confirmed") is True else "not confirmed")
                                          if np.isfinite(_f(br.get("strength_at_lit")))
                                          else ("not testable (longer than the light curve)"
                                                if np.isfinite(_f(br.get("P_lit_tested"))) else "")),
            "New vs literature": br.get("relation_to_lit", ""),
            "Your P_orb (my results)": r0.get("my_P_orb", ""),
            "New vs your P_orb": compare(P, sP, r0.get("my_P_orb", np.nan), T, daily=True,
                                         others=[(c["period"], c["sigma"], f"{c['period_str']} d")
                                                 for c in b.get("cands", []) if not c["adopted"]
                                                 and c["search"].endswith("search")]) if np.isfinite(T) else "",
            "P_spin (s)": Ps,
            "sigma P_spin (s)": sPs,
            "P_spin (rounded)": xr.get("P_spin_str", ""),
            "FAP (spin)": xr.get("spin_FAP", np.nan),
            "Spin method": xr.get("spin_method", "" if not x else ""),
            "Pulsed fraction": xr.get("pulsed_fraction", np.nan),
            "XMM observations": xr.get("n_xmm_obs", np.nan) if x else np.nan,
            "Literature P_spin (s)": r0.get("P_spin_lit", np.nan),
            "Literature spin check": (((f"FAP = {xr['spin_lit_FAP']:.2g} -> "
                                        + ("confirmed" if xr.get("spin_lit_confirmed") is True else "not confirmed"))
                                       if np.isfinite(_f(xr.get("spin_lit_FAP"))) else "")
                                      + (f"; adopted spin differs by {(Ps - _f(r0.get('P_spin_lit'))) / _f(r0.get('P_spin_lit')):+.2%}"
                                         if np.isfinite(Ps) and np.isfinite(_f(r0.get("P_spin_lit")))
                                         and abs(Ps / _f(r0.get("P_spin_lit")) - 1) > 1e-3 else "")),
            "Your P_spin (my results)": r0.get("my_P_spin", ""),
            "New vs your P_spin": (compare(Ps, sPs, r0.get("my_P_spin", np.nan), T_x)
                                   if (x.get("ok") and _f(xr.get("n_xmm_obs")) > 0) else ""),
            "Notes": " | ".join(notes),
        })
    return {"Summary": pd.DataFrame(summary), "Orbital (BAT)": pd.DataFrame(bat_rows),
            "Spin (XMM)": pd.DataFrame(xmm_obs), "Peaks": pd.DataFrame(peaks),
            "README": readme_frame()}


def readme_frame():
    rows = [
        ("General", "Made by build_results_sheet.py (bat_period_pipeline.py, xmm_spin_pipeline.py, hmxb_common.py). "
                    "Your 'my results.xlsx' is only read, never changed."),
        ("Periods", "P and sigma_P are unrounded fit values; the 'rounded' columns give P rounded to the decimal "
                    "place of its 1-sigma error (error with 2 significant digits)."),
        ("Strength S", "BAT: Lomb-Scargle power divided by the local red-noise level (running median, +/-0.15 dex). "
                       "Single-trial false-alarm probability p = 2^-S. The blind 1% threshold depends on the number "
                       "of independent frequencies (about 18 for daily light curves, 22 for orbit light curves); "
                       "checking a known period needs only S >= 6.6."),
        ("FAP", "Global false-alarm probability (all trials). For BAT detections below S = 30 it is calibrated with "
                "200 simulated red-noise light curves where noted ('fap_sim')."),
        ("Orbital method", "Light curve used (daily for P > 5 d, orbit-resolution below), weighted Lomb-Scargle "
                           "(semi-weighting: intrinsic scatter added to the errors) and the number of harmonics in "
                           "the final fit. 'targeted' = the literature period was confirmed although no blind peak "
                           "passed the blind threshold."),
        ("Halves consistent", "The period measured separately in the first and second half of the data agrees "
                              "within 3 sigma."),
        ("Outburst timing", "Be transients: Gaussian fit to each outburst peak, straight-line ephemeris "
                            "T_n = T0 + n P; independent of the periodogram."),
        ("New vs literature / New vs your value", "same = within 3 sigma or one frequency resolution element "
                                                  "(P^2/T); 2 x, 1/2 x, 3 x, 1/3 x = harmonic relations; "
                                                  "1-day alias = 1/(1 - 1/P); 'less precise' = your value agrees "
                                                  "only to the number of decimals you gave."),
        ("Spin (XMM)", "PPS source light curves (PN preferred; timing-mode light curves are used when present), "
                       "binned to >= 0.25 s (a whole number of pipeline bins). Power spectrum at the independent "
                       "Fourier frequencies, normalised by the local mean power (Israel & Stella 1996), "
                       "harmonic sums of 1-4 harmonics, FAP for all trials. Each observation is also searched in "
                       "10 ks segments (trials multiplied by the number of segments + 1), which recovers fast "
                       "pulsars whose period is Doppler-smeared by the orbit (e.g. Cen X-3)."),
        ("Fundamental check (spin)", "A peak can be a harmonic (P/h) or sub-harmonic (k P) of the spin. The power "
                                     "at j/(kP) (j not a multiple of k) decides whether the profile repeats after "
                                     "P; harmonics and sub-harmonics of a stronger detection are marked "
                                     "'related_to' in the Peaks sheet and are not adopted."),
        ("Pulsed fraction", "(max - min) / (max + min) of the folded profile (16 bins)."),
        ("Literature check (spin)", "Targeted search within +/- 1/T (or 3 sigma of the literature error) of the "
                                    "catalogue spin, trials = number of frequencies in that window x harmonics."),
        ("Not detected", "No significant periodicity does not mean the period is absent: the source may be too "
                         "faint, the observation too short, or the period outside the searched range "
                         "(2 bins to T/4 for XMM; 1/T to 0.5 c/d (daily) or 0.2-6 c/d (orbit) for BAT)."),
    ]
    return pd.DataFrame(rows, columns=["Item", "Meaning"])


def write_workbook(path, sheets, date):
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name, index=False)
        wb = xw.book
        head = PatternFill("solid", fgColor="DDE6F0")
        for name, df in sheets.items():
            ws = wb[name]
            ws.freeze_panes = "B2" if name != "README" else "A2"
            for c in ws[1]:
                c.font = Font(bold=True)
                c.fill = head
                c.alignment = Alignment(wrap_text=True, vertical="top")
            for j, col in enumerate(df.columns, start=1):
                vals = [str(col)] + [("" if pd.isna(v) else str(v)) for v in df[col].head(300)]
                width = min(max(len(v) for v in vals) + 2, 70 if col in ("Notes", "Meaning", "notes") else 40)
                ws.column_dimensions[get_column_letter(j)].width = max(width, 8)
                if col in ("Notes", "Meaning", "notes", "spin_notes", "top_peaks", "segment_detections"):
                    for cell in ws.iter_cols(min_col=j, max_col=j, min_row=2):
                        for c in cell:
                            c.alignment = Alignment(wrap_text=True, vertical="top")
        wb["README"].cell(row=len(sheets["README"]) + 3, column=1, value=f"Generated {date}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Regenerate the HMXB results workbook")
    ap.add_argument("--root", required=True, help=r"folder that contains Stars\ and HMXBwebcat_latest.xlsx")
    ap.add_argument("--out", default=None, help="folder for figures/JSON (default <root>/pipeline_output)")
    ap.add_argument("--xlsx", default=None, help="output workbook (default <root>/HMXB pipeline results <date>.xlsx)")
    ap.add_argument("--only", nargs="*", default=None, help="star folder names to run")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--sim", type=int, default=200, help="red-noise simulations for borderline BAT detections")
    ap.add_argument("--no-bat", action="store_true")
    ap.add_argument("--no-xmm", action="store_true")
    ap.add_argument("--reuse", nargs="*", default=[], choices=["bat", "xmm"],
                    help="take these parts from the JSON files of a previous run (in --out) instead of "
                         "recomputing, e.g. --reuse xmm")
    a = ap.parse_args(argv)
    return run(a.root, out_dir=a.out, xlsx=a.xlsx, only=a.only, workers=a.workers,
               do_bat=not a.no_bat, do_xmm=not a.no_xmm, n_sim=a.sim, reuse=tuple(a.reuse))


if __name__ == "__main__":
    main()
