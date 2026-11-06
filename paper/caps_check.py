"""Check of the candidate caps (Sect. 4.1): the pipeline keeps the eight strongest peaks per search
and vets at most five candidates per source. For every source with more than five peaks above the
blind threshold the candidate stage is rerun with 30 peaks and 30 candidates.
Slow (tens of minutes). Writes caps_check.json."""
import sys
import warnings

import numpy as np

import config  # noqa: F401 (puts hmxb/ on sys.path)
import bat_period_pipeline as bat
import common

warnings.filterwarnings("ignore")
_orig = bat.find_peaks


def _many(pg, n=8, **kw):
    return _orig(pg, n=30, **kw)


def main(names=None):
    bat.find_peaks = _many
    s = common.load_summary().set_index("Star (folder)")
    if not names:
        names = []
        for d in common.all_bat():
            raw = ([p for p in d.get("peaks_daily", []) if p["strength"] >= d["threshold_daily"]]
                   + [p for p in d.get("peaks_orbit", []) if p["strength"] >= d["threshold_orbit"]])
            if len(raw) > 5:
                names.append(d["name"])
    out = {}
    for n in names:
        Pl = s.loc[n, "Literature P_orb (d)"]
        res = bat.analyse_star(common.star_dir(n), known_period=Pl if np.isfinite(Pl) else None,
                               max_candidates=30, show=False, verbose=False)
        d0 = common.bat_json(n)
        q0 = sum(1 for c in d0.get("candidates", []) if not c.get("harmonic_of") and not c.get("instrumental")
                 and c["halves"]["detected_both"])
        cands = [dict(P=c["P"], S=c["strength"], fap=c["fap"], search=c["search"], related=c.get("harmonic_of"),
                      instrumental=c.get("instrumental"), both=c["halves"]["detected_both"]) for c in res["candidates"]]
        q1 = sum(1 for c in cands if not c["related"] and not c["instrumental"] and c["both"])
        b0 = (d0.get("best") or {}).get("P")
        b1 = (res.get("best") or {}).get("P")
        same = (b0 is None and b1 is None) or (b0 is not None and b1 is not None and abs(b0 - b1) < 1e-6 * b0)
        out[n] = dict(qualified_default=q0, qualified_enlarged=q1, adopted_unchanged=same,
                      candidates_default=len(d0.get("candidates", [])), candidates_enlarged=len(cands),
                      candidates=cands)
        print(f"{n:18s} qualified {q0} -> {q1}; adopted period unchanged: {same}", flush=True)
    keys = ("candidates_default", "candidates_enlarged", "qualified_default", "qualified_enlarged", "adopted_unchanged")
    out["_totals"] = dict(sources=len(names), **{k: int(sum(v[k] for v in out.values())) for k in keys})
    print(out["_totals"])
    common.dump(out, "caps_check.json")


if __name__ == "__main__":
    main(sys.argv[1:])
