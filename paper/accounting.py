"""Accounting of the orbital-period search (Tables 1 and 2 of the paper).

Stages, per source and per candidate peak:
  1. peaks above the analytic 1 % blind threshold (eight strongest local maxima per search)
  2. candidates after the red-noise simulations (borderline S < 30) and after merging peaks found
     in both the daily and the orbit search (at most five vetted per source)
  3. vetting: related to a stronger peak / instrumental / not in both halves / qualified
  4. adoption (blind, targeted, none), rejection on physical grounds
Writes accounting.json and candidates.csv to config.OUT.
"""
import numpy as np
import pandas as pd

import common
import config


def main():
    s = common.load_summary().set_index("Star (folder)")
    rows, cand_rows = [], []
    for d in common.all_bat():
        n = d["name"]
        lit = bool(np.isfinite(s.loc[n, "Literature P_orb (d)"]))
        raw = ([p for p in d.get("peaks_daily", []) if p["strength"] >= d["threshold_daily"]]
               + [p for p in d.get("peaks_orbit", []) if p["strength"] >= d["threshold_orbit"]])
        cands = d.get("candidates", [])
        failed = d.get("failed_simulation", [])
        best = d.get("best") or {}
        kind = best.get("kind") if best else None
        known = d.get("known") or {}
        testable = not ("cannot be tested" in str(known.get("note", "")))
        for c in cands:
            if c.get("harmonic_of"):
                stage = "related"
            elif c.get("instrumental"):
                stage = "instrumental"
            elif not c["halves"]["detected_both"]:
                stage = "not in both halves"
            else:
                stage = "qualified"
            adopted = bool(best) and abs(c["P"] - best["P"]) < 1e-9 * best["P"] and kind == "blind"
            cand_rows.append(dict(source=n, P=c["P"], S=c["strength"], fap=c["fap"], search=c["search"],
                                  stage=stage, relation=str(c.get("harmonic_of") or ""),
                                  instrumental=c.get("instrumental") or "", adopted_orbital=adopted))
        rows.append(dict(source=n, cls=common.class_group(s.loc[n, "Class"]), lit=lit, testable=testable,
                         n_raw=len(raw), n_failsim=len(failed), n_cand=len(cands), kind=kind,
                         rejected=n in common.REJECTED, known_confirmed=bool(known.get("confirmed"))))
    df = pd.DataFrame(rows)
    cd = pd.DataFrame(cand_rows)
    cd.to_csv(config.OUT / "candidates.csv", index=False)

    flow = dict(
        sources=len(df), searches=int(2 * len(df)),
        sources_with_peaks_above_threshold=int((df.n_raw > 0).sum()),
        peaks_above_threshold=int(df.n_raw.sum()),
        sources_only_failing_simulations=int(((df.n_cand == 0) & (df.n_failsim > 0)).sum()),
        peaks_failing_simulations=int(df.n_failsim.sum()),
        sources_with_candidates=int((df.n_cand > 0).sum()), candidates=int(len(cd)),
        candidates_related=int((cd.stage == "related").sum()),
        candidates_instrumental=int((cd.stage == "instrumental").sum()),
        sources_instrumental=int(cd[cd.stage == "instrumental"].source.nunique()),
        candidates_not_both_halves=int((cd.stage == "not in both halves").sum()),
        sources_not_both_halves=int(cd[cd.stage == "not in both halves"].source.nunique()),
        candidates_qualified=int((cd.stage == "qualified").sum()),
        sources_qualified=int(cd[cd.stage == "qualified"].source.nunique()),
        blind_adopted=int((df.kind == "blind").sum()),
        blind_adopted_not_both_halves=int(((cd.stage != "qualified") & cd.adopted_orbital).sum()),
        blind_rejected=int((df.rejected & (df.kind == "blind")).sum()),
        blind_retained=int(((df.kind == "blind") & ~df.rejected).sum()),
        qualified_not_orbital=int(((cd.stage == "qualified") & ~cd.adopted_orbital).sum()),
        targeted=int((df.kind == "targeted").sum()),
        catalogue_periods=int(df.lit.sum()),
        catalogue_not_testable=int((df.lit & ~df.testable).sum()),
        catalogue_not_recovered=int((df.lit & df.testable & df.kind.isna()).sum()),
        no_catalogue_period=int((~df.lit).sum()),
        no_catalogue_with_peaks=int(((~df.lit) & (df.n_raw > 0)).sum()),
    )
    by = []
    for g in ("Be", "sg", "SFXT", "gamma", "other", "unclassified"):
        x = df[df.cls == g]
        by.append(dict(cls=g, N=len(x), no_cat=int((~x.lit).sum()), cat=int(x.lit.sum()),
                       blind_retained=int(((x.kind == "blind") & ~x.rejected).sum()),
                       targeted=int((x.kind == "targeted").sum()),
                       not_recovered=int((x.lit & x.testable & x.kind.isna()).sum()),
                       not_testable=int((x.lit & ~x.testable).sum()),
                       blind_rejected=int((x.rejected & (x.kind == "blind")).sum())))
    common.dump(dict(flow=flow, by_class=by), "accounting.json")
    for k, v in flow.items():
        print(f"{k:40s} {v}")
    print(pd.DataFrame(by).to_string(index=False))
    # consistency checks
    assert flow["blind_retained"] + flow["blind_rejected"] == flow["blind_adopted"]
    assert (flow["blind_retained"] + flow["targeted"] + flow["catalogue_not_recovered"]
            + flow["catalogue_not_testable"]) == flow["catalogue_periods"]
    assert flow["catalogue_periods"] + flow["no_catalogue_period"] == flow["sources"]


if __name__ == "__main__":
    main()
