"""Comparison of the BAT orbital periods with the catalogue of Fortin et al. (2023) (Sect. 4.2).

Groups: systems with documented orbital-period evolution (common.EVOLVING), Be/X-ray binaries,
and all other systems. The catalogue gives one uncertainty per period, treated as a symmetric
1-sigma error; values without an error are only tested against one frequency resolution element
(|dP| < P^2/T) and are excluded from the z statistics. Agreement is reported with the formal
errors and with our errors enlarged by 1.5 (Sect. 4.11).
Writes catalogue_comparison.csv and catalogue_comparison.json.
"""
import numpy as np
import pandas as pd

import common
import config

INFLATE = 1.5


def main():
    s = common.load_summary()
    w = common.load_catalogue()
    T = {d["name"]: d["T_days"] for d in common.all_bat()}
    det = s[s["P_orb (d)"].notna() & ~s["Star (folder)"].isin(common.REJECTED)].copy()
    det["ecat"] = det["Catalogue name"].map(lambda n: w["Period_err"].get(n, np.nan))
    det["T"] = det["Star (folder)"].map(T)

    def group(r):
        if r["Star (folder)"] in common.EVOLVING:
            return "documented period evolution"
        return "Be/X-ray binaries" if str(r["Class"]).startswith("Be") else "other systems"
    det["group"] = det.apply(group, axis=1)
    P, sP, Pc, ec = det["P_orb (d)"], det["sigma P_orb (d)"], det["Literature P_orb (d)"], det["ecat"]
    det["has_err"] = ec.notna() & (ec > 0)
    det["z"] = np.where(det.has_err, (P - Pc) / np.hypot(sP, ec.fillna(0)), np.nan)
    det["z_inflated"] = np.where(det.has_err, (P - Pc) / np.hypot(INFLATE * sP, ec.fillna(0)), np.nan)
    det["within_resolution"] = (P - Pc).abs() < P ** 2 / det["T"]
    det["frac"] = ((P - Pc) / Pc).abs()
    det["sigma_ratio"] = np.where(det.has_err, ec / sP, np.nan)
    out = []
    for g in ("other systems", "Be/X-ray binaries", "documented period evolution", "all"):
        d = det if g == "all" else det[det.group == g]
        e = d[d.has_err]
        out.append(dict(group=g, N=len(d), N_with_cat_error=len(e),
                        within_3sigma=int((e.z.abs() < 3).sum()),
                        within_3sigma_inflated=int((e.z_inflated.abs() < 3).sum()),
                        no_cat_error=int(len(d) - len(e)),
                        no_cat_error_within_resolution=int(d[~d.has_err].within_resolution.sum()),
                        median_frac_diff=float(d.frac.median()), median_abs_z=float(e.z.abs().median()),
                        cat_error_larger_than_formal=int((e.sigma_ratio > 1).sum()),
                        cat_error_larger_than_inflated=int((e.sigma_ratio > INFLATE).sum())))
    det.to_csv(config.OUT / "catalogue_comparison.csv", index=False)
    common.dump(out, "catalogue_comparison.json")
    print(pd.DataFrame(out).to_string(index=False))
    print("\nbeyond 3 sigma (formal):")
    print(det[det.z.abs() >= 3][["Star (folder)", "group", "z", "z_inflated"]].to_string(index=False))


if __name__ == "__main__":
    main()
