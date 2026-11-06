"""AX J1820.5-1434: strengths at 54.0, 55.6 and 111.2 d in time subsets, and the folded profile
at 111.23 d near maximum and half a cycle later (Sect. 4.4). Writes ax1820.json."""
import warnings

import numpy as np

import config  # noqa: F401 (puts hmxb/ on sys.path)
import bat_period_pipeline as bat
import common

warnings.filterwarnings("ignore")
P111 = 111.2299724384408
T0 = 53416.0


def main():
    dpath, _ = bat.find_bat_files(common.star_dir("AX J1820.5-1434"))
    lc = bat.load_bat_lc(dpath)
    med = np.median(lc.t)
    subsets = [("all", np.ones_like(lc.t, bool)), ("MJD < 56000", lc.t < 56000), ("MJD < 55200", lc.t < 55200),
               ("MJD >= 55200", lc.t >= 55200), ("first half", lc.t < med), ("second half", lc.t >= med)]
    out = {"subsets": []}
    for label, m in subsets:
        sub = lc.subset(m)
        pg = bat.compute_periodogram(sub, weighting="semi")
        per = 1 / pg.freq
        sel = (per > 20) & (per < 400)
        i = int(np.argmax(pg.strength[sel]))
        row = dict(subset=label, T=sub.T, n=len(sub.t), threshold=bat.threshold_median_norm(pg.n_trials, 0.01),
                   best_P=float(per[sel][i]), best_S=float(pg.strength[sel][i]))
        for P in (54.0, 55.61, P111):
            Pk, S, _ = pg.strength_at(P, halfwidth=0.5 / sub.T)
            row[f"S_{P:.1f}"] = S
            row[f"P_{P:.1f}"] = Pk
        out["subsets"].append(row)
        print(row)
    dy, _ = bat.effective_errors(lc.rate, lc.err, "semi")
    w = 1 / dy ** 2
    out["profiles"] = []
    for label, m in (("all", np.ones_like(lc.t, bool)), ("first half", lc.t < med), ("second half", lc.t >= med)):
        pr = bat.fold_profile(lc.t[m], lc.rate[m], w[m], P111, nbins=16, t0=T0, by_cycle=True)
        i = int(np.nanargmax(pr["mean"]))
        j = (i + 8) % 16
        win = lambda c: [(c + k) % 16 for k in (-1, 0, 1)]
        mx, ex = np.nanmean(pr["mean"][win(i)]), np.sqrt(np.nansum(pr["err"][win(i)] ** 2)) / 3
        mo, eo = np.nanmean(pr["mean"][win(j)]), np.sqrt(np.nansum(pr["err"][win(j)] ** 2)) / 3
        out["profiles"].append(dict(subset=label, max=mx, e_max=ex, opposite=mo, e_opposite=eo))
        print(label, f"max {mx:.3e}+/-{ex:.1e}  opposite {mo:.3e}+/-{eo:.1e}")
    obs, info = bat.detect_outbursts(lc)
    out["n_outbursts"] = len(obs)
    common.dump(out, "ax1820.json")


if __name__ == "__main__":
    main()
