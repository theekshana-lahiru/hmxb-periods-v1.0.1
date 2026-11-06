"""Writes run_pipeline.ipynb (example notebook for the HMXB period pipelines)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = []

cells.append(md(r"""# HMXB period pipeline: worked examples

This notebook runs the corrected analysis on a few stars and shows how to regenerate the whole
results workbook. The code lives in four files, which must sit in the same folder
(here `D:\HMXBs\code`):

| file | what it does |
|---|---|
| `hmxb_common.py` | shared tools: harmonic fit with Δχ² = 1 errors, rounding, false-alarm probabilities, catalogue/“my results” loader |
| `bat_period_pipeline.py` | Swift/BAT orbital periods (steps 1-6 below) |
| `xmm_spin_pipeline.py` | XMM-Newton spin periods (Israel & Stella normalised power spectrum) |
| `build_results_sheet.py` | runs both on every star and writes `HMXB pipeline results <date>.xlsx` |

**BAT steps.** 1) daily light curve for P > 5 d, orbit light curve below; DATA_FLAG = 0; only absurd
errors removed, bright points never clipped. 2) Lomb-Scargle of the raw rates with errors
(semi-weighting: the source's own scatter is added to each error). 3) Strength S = power / local
red-noise level; blind 1 % threshold ≈ 18 (daily) or ≈ 22 (orbit light curve), S ≥ 6.6 confirms a
known period; optional red-noise simulations. 4) Family check: P/2, P, 2P, the 1-day alias
1/(1 − f) and both halves of the data. 5) Outburst timing for Be transients. 6) Harmonic fit,
errors rescaled to χ²_red = 1, 1σ at Δχ² = 1, period rounded to the error's decimal place.

Needs: numpy, scipy, astropy, matplotlib, pandas, openpyxl (all in Anaconda)."""))

cells.append(code(r"""import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
%matplotlib inline

ROOT = Path(r"D:\HMXBs")            # folder with Stars\ and HMXBwebcat_latest.xlsx
CODE = ROOT / "code"                # folder with the four .py files
STARS = ROOT / "Stars"
OUT = ROOT / "pipeline_output"      # figures and JSON files go here
sys.path.insert(0, str(CODE))

import hmxb_common as hc
import bat_period_pipeline as bat
import xmm_spin_pipeline as xmm

# literature values (HMXBwebcat) and your previous values (my results.xlsx, read only)
ref = hc.load_references(ROOT / "HMXBwebcat_latest.xlsx", STARS / "my results.xlsx", STARS)
ref[["main_id", "P_orb_lit", "P_spin_lit", "my_P_orb", "my_P_spin"]].head(10)"""))

cells.append(md(r"""## 1. One star, Swift/BAT orbital period

`known_period` is optional: when given, the literature period is also tested on its own (only
S ≥ 6.6 is needed for that, because it is a single trial). The result dictionary holds every
intermediate step; `print_summary` and the figure show the essentials."""))

cells.append(code(r"""star = "IGR J00370+6122"
r = ref.loc[star]
res = bat.analyse_star(STARS / star, known_period=r.P_orb_lit, known_period_err=r.P_orb_lit_err,
                       out_dir=OUT / "BAT")
best = res["best"]
print("\nadopted:", best["P_str"], "d  |", best["method"])"""))

cells.append(md(r"""## 2. A short orbital period: the orbit light curve breaks the 1-day alias

Daily bins cannot tell f from 1 − f (Cen X-3: 2.087 d and 1.920 d look identical). For P < 5 d
the pipeline uses the `*.orbit.lc.fits` file, where the alias is resolved."""))

cells.append(code(r"""star = "V779 Cen"            # Cen X-3
r = ref.loc[star]
res_cen = bat.analyse_star(STARS / star, known_period=r.P_orb_lit, known_period_err=r.P_orb_lit_err,
                           out_dir=OUT / "BAT")"""))

cells.append(md(r"""## 3. A Be transient: periodogram and outburst timing

For transients with a few outbursts the periodogram is weak; the pipeline also fits each
outburst peak and a straight-line ephemeris T_n = T0 + n P (see the O − C panel)."""))

cells.append(code(r"""star = "V850 Cen"            # GX 304-1
r = ref.loc[star]
res_gx = bat.analyse_star(STARS / star, known_period=r.P_orb_lit, known_period_err=r.P_orb_lit_err,
                          out_dir=OUT / "BAT")
print(res_gx.get("best_outburst"))"""))

cells.append(md(r"""## 4. Calibrating a borderline detection with red-noise simulations

The analytic threshold assumes the strengths are independent; simulated red-noise light curves
with the same sampling, errors and noise slope give a calibrated false-alarm probability
(`fap_sim`). 200 simulations take a minute or two."""))

cells.append(code(r"""star = "AX J1820.5-1434"     # literature 54 d; the pipeline finds ~111 d
r = ref.loc[star]
res_ax = bat.analyse_star(STARS / star, known_period=r.P_orb_lit, known_period_err=r.P_orb_lit_err,
                          n_sim=200, show=False)
print(res_ax["simulation"])"""))

cells.append(md(r"""## 5. XMM-Newton spin period

The pipeline finds the target's PPS light curve by position (PN preferred), bins it to ≥ 0.25 s
(a whole number of pipeline bins), and computes the power spectrum normalised by the local mean
power (Israel & Stella 1996), with harmonic sums for peaked profiles. A family check makes sure
the adopted period is the fundamental and not a harmonic (Vela X-1's strongest peak is P/4)."""))

cells.append(code(r"""star = "GP Vel"              # Vela X-1
r = ref.loc[star]
res_vela = xmm.analyse_xmm_star(STARS / star, ra=r.ra, dec=r.dec, known_spin=r.P_spin_lit,
                                known_spin_err=r.P_spin_lit_err, min_bin=0.25, out_dir=OUT / "XMM")"""))

cells.append(md(r"""## 6. Fast pulsar in a tight orbit: search in segments

Cen X-3's 4.8 s pulse is Doppler-shifted by its 2.09 d orbit, so over a whole 70-80 ks observation
the power is smeared over many frequency bins and disappears. Searching 10 ks segments (the
trials are multiplied by the number of segments) recovers it, and the period changes from segment
to segment as the neutron star moves along its orbit."""))

cells.append(code(r"""star = "V779 Cen"
r = ref.loc[star]
res_cen_x = xmm.analyse_xmm_star(STARS / star, ra=r.ra, dec=r.dec, known_spin=r.P_spin_lit,
                                 known_spin_err=r.P_spin_lit_err, min_bin=0.25,
                                 segment_length=10000, show=False)"""))

cells.append(md(r"""## 7. Regenerate the whole results workbook

This runs every star (BAT and XMM) and writes `D:\HMXBs\HMXB pipeline results <date>.xlsx`
with the sheets Summary, Orbital (BAT), Spin (XMM), Peaks and README. `my results.xlsx` is never
changed. A full run takes roughly 30-60 minutes; from a command prompt you can use several
processes:

```
cd D:\HMXBs\code
python build_results_sheet.py --root "D:\HMXBs" --workers 4
```

Inside Jupyter keep `--workers 1` (Windows cannot start worker processes from a notebook cell
reliably). Use `--only` to rerun a few stars."""))

cells.append(code(r"""import build_results_sheet as brs
xlsx, sheets = brs.main(["--root", str(ROOT), "--only", "IGR J00370+6122", "V779 Cen", "--workers", "1"])
sheets["Summary"]"""))

cells.append(md(r"""## Reading the results

* **P (rounded)** is the value to quote: the period rounded to the decimal place of its 1σ error.
* **Strength / FAP**: S below the blind threshold (or FAP above 0.01) is not a detection, however
  sharp the peak looks.
* **Halves consistent = False** can be physics (orbital decay of Cen X-3, spin changes of
  accreting pulsars) or a warning that the signal is not stable.
* **related_to** in the Peaks sheet marks harmonics, sub-harmonics and beat periods of a stronger
  detection; they are not new periods.
* **No significant periodicity** means the data cannot show it at the 1 % level, not that the
  period does not exist."""))

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.write(nb, "run_pipeline.ipynb")
print("written run_pipeline.ipynb")
