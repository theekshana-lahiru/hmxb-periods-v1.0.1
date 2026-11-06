# hmxb-periods v1.0.1

Code, results and input-file manifests for

> T. Piyadasa, *Orbital and spin periods of 102 high-mass X-ray binaries from Swift/BAT and
> XMM-Newton: a homogeneous re-analysis*, submitted to Astronomische Nachrichten (2026).

The package reproduces every period, significance level, table and figure of the paper from the public Swift/BAT transient-monitor light curves and XMM-Newton pipeline (PPS) light curves.

* Version: 1.0.1 (2026-11-02). Archived release: Zenodo DOI to be added after the first release (see `CITATION.cff`).
* License: MIT (`LICENSE`).
* Contact: Theekshana Piyadasa, Arthur C Clarke Institute for Modern Technologies, Sri Lanka.

## Contents

| Folder | What it contains |
|---|---|
| `hmxb/` | The analysis pipeline. `bat_period_pipeline.py` (orbital periods from Swift/BAT), `xmm_spin_pipeline.py` (spin periods from XMM-Newton), `hmxb_common.py` (significance, harmonic fits, catalogue matching), `build_results_sheet.py` (runs both on every source), `run_pipeline.ipynb` (worked example). |
| `paper/` | Scripts that turn the pipeline results into the numbers, tables and figures of the paper (`reproduce_all.py` runs them in order). |
| `results/` | Pipeline output used in the paper: one JSON file per source and instrument (`BAT/`, `XMM/`) with every peak, candidate, vetting step and fit, and the summary tables `Summary.csv`, `Orbital.csv`, `Spin.csv`, `Peaks.csv` (column definitions in `README.csv`). |
| `paper_output/` | Output of `paper/reproduce_all.py` for v1.0.1 (tables in LaTeX, figures, JSON summaries). |
| `data/` | Manifests of the 329 input files with SHA-256 checksums, product versions and dates; `source_names.csv` (folder name to catalogue name); `verify_inputs.py`; `make_data_snapshot.py`. |

## Requirements

Python 3.10 or later with NumPy, SciPy, Astropy, pandas, Matplotlib and openpyxl (`pip install -r requirements.txt`, or `conda env create -f environment.yml`). The results in the paper were produced with Python 3.13.17, NumPy 2.5.3, SciPy 1.18.1, Astropy 8.0.2, pandas 2.3.3, Matplotlib 3.11.2 and openpyxl 3.1.5. The red-noise simulations use a fixed random seed, so a rerun gives identical numbers.

## Input data

The scripts expect a data folder (environment variable `HMXB_DATA`, default `D:\HMXBs`) with

```
HMXB_DATA/
  HMXBwebcat_latest.xlsx                      catalogue of Fortin et al. (2023)
  Stars/<source>/swift/<name>.lc.fits         BAT transient monitor, daily bins
  Stars/<source>/swift/<name>.orbit.lc.fits   BAT transient monitor, Swift-orbit bins
  Stars/<source>/XMM Newtonian/<ObsID>/pps/P<ObsID><cam>X000SRCTSR<n>.FTZ   PPS source time series
```

The exact files are listed in `data/manifest_*.csv` (file name, size, SHA-256, object, dates).

* **Swift/BAT**: 204 light curves (102 daily, 102 orbit) of the BAT Hard X-ray Transient Monitor (Krimm et al. 2013), https://swift.gsfc.nasa.gov/results/transients/ . All files carry the FITS creation date 2026-02-01 and were downloaded between 2026-02-25 and 2026-03-04; they end at MJD 61071 (2026 January 31). The monitor products are regenerated as new data arrive, so files downloaded later contain more data and differ from these.
* **XMM-Newton**: 124 PPS source time series (121 EPIC-pn, 3 EPIC-MOS2) of 41 sources from the XMM-Newton Science Archive, https://nxsa.esac.esa.int . 105 were produced by PPS version 17.56 (SAS 18), the others by PPS 18.02 to 21.24 (SAS 19 to 21); `manifest_xmm_pps_lightcurves.csv` lists the ObsID, camera, mode, bin size, PPS and SAS versions and the file date of each.

`python data/verify_inputs.py <data folder>` checks every file against the manifests.
`python data/make_data_snapshot.py <data folder> <zip>` packs the exact input files into one archive
(about 1.2 GB) that can be deposited next to the code, so that the analysis can be repeated on the same data even after the online products have been updated.

## Reproducing the paper

1. Pipeline (all sources; a few hours on one core, `--workers N` runs in parallel):

   ```
   python hmxb/build_results_sheet.py --root <data folder> --out <results folder>
   ```

   It writes one JSON file per source and instrument, the summary CSV tables and a workbook. The JSON files and CSV tables in `results/` are the output of this step (unchanged since v1.0.0).

2. Paper numbers, tables and figures (set `HMXB_RESULTS` to the results folder if it is not
   `results/`):

   ```
   python paper/reproduce_all.py
   ```

   The output in `paper_output/` corresponds to the paper as follows: `accounting.json` (Tables 1-2), `catalogue_comparison.json` (Table 3), `ax1820.json` (Table 4), `tab_pdot.tex`, `period_derivatives.json`, `eclipse_sensitivity.json`, `pdot_summary.json` (Table 5), `tab_bat_full.tex` and `tab_xmm_full.tex` (Tables A1-A2), `xmm_timing_bounds.csv`, `cenx3_doppler.json`, `halves_stats.json` (Sect. 4.11), `caps_check.json` (Sect. 4.1), and `Fig1`-`Fig6`.

## Where each step of the method is implemented

| Step (section of the paper) | Function |
|---|---|
| Light-curve cleaning (2.2) | `bat_period_pipeline.load_bat_lc` |
| Semi-weighting, Eqs. (1)-(2) (3.1) | `bat_period_pipeline.effective_errors` |
| Generalised Lomb-Scargle periodogram (3.1) | `bat_period_pipeline.compute_periodogram` |
| Local continuum and strength S, Eq. (3) (3.2) | `bat_period_pipeline.local_continuum` |
| Analytic FAP and thresholds, Eq. (4) (3.2) | `hmxb_common.fap_median_norm`, `threshold_median_norm` |
| Red-noise simulations (3.2) | `bat_period_pipeline.simulate_max_strength`, `simulated_fap` |
| Targeted test of a catalogue period (3.2) | `bat_period_pipeline.analyse_star` (known-period branch) |
| Family check (3.3 i) | `bat_period_pipeline.family_check`, `fold_profile` |
| One-day alias (3.3 ii) | `bat_period_pipeline.alias_check` |
| Split-data test (3.3 iii) | `bat_period_pipeline.data_halves_check` |
| Related peaks (3.3 iv) | `bat_period_pipeline.mark_harmonics`, `relation_to` |
| Instrumental periods (3.3 v) | `bat_period_pipeline.instrumental_flag` |
| Adoption rules (3.3) | `bat_period_pipeline.analyse_star` |
| Harmonic fit, Eq. (5), and errors (3.4) | `hmxb_common.harmonic_period_fit` (via `measure_period`) |
| Outburst timing (3.5) | `bat_period_pipeline.detect_outbursts`, `outburst_timing`, `fit_ephemeris` |
| XMM light curves, rebinning (2.3) | `xmm_spin_pipeline.select_target_lightcurves`, `load_xmm_lc`, `rebin` |
| Normalised power, Eq. (6) (3.6) | `xmm_spin_pipeline.spin_spectrum`, `local_mean_power` |
| Harmonic sums and FAP, Eq. (7) (3.6) | `xmm_spin_pipeline.harmonic_sum`, `gamma_fap`, `search_spin` |
| Fundamental check (3.6) | `xmm_spin_pipeline.spin_family_check` |
| 10-ks segments and trial factors (3.6) | `xmm_spin_pipeline.analyse_xmm_star`, `analyse_observation` |
| Per-source adoption, N_obs (3.6) | `xmm_spin_pipeline.choose_best_spin` |
| Period derivatives, O-C (4.5) | `paper/period_derivatives.py`, `paper/eclipse_sensitivity.py` |
| Timing-correction bounds (4.9) | `paper/xmm_timing_bounds.py` |

## Notes

* `build_results_sheet.py` can also compare the results with a personal table of earlier values (`Stars/my results.xlsx`); this file is optional, is only read, and is not part of the package.
* The FAP values of very strong signals are analytic and are reported in the paper as upper bounds (`< 1e-10`); in the JSON files a FAP of exactly 0 means that the value is below the double-precision range (about 1e-308).
* The candidate-cap check of Sect. 4.1 (`paper/caps_check.py`) reruns the candidate stage of 19 sources with 30 instead of 8 peaks per search and takes several hours.

## Citation

Please cite the paper and the archived version of this package (`CITATION.cff`).
