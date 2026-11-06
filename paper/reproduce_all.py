"""Reproduce every number, table and figure of the paper from the pipeline results.

    python paper/reproduce_all.py                 # all steps (about 30 min; period_derivatives is the slow one)
    python paper/reproduce_all.py --skip-slow     # reuse period_derivatives.json if it exists
    python paper/reproduce_all.py --with-caps     # also rerun the candidate-cap check (several hours)

Inputs: the pipeline results (results/ of this package, or HMXB_RESULTS) and, for the steps that
read light curves, the input data (HMXB_DATA, see README.md). Outputs go to paper_output/ (or HMXB_OUT).

Step                      paper item
accounting.py             Tables 1 and 2, Sect. 4.1 (candidate flow and results by class)
catalogue_comparison.py   Table 3, Sect. 4.2, Fig. 1a
ax1820_subsets.py         Table 4, Sect. 4.4                         (light curve)
period_derivatives.py     Table 5, Sect. 4.5, Fig. 3                 (light curves, slow)
eclipse_sensitivity.py    Table 5 (eclipse-only variants, effective epochs)
pdot_table.py             Table 5 (LaTeX), epoch-matched ephemeris comparison
xmm_timing_bounds.py      Table A2 (delta_max), Sect. 4.9, Fig. 1b
cenx3_doppler.py          Sect. 4.10, Fig. 4                         (light curve)
make_tables.py            Tables A1 and A2
figures.py                Figs. 1-6; halves_stats.json (Sect. 4.11, Fig. 5)
caps_check.py             Sect. 4.1, enlarged candidate caps         (optional, very slow)
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config  # noqa: E402

STEPS = ["accounting.py", "catalogue_comparison.py", "xmm_timing_bounds.py", "cenx3_doppler.py",
         "ax1820_subsets.py", "period_derivatives.py", "eclipse_sensitivity.py", "pdot_table.py",
         "make_tables.py", "figures.py"]


def main(argv):
    skip_slow = "--skip-slow" in argv
    steps = list(STEPS) + (["caps_check.py"] if "--with-caps" in argv else [])
    for st in steps:
        if skip_slow and st == "period_derivatives.py" and (config.OUT / "period_derivatives.json").exists():
            print(f"-- {st}: skipped (using existing period_derivatives.json)")
            continue
        t0 = time.time()
        print(f"-- {st}", flush=True)
        r = subprocess.run([sys.executable, str(HERE / st)], cwd=str(HERE))
        if r.returncode != 0:
            print(f"   FAILED ({st}); stopping", flush=True)
            return r.returncode
        print(f"   done in {time.time() - t0:.0f} s", flush=True)
    print(f"All outputs in {config.OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
