"""Paths used by the paper scripts. Edit here or set environment variables.

HMXB_DATA     folder that contains Stars\\ (one sub-folder per source with the Swift/BAT and
              XMM-Newton files) and HMXBwebcat_latest.xlsx          (default D:\\HMXBs)
HMXB_RESULTS  output folder of hmxb/build_results_sheet.py with BAT\\*.json, XMM\\*.json and
              Summary.csv (or the results workbook)                 (default <package>/results)
HMXB_OUT      where the paper tables, figures and JSON summaries are written
                                                                    (default <package>/paper_output)
"""
import os
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("HMXB_DATA", r"D:\HMXBs"))
STARS = DATA_ROOT / "Stars"
CATALOG = DATA_ROOT / "HMXBwebcat_latest.xlsx"
RESULTS = Path(os.environ.get("HMXB_RESULTS", PKG / "results"))
OUT = Path(os.environ.get("HMXB_OUT", PKG / "paper_output"))
CODE = PKG / "hmxb"
OUT.mkdir(parents=True, exist_ok=True)
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
TREF = 57242.0          # middle of the BAT data (MJD), reference epoch for period changes
