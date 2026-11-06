"""Shared helpers for the paper scripts."""
import glob
import json

import numpy as np
import pandas as pd

import config


def stem(name):
    return name.replace(" ", "_")


def load_summary():
    csv = config.RESULTS / "Summary.csv"
    if csv.exists():
        return pd.read_csv(csv)
    xl = sorted(config.RESULTS.glob("HMXB pipeline results*.xlsx"))
    if not xl:
        raise FileNotFoundError(f"no Summary.csv or results workbook in {config.RESULTS}")
    return pd.read_excel(xl[-1], sheet_name="Summary")


def load_catalogue():
    return pd.read_excel(config.CATALOG).set_index("Main_ID")


def bat_json(name):
    return json.load(open(config.RESULTS / "BAT" / f"{stem(name)}_BAT.json"))


def xmm_json(name):
    return json.load(open(config.RESULTS / "XMM" / f"{stem(name)}_XMM.json"))


def all_bat():
    return [json.load(open(f)) for f in sorted(glob.glob(str(config.RESULTS / "BAT" / "*_BAT.json")))]


def all_xmm():
    return [json.load(open(f)) for f in sorted(glob.glob(str(config.RESULTS / "XMM" / "*_XMM.json")))]


def star_dir(name):
    return config.STARS / name


def class_group(c):
    c = str(c)
    if c.startswith("Be"):
        return "Be"
    if c.startswith("sg"):
        return "sg"
    if c.startswith("SFXT"):
        return "SFXT"
    if "gamma" in c:
        return "gamma"
    if c == "nan":
        return "unclassified"
    return "other"


REJECTED = {"PSR J2032+4127"}          # blind detection rejected as physically implausible (Sect. 4.8)
# systems with documented orbital period evolution (folder names)
EVOLVING = {"V779 Cen": "Cen X-3", "V1521 Cyg": "Cyg X-3", "BP Cru": "GX 301-2",
            "OAO 1657-415": "OAO 1657-415"}


def dump(obj, name):
    def conv(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        raise TypeError(type(o))
    with open(config.OUT / name, "w") as fh:
        json.dump(obj, fh, indent=1, default=conv)
