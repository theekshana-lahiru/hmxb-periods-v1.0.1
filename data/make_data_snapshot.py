"""Copy the exact input files used for the paper into one zip archive for long-term storage
(for example as a Zenodo data record next to the code record).

    python data/make_data_snapshot.py [path-to-data-folder] [output.zip]

Only the files listed in the manifests are included (Swift/BAT transient-monitor daily and orbit
light curves, XMM-Newton PPS source time series, the HMXB catalogue file), with the same relative
paths, together with the manifests. Both data sets are public: BAT transient-monitor results from
NASA/GSFC and XMM-Newton pipeline products from the ESA XMM-Newton Science Archive; acknowledge
them as in the paper when reusing the snapshot."""
import csv
import os
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main(argv):
    root = Path(argv[0]) if argv else Path(os.environ.get("HMXB_DATA", r"D:\HMXBs"))
    out = Path(argv[1]) if len(argv) > 1 else Path.cwd() / "hmxb-periods-input-data-2026-02.zip"
    n = 0
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        for man, base, prefix in (("manifest_bat_lightcurves.csv", root / "Stars", "Stars/"),
                                  ("manifest_xmm_pps_lightcurves.csv", root / "Stars", "Stars/"),
                                  ("manifest_catalogue.csv", root, "")):
            z.write(HERE / man, man)
            with open(HERE / man, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    p = base / row["file"]
                    if p.exists():
                        z.write(p, prefix + row["file"])
                        n += 1
                    else:
                        print("missing", p)
        z.write(HERE / "source_names.csv", "source_names.csv")
    print(f"{n} files written to {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
