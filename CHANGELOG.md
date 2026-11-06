# Changelog

## 1.0.1 (2026-11-02)

Consistency fixes for the version of the paper in the official AN (Wiley) template.

* Table A2 (`paper/make_tables.py`): layout for the AN template; a superscript s marks detections in a
  10-ks segment; note (e) gives the reason why the catalogue periods of AX J1838.0-0655 and
  XTE J1829-098 cannot be tested; the "no target light curve" entry is now note (k). FAPs below 1e-10
  are given as upper bounds, as before.
* Fig. 5 and Sect. 4.11 (`paper/figures.py`): the half-to-half comparison uses the 35 blind detections
  without documented period evolution (GX 301-2 and OAO 1657-415 are now excluded as well), stacked for
  Be/X-ray binaries and other systems; the statistics are written to `halves_stats.json`.
* Table 5 (`paper/pdot_table.py`): layout for the AN template.
* `paper_output/caps_check.json`: output of the candidate-cap test included.

## 1.0.0 (2026-10-27)

First public release, accompanying the revised manuscript submitted to Astronomische Nachrichten.

* Pipeline (`hmxb/`) as used for all results of the paper.
* `paper/`: scripts for the candidate accounting (Tables 1-2), the catalogue comparison by group
  (Table 3), the AX J1820.5-1434 subsets (Table 4), the O-C and two-interval period-derivative
  analysis with its sensitivity tests and the epoch-matched comparison with published ephemerides
  (Table 5), the timing-correction bounds of the apparent XMM spin periods, the Cen X-3 Doppler fit,
  the appendix tables and all figures.
* `data/`: manifests with SHA-256 checksums and product versions of the 329 input files, the
  folder-to-catalogue name table (the pipeline no longer needs a personal results file to match
  names), an input checker and a data-snapshot script.
