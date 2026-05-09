#!/usr/bin/env python3
"""
NSD-style decomposition of corphin (F430) macrocycles in Gaussian opt logs.

Reuses the porphyrin pipeline (nsd_porphyrin.py): the 24-atom inner ring of
corphin has the same atomic topology as porphyrin (4 N + 8 Calpha + 4 Cmeso +
8 Cbeta), so the macrocycle identifier and the analytic D4h minimal basis
apply directly.

IMPORTANT CAVEAT
================
Corphin is not D4h. The saturation pattern of F430 (sp3 carbons in one or more
pyrrole rings, plus the fused F-ring) breaks the four-fold symmetry. As a
consequence:

  - The mode labels sad, ruf, dom, wav(x), wav(y), pro (OOP) and bre, N-str,
    m-str, rot, trn(x), trn(y) (IP) are APPROXIMATE classifications. The
    actual normal modes of the molecule are not pure irreps of D4h.
  - The signed amplitudes d_X are well-defined geometric projections onto a
    fixed analytic basis, and are useful for cross-comparison across metals
    and spin states. They should not be interpreted as exact normal-mode
    amplitudes.
  - The per-irrep magnitudes |D^Gamma| are NOT rigorously basis-independent
    for corphin in the way they are for porphyrin (because the molecule does
    not actually have those irreps).

  - The geometric quantities (mean-plane fit, max |dz|, |D_oop|, |D_ip|,
    M-plane, M-N, per-class mean radii, per-atom dz profile) ARE rigorous,
    basis-independent, and the most reliable summary of corphin distortion.

For a basis-rigorous corphin NSD analysis, run a Gaussian opt+freq job on a
reference corphin geometry (e.g. F430_Ni2 -- the F430-native S=1/2 Ni state)
at the manuscript's level of theory and use nsd_corrin.py with --ref pointing
to that log. The corrin pipeline already handles non-D4h reference systems.

Usage:
    python3 nsd_corphin.py <file.log>
    python3 nsd_corphin.py <directory>
    python3 nsd_corphin.py <directory> --csv out.csv --plot plots/corphin
"""

import argparse
import sys
from pathlib import Path

# Import the core analysis from nsd_porphyrin (same module dir)
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np

import nsd_porphyrin as porph


_BANNER = (
    "================================================================\n"
    " nsd_corphin.py -- NSD-style analysis of corphin/F430 macrocycles\n"
    " Reusing the porphyrin pipeline. NB: corphin is NOT D4h.\n"
    " Mode labels (sad/ruf/dom/wav/pro, bre/N-str/m-str/rot/trn) are\n"
    " APPROXIMATE for corphin. Geometric quantities (|D_oop|, |D_ip|,\n"
    " M-plane, M-N, class radii, per-atom dz) are rigorous.\n"
    "================================================================"
)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="Gaussian .log file or directory of .log files")
    ap.add_argument("--csv", help="Write per-file results to this CSV")
    ap.add_argument("-q", "--quiet", action="store_true",
                    help="Suppress per-file detailed report")
    ap.add_argument("--plot", metavar="DIR",
                    help="Write per-file PNG plots and a cross-file summary to DIR")
    ap.add_argument("--report", metavar="DIR",
                    help="Write per-file scientific .txt report to DIR "
                         "(one <stem>.report.txt per input)")
    args = ap.parse_args()

    print(_BANNER)
    print()

    p = Path(args.path)
    if p.is_dir():
        # Recursive: find all .log files anywhere under the given directory.
        files = sorted(p.rglob("*.log"))
    elif p.is_file():
        files = [p]
    else:
        sys.exit(f"Path not found: {p}")
    if not files:
        sys.exit(f"No .log files in {p}")

    rows = []
    for f in files:
        try:
            r = porph.analyse_log(f, verbose=not args.quiet, plot_dir=args.plot,
                                  report_dir=args.report,
                                  macrocycle_name="corphin / F430",
                                  non_d4h_caveat=True)
            rows.append(r)
        except Exception as e:
            print(f"!! {f.name}: {e}", file=sys.stderr)

    import csv
    if args.csv and rows:
        keys = list(rows[0].keys())
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nWrote {len(rows)} rows to {args.csv}")

    if len(rows) > 1:
        print("\n" + "=" * 100)
        print(" Conformation summary (D4h-approximate labels for corphin):")
        print(f"  {'file':28s} {'metal':>3s}   {'OOP conformation':<35s} {'IP conformation':<25s}")
        for r in rows:
            print(f"  {r['file']:28s} {r['metal']:>3s}   "
                  f"{r['conformation_oop']:<35s} {r['conformation_ip']:<25s}")
        print("\n" + "=" * 88)
        print(" Distortion magnitudes (geometric, basis-independent):")
        print(" file                         metal   max|dz| |D_oop| |D_ip|  ruf*    M-N    %expl")
        for r in rows:
            wav = np.hypot(r["d_wav_x_A"], r["d_wav_y_A"])
            ip_obs = r.get("D_ip_obs_A", 0.0)
            print(f"  {r['file']:28s} {r['metal']:>3s}    "
                  f"{r['max_abs_dz_A']: .3f}  "
                  f"{r['D_oop_obs_A']: .3f}  "
                  f"{ip_obs: .3f}  "
                  f"{r['d_ruf_A']:+.3f}  "
                  f"{r['M_N_mean_A']: .3f}  "
                  f"{100*r['frac_explained_oop']:5.1f}")
        print(" * mode amplitudes (sad/ruf/dom/wav/pro) are D4h-approximate for corphin.")

    # Cross-file summary plots (re-uses porphyrin plotter via nsd_porphyrin's
    # plot block embedded in main(); to avoid double-running it, we build it
    # explicitly here)
    if args.plot and rows:
        import nsd_plot
        plot_dir = Path(args.plot)
        plot_dir.mkdir(parents=True, exist_ok=True)
        labels = [f"{r['file'].replace('.log','')}\n({r['metal']})" for r in rows]
        oop_modes = ["sad", "ruf", "dom", "wav(x)", "wav(y)", "pro"]
        oop_keys = ["d_sad_A", "d_ruf_A", "d_dom_A",
                    "d_wav_x_A", "d_wav_y_A", "d_pro_A"]
        oop_mat = np.array([[r[k] for k in oop_keys] for r in rows])
        nsd_plot.plot_summary(labels, oop_mat, oop_modes,
                              "Corphin -- OOP minimal-basis amplitudes (D4h-approximate labels)",
                              plot_dir / "_summary_OOP.png")
        ip_modes = ["bre (~0)", "N-str", "m-str", "rot", "trn(x)", "trn(y)"]
        ip_keys = ["d_bre_A", "d_Nstr_A", "d_mstr_A",
                   "d_rot_A", "d_trn_x_A", "d_trn_y_A"]
        ip_mat = np.array([[r[k] for k in ip_keys] for r in rows])
        nsd_plot.plot_summary(labels, ip_mat, ip_modes,
                              "Corphin -- in-plane minimal-basis amplitudes (D4h-approximate labels)",
                              plot_dir / "_summary_IP.png")
        print(f"\nWrote plots to {plot_dir}")


if __name__ == "__main__":
    main()
