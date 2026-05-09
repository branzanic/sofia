#!/usr/bin/env python3
"""
NSD-style decomposition of F430 macrocycles in ORCA QM/MM output.

For each ORCA QM/MM calculation in a directory tree (one subfolder per run),
this script reads the converged QM-region geometry from <name>.QMRegion.xyz
and runs the corphin (D4h-approximate, geometry-only) NSD pipeline on the
F430 inner ring (4 N + 8 Cα + 4 Cmeso + 8 Cβ).

The metal is identified automatically from the QM-region atomic composition;
the spin multiplicity is read from the matching <name>.out (ORCA stdout) so
the per-file table can label each calculation by metal and S.

CAVEAT (inherited from nsd_corphin.py)
======================================
Corphin / F430 has C1 symmetry, not D4h. The mode labels (saddled, ruffled,
domed, waved, propellered) are approximate. Geometric quantities
(M-plane, M-N, max |Δz|, |D_oop|, |D_ip|, class radii) are rigorous.

Usage:
    python3 nsd_qmmm.py /path/to/QMMM_articol --csv qmmm_nsd.csv --plot plots/qmmm
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import nsd_porphyrin as porph


_BANNER = (
    "================================================================\n"
    " nsd_qmmm.py -- NSD analysis of F430 in ORCA QM/MM output\n"
    " Parses QMRegion.xyz; runs corphin (D4h-approximate) pipeline.\n"
    " Conformation labels are D4h-approximate (corphin is C1).\n"
    "================================================================"
)

# Element symbol to atomic number (subset relevant here)
_Z = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "Na": 11, "Mg": 12, "P": 15,
      "S": 16, "Cl": 17, "K": 19, "Ca": 20, "Mn": 25, "Fe": 26, "Co": 27,
      "Ni": 28, "Cu": 29, "Zn": 30, "Rh": 45}


def parse_xyz(path):
    """Parse a standard XYZ file. Returns (atoms[N], coords[N,3])."""
    with open(path) as f:
        lines = f.readlines()
    n = int(lines[0].strip())
    atoms = np.zeros(n, dtype=int)
    coords = np.zeros((n, 3), dtype=float)
    for i in range(n):
        parts = lines[2 + i].split()
        sym = parts[0]
        atoms[i] = _Z.get(sym, 0)
        coords[i] = [float(parts[1]), float(parts[2]), float(parts[3])]
    return atoms, coords


def parse_orca_multiplicity(out_path):
    """Read the multiplicity (M = 2S+1) from an ORCA .out file. None if not found."""
    if not out_path.exists():
        return None
    with open(out_path) as f:
        for line in f:
            if "Multiplicity" in line and "Mult" in line:
                m = re.search(r"\.\.\.\.\s*(\d+)", line)
                if m:
                    return int(m.group(1))
    return None


def mult_to_spin(mult):
    if mult is None:
        return "?"
    s = (mult - 1) / 2
    if s == 0:    return "S=0"
    if s == 0.5:  return "S=½"
    if s == 1:    return "S=1"
    if s == 1.5:  return "S=3/2"
    if s == 2:    return "S=2"
    if s == 2.5:  return "S=5/2"
    return f"S={s}"


def find_qmmm_runs(root):
    """Yield (xyz_path, out_path, label) tuples for each QM/MM calculation
    discovered anywhere under root (recursive). Each .QMRegion.xyz file
    found under root is treated as a separate run; the matching .out is
    searched in the same directory; the label is the parent-directory name
    so the user sees which calculation each row came from."""
    root = Path(root)
    for xyz in sorted(root.rglob("*.QMRegion.xyz")):
        stem = xyz.stem.replace(".QMRegion", "")
        out_candidates = list(xyz.parent.glob(f"{stem}.out"))
        out_path = out_candidates[0] if out_candidates else None
        yield xyz, out_path, xyz.parent.name


def analyse_qmmm(xyz_path, out_path, label, plot_dir=None, verbose=False,
                 report_dir=None):
    atoms, coords = parse_xyz(xyz_path)
    classes, macro_idx = porph.identify_macrocycle(atoms, coords)
    macro_coords = coords[macro_idx]

    centroid, normal = porph.fit_mean_plane(macro_coords)
    R = porph.rotation_to_z(normal)
    plane_coords = (macro_coords - centroid) @ R.T

    n_local = np.arange(4)
    canonical = porph.canonicalise_xy(plane_coords, n_local)

    dz = canonical[:, 2].copy()
    basis = porph.build_oop_basis(canonical)
    amps = porph.project(dz, basis)
    recon = porph.reconstruct(amps, basis)

    irrep_mag = {
        "B2u (sad)": abs(amps["sad"]),
        "B1u (ruf)": abs(amps["ruf"]),
        "A2u (dom)": abs(amps["dom"]),
        "Eg (wav)":  float(np.hypot(amps["wav_x"], amps["wav_y"])),
        "A1u (pro)": abs(amps["pro"]),
    }

    d_total_oop = float(np.linalg.norm(dz))
    d_minbasis = float(np.linalg.norm(recon))

    metal_z = int(atoms[classes["metal"][0]])
    metal_sym = porph.ELEMENT_SYMBOLS.get(metal_z, f"Z{metal_z}")
    max_abs_dz = float(np.max(np.abs(dz)))

    # Metal-to-plane displacement and M-N distances
    metal_idx = classes["metal"][0]
    metal_in_plane = (coords[metal_idx] - centroid) @ R.T
    metal_to_plane = float(metal_in_plane[2])
    M_N_distances = [float(np.linalg.norm(coords[metal_idx] - coords[n]))
                     for n in classes["N"]]
    M_N_mean = float(np.mean(M_N_distances))

    conf_oop = porph.describe_oop_conformation(amps)

    class_array = ["N"] * 4 + ["Calpha"] * 8 + ["Cmeso"] * 4 + ["Cbeta"] * 8
    ip = porph.analyse_in_plane(canonical, class_array)
    ip_amps = ip["amps"]
    ip_irrep = ip["irrep_mag"]
    radii = ip["class_radii"]
    conf_ip = porph.describe_ip_conformation(ip_amps)

    mult = parse_orca_multiplicity(out_path) if out_path else None
    spin = mult_to_spin(mult)

    # Check for axial S coordination (CoM ligand)
    s_atoms = [i for i, z in enumerate(atoms) if int(z) == 16]
    M_S_distance = None
    if s_atoms:
        ds = [np.linalg.norm(coords[metal_idx] - coords[i]) for i in s_atoms]
        M_S_distance = float(min(ds))

    result = {
        "calculation": label,
        "metal": metal_sym,
        "spin": spin,
        "multiplicity": mult,
        "n_atoms_qm": len(atoms),
        "conformation_oop": conf_oop,
        "conformation_ip":  conf_ip,
        "max_abs_dz_A": max_abs_dz,
        "rms_dz_A": float(np.sqrt(np.mean(dz ** 2))),
        "D_oop_obs_A": d_total_oop,
        "D_oop_minbasis_A": d_minbasis,
        "frac_explained_oop": (d_minbasis / d_total_oop) if d_total_oop > 1e-10 else 1.0,
        "d_sad_A": amps["sad"], "d_ruf_A": amps["ruf"],
        "d_dom_A": amps["dom"],
        "d_wav_x_A": amps["wav_x"], "d_wav_y_A": amps["wav_y"],
        "d_pro_A": amps["pro"],
        "Mag_B2u_sad_A": irrep_mag["B2u (sad)"],
        "Mag_B1u_ruf_A": irrep_mag["B1u (ruf)"],
        "Mag_A2u_dom_A": irrep_mag["A2u (dom)"],
        "Mag_Eg_wav_A":  irrep_mag["Eg (wav)"],
        "Mag_A1u_pro_A": irrep_mag["A1u (pro)"],
        "D_ip_obs_A": ip["D_ip_obs"],
        "D_ip_minbasis_A": ip["D_ip_min"],
        "frac_explained_ip": ip["frac_explained_ip"],
        "d_bre_A": ip_amps["bre"], "d_Nstr_A": ip_amps["Nstr"],
        "d_mstr_A": ip_amps["mstr"], "d_rot_A": ip_amps["rot"],
        "d_trn_x_A": ip_amps["trn_x"], "d_trn_y_A": ip_amps["trn_y"],
        "M_to_plane_A": metal_to_plane,
        "M_N_mean_A":   M_N_mean,
        "M_S_axial_A":  M_S_distance,
        "r_M_N_A":     radii["N"],
        "r_Calpha_A":  radii["Calpha"],
        "r_Cmeso_A":   radii["Cmeso"],
        "r_Cbeta_A":   radii["Cbeta"],
    }

    if verbose:
        _print_qmmm_report(result, classes, atoms, canonical, dz, recon, ip,
                          source_path=xyz_path)

    if report_dir is not None:
        out_path = _save_qmmm_report(
            result, classes, atoms, canonical, dz, recon, ip,
            report_dir, source_path=xyz_path)
        if verbose:
            print(f"  Wrote report: {out_path}")

    if plot_dir is not None:
        import nsd_plot
        plot_dir = Path(plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
        class_labels = ["N"] * 4 + ["Cα"] * 8 + ["Cmeso"] * 4 + ["Cβ"] * 8
        oop_modes = ["sad\n(B2u)", "ruf\n(B1u)", "dom\n(A2u)",
                     "wav(x)\n(Eg)", "wav(y)\n(Eg)", "pro\n(A1u)"]
        oop_amps = [amps["sad"], amps["ruf"], amps["dom"],
                    amps["wav_x"], amps["wav_y"], amps["pro"]]
        nsd_plot.plot_per_file_combined(
            canonical[:, :2], dz, recon, class_labels,
            oop_modes, oop_amps,
            f"{label}\n{metal_sym} {spin} (QM/MM, F430 in ECR enzyme)",
            plot_dir / f"{label}_OOP.png")

    return result


def _qmmm_result_for_format(r):
    """Adapt the QM/MM result dict to the standard keys expected by
    porph.format_report ('file', 'n_atoms_total')."""
    rr = dict(r)
    rr["file"] = r["calculation"] + ".QMRegion.xyz"
    rr["n_atoms_total"] = r["n_atoms_qm"]
    return rr


def _print_qmmm_report(r, classes, atoms, canonical, dz, recon, ip,
                       source_path=None):
    """Backwards-compatible wrapper — calls porph.format_report with QM/MM
    parameters and prints the result."""
    print(porph.format_report(
        _qmmm_result_for_format(r), classes, atoms, canonical, dz, recon, ip,
        source_path=source_path,
        macrocycle_name="F430 / corphin (QM/MM)",
        non_d4h_caveat=True,
        m_s_distance=r.get("M_S_axial_A"),
    ))


def _save_qmmm_report(r, classes, atoms, canonical, dz, recon, ip,
                     report_dir, source_path=None):
    """Write the per-run QM/MM report as <stem>.report.txt under report_dir."""
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    stem = r["calculation"]
    out = Path(report_dir) / f"{stem}.report.txt"
    text = porph.format_report(
        _qmmm_result_for_format(r), classes, atoms, canonical, dz, recon, ip,
        source_path=source_path,
        macrocycle_name="F430 / corphin (QM/MM)",
        non_d4h_caveat=True,
        m_s_distance=r.get("M_S_axial_A"),
    )
    out.write_text(text)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="QMMM_articol root directory (containing one subfolder per QM/MM run)")
    ap.add_argument("--csv", help="Write per-run results to this CSV")
    ap.add_argument("-q", "--quiet", action="store_true",
                    help="Suppress per-run detailed report")
    ap.add_argument("--plot", metavar="DIR",
                    help="Write per-run PNG plots and a cross-run summary to DIR")
    ap.add_argument("--report", metavar="DIR",
                    help="Write per-run scientific .txt report to DIR "
                         "(one <label>.report.txt per QM/MM run)")
    args = ap.parse_args()

    print(_BANNER)
    print()

    runs = list(find_qmmm_runs(args.path))
    if not runs:
        sys.exit(f"No QM/MM runs found under {args.path}")
    print(f"Found {len(runs)} QM/MM run(s):")
    for xyz, out, label in runs:
        print(f"  {label}")
    print()

    rows = []
    for xyz, out, label in runs:
        try:
            r = analyse_qmmm(xyz, out, label, plot_dir=args.plot,
                             verbose=not args.quiet, report_dir=args.report)
            rows.append(r)
        except Exception as e:
            print(f"!! {label}: {e}", file=sys.stderr)

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
        print(" QM/MM summary (F430 macrocycle in ECR enzyme):")
        print(f"  {'calculation':<40s}  {'metal':<3s} {'spin':<6s} "
              f"{'M-plane':>8s} {'M-N':>7s} {'M-S':>7s}  {'|D_oop|':>8s}  conformation")
        for r in rows:
            ms = f"{r['M_S_axial_A']:.3f}" if r['M_S_axial_A'] is not None else "  -"
            print(f"  {r['calculation']:<40s}  {r['metal']:<3s} {r['spin']:<6s} "
                  f"{r['M_to_plane_A']:>+8.3f} {r['M_N_mean_A']:>7.3f} {ms:>7s}  "
                  f"{r['D_oop_obs_A']:>8.3f}  {r['conformation_oop']}")

    if args.plot and rows:
        import nsd_plot
        plot_dir = Path(args.plot)
        plot_dir.mkdir(parents=True, exist_ok=True)
        labels = [f"{r['calculation'].replace('7b1s_qmmm_', '').replace('_readmo_', '_')}\n({r['metal']} {r['spin']})"
                  for r in rows]
        oop_modes = ["sad", "ruf", "dom", "wav(x)", "wav(y)", "pro"]
        oop_keys = ["d_sad_A", "d_ruf_A", "d_dom_A",
                    "d_wav_x_A", "d_wav_y_A", "d_pro_A"]
        oop_mat = np.array([[r[k] for k in oop_keys] for r in rows])
        nsd_plot.plot_summary(labels, oop_mat, oop_modes,
                              "QM/MM F430 in ECR -- OOP amplitudes (D4h-approximate, geometry-only)",
                              plot_dir / "_summary_OOP.png")
        print(f"\nWrote plots to {plot_dir}")


if __name__ == "__main__":
    main()
