#!/usr/bin/env python3
"""
NSD-style decomposition of corrin macrocycles in Gaussian opt logs --
GEOMETRY-ONLY analysis (no frequency data required).

Uses the SAME analytic D4h-pattern minimal basis as nsd_porphyrin.py and
nsd_corphin.py, applied to the 23-atom inner ring of corrin (4 N + 8 Calpha
+ 3 Cmeso + 8 Cbeta -- corrin has one direct Calpha-Calpha bond replacing
a meso). The basis vectors are evaluated at the 23 actual atom positions
in the molecule's canonical (mean-plane, N-on-axes) frame.

This script is the GEOMETRY-ONLY counterpart to nsd_corrin_freq.py, which
uses the lowest-frequency vibrational normal modes of an opt+freq reference
calculation as the basis. The freq-based version is more rigorous (the basis
is the actual molecular eigenvectors, not analytic patterns) but requires
opt+freq logs. The geometry-only version produces output consistent in
format and labels with nsd_porphyrin.py / nsd_corphin.py for the unified
project report.

CAVEAT (same as for corphin)
============================
Corrin is C1 / approximately C2v (lower than D4h). The mode labels saddled,
ruffled, domed, waved, propellered (OOP) and breathing, N-stretched,
meso-stretched, rotated, translated (IP) are APPROXIMATE classifications
when applied to corrin: the actual molecular modes are not pure irreps of
D4h. Conformation labels remain useful for cross-comparison across metal /
spin combinations, but should not be interpreted as exact normal-mode
identifications.

The geometric quantities (mean-plane fit, max |dz|, |D_oop|, |D_ip|,
M-plane, M-N, per-class mean radii, per-atom dz profile) are rigorous,
basis-independent, and the most reliable summary of corrin distortion.

For a basis-rigorous corrin analysis use nsd_corrin_freq.py.

Usage:
    python3 nsd_corrin.py <file.log>
    python3 nsd_corrin.py <directory>
    python3 nsd_corrin.py <directory> --csv corrin_nsd.csv --plot plots/corrin -q
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

# Reuse helpers from the porphyrin module
sys.path.insert(0, str(Path(__file__).parent))
import nsd_porphyrin as porph


# -------------------- corrin-specific topology identifier --------------------

def identify_corrin(atoms, coords):
    """Identify the 23-atom corrin macrocycle by graph topology.

    Robust to peripheral C substituents (e.g. propionamide / methyl chains in
    full cobalamin models). Algorithm:

      1. Locate the metal and its 4 N neighbours.
      2. Cα = C atoms bonded to those N's (8 total).
      3. Cβ = the two non-Cα carbons in each pyrrole 5-ring N-Cα-Cβ-Cβ-Cα-N
         (walked explicitly per pyrrole — does NOT rely on heavy-atom degree).
      4. Cmeso = C atoms bonded to two Cα belonging to DIFFERENT pyrroles
         (works whether Cmeso has a peripheral substituent or not).
      5. Verify exactly one direct Cα-Cα bond exists (the corrin closure).
    """
    adj = porph.build_adjacency(atoms, coords)

    metals = [i for i, z in enumerate(atoms) if int(z) in porph.TRANSITION_METALS]
    if len(metals) != 1:
        raise ValueError(f"Expected 1 transition metal; found {len(metals)}")
    metal = metals[0]

    # Get all N atoms bonded to metal. In biological cobalamin models the
    # metal can also be bonded to a dimethylbenzimidazole (DMB) axial N --
    # filter to the 4 N's that are PYRROLE N's, identified by being part of
    # a five-membered ring N-Calpha-Cbeta-Cbeta-Calpha-N closure.
    def _is_pyrrole_n(n):
        c_nbrs = [k for k in adj[n] if int(atoms[k]) == 6]
        if len(c_nbrs) < 2:
            return False
        # Try every pair of C neighbours as the two Calpha
        for i_ca, ca1 in enumerate(c_nbrs):
            for ca2 in c_nbrs[i_ca + 1:]:
                # Walk: ca1 -> cb1 -> cb2 -> ca2 (all C, none = n)
                for cb1 in adj[ca1]:
                    if int(atoms[cb1]) != 6 or cb1 == n or cb1 == ca2:
                        continue
                    for cb2 in adj[cb1]:
                        if int(atoms[cb2]) != 6 or cb2 == ca1 or cb2 == n:
                            continue
                        if ca2 in adj[cb2]:
                            return True
        return False

    n_neighbours = sorted(j for j in adj[metal] if int(atoms[j]) == 7)
    pyrrole_n = [n for n in n_neighbours if _is_pyrrole_n(n)]
    if len(pyrrole_n) != 4:
        raise ValueError(f"Expected 4 pyrrole N bonded to metal "
                         f"(after pyrrole-5-ring filter); found "
                         f"{len(pyrrole_n)} of {len(n_neighbours)} N neighbours")
    # Non-pyrrole N's bonded to metal are recorded as axial N ligands
    axial_n_bonded = [n for n in n_neighbours if n not in pyrrole_n]
    n_idx = sorted(pyrrole_n)

    calpha = set()
    for n in n_idx:
        for k in adj[n]:
            if int(atoms[k]) == 6:
                calpha.add(k)
    if len(calpha) != 8:
        raise ValueError(f"Expected 8 Calpha; found {len(calpha)}")

    # Group Cα by their pyrrole (= which N each Cα is bonded to)
    pyrrole_calpha = {n: [k for k in adj[n] if k in calpha] for n in n_idx}

    # Identify Cβ via the 5-ring walk N-Cα-Cβ-Cβ-Cα-N
    cbeta = set()
    for n, cas in pyrrole_calpha.items():
        if len(cas) != 2:
            raise ValueError(f"Pyrrole at N {n+1} has {len(cas)} Cα (expected 2)")
        ca1, ca2 = cas
        # Look for cb1 bonded to ca1 (carbon, not in macrocycle, not n) and
        # cb2 bonded to cb1 (carbon, not n, not ca1) and ca2 in adj[cb2].
        found = False
        for cb1 in adj[ca1]:
            if int(atoms[cb1]) != 6 or cb1 in calpha or cb1 == n:
                continue
            for cb2 in adj[cb1]:
                if int(atoms[cb2]) != 6 or cb2 == ca1 or cb2 == n:
                    continue
                if ca2 in adj[cb2]:
                    cbeta.add(cb1); cbeta.add(cb2)
                    found = True
                    break
            if found:
                break
        if not found:
            raise ValueError(f"Could not close pyrrole 5-ring at N {n+1}")
    if len(cbeta) != 8:
        raise ValueError(f"Expected 8 Cbeta from pyrrole 5-rings; "
                         f"found {len(cbeta)}")

    # Identify Cmeso: C atoms bonded to two Cα from DIFFERENT pyrroles.
    # First map each Cα to its pyrrole index (0..3).
    pyrrole_of = {ca: i for i, n in enumerate(n_idx)
                       for ca in pyrrole_calpha[n]}
    cmeso = set()
    for c in range(len(atoms)):
        if int(atoms[c]) != 6 or c in calpha or c in cbeta:
            continue
        ca_neighbours = [k for k in adj[c] if k in calpha]
        if len(ca_neighbours) == 2 and pyrrole_of[ca_neighbours[0]] != pyrrole_of[ca_neighbours[1]]:
            cmeso.add(c)
    if len(cmeso) != 3:
        raise ValueError(f"Expected 3 Cmeso (corrin); found {len(cmeso)}")

    # Verify the corrin closure: exactly one Cα-Cα direct bond
    direct_pairs = [(a, b) for a in calpha for b in calpha
                    if a < b and b in adj[a]]
    if len(direct_pairs) != 1:
        raise ValueError(f"Expected exactly 1 Calpha-Calpha direct bond "
                         f"(corrin closure); found {len(direct_pairs)}")

    classes = {
        "metal": [metal],
        "N": sorted(n_idx),
        "Calpha": sorted(calpha),
        "Cmeso": sorted(cmeso),
        "Cbeta": sorted(cbeta),
    }
    macro_indices = np.array(classes["N"] + classes["Calpha"]
                             + classes["Cmeso"] + classes["Cbeta"], dtype=int)
    return classes, macro_indices, direct_pairs[0]


# -------------------- analysis pipeline --------------------

def analyse_log(path, verbose=False, plot_dir=None, report_dir=None):
    """Same shape as nsd_porphyrin.analyse_log but for the 23-atom corrin ring."""
    atoms, coords = porph.parse_last_standard_orientation(path)
    classes, macro_idx, direct_pair = identify_corrin(atoms, coords)
    macro_coords = coords[macro_idx]

    centroid, normal = porph.fit_mean_plane(macro_coords)
    R = porph.rotation_to_z(normal)
    plane_coords = (macro_coords - centroid) @ R.T

    # Place the four N atoms (canonical class indices 0..3) on +-x, +-y axes
    n_local = np.arange(4)
    canonical = porph.canonicalise_xy(plane_coords, n_local)

    dz = canonical[:, 2].copy()
    basis = porph.build_oop_basis(canonical)
    amps = porph.project(dz, basis)
    recon = porph.reconstruct(amps, basis)
    residual = dz - recon
    dz_recon = recon.copy()

    # Per-irrep magnitudes (D4h-approximate for corrin)
    irrep_mag = {
        "B2u (sad)": abs(amps["sad"]),
        "B1u (ruf)": abs(amps["ruf"]),
        "A2u (dom)": abs(amps["dom"]),
        "Eg (wav)":  float(np.hypot(amps["wav_x"], amps["wav_y"])),
        "A1u (pro)": abs(amps["pro"]),
    }

    d_total_oop = float(np.linalg.norm(dz))
    d_minbasis = float(np.linalg.norm(recon))
    rms_dev = float(np.sqrt(np.mean(residual ** 2)))
    max_abs_dz = float(np.max(np.abs(dz)))

    metal_z = int(atoms[classes["metal"][0]])
    metal_sym = porph.ELEMENT_SYMBOLS.get(metal_z, f"Z{metal_z}")

    metal_idx = classes["metal"][0]
    metal_in_plane = (coords[metal_idx] - centroid) @ R.T
    metal_to_plane = float(metal_in_plane[2])
    M_N_distances = [float(np.linalg.norm(coords[metal_idx] - coords[n]))
                     for n in classes["N"]]
    M_N_mean = float(np.mean(M_N_distances))

    # Axial ligand detection: any non-macrocycle heavy atom (O, N, S, C)
    # within 2.8 Å of the metal counts as axially coordinated. Reported is
    # the closest above-plane and below-plane neighbour (in canonical frame),
    # together with the heteroatom count.
    macro_set = set(int(i) for i in macro_idx)
    axial_above = None  # (element, distance, signed_z_in_canonical_frame)
    axial_below = None
    n_axial = 0
    AXIAL_CUTOFF = 2.80
    for j, z_atomic in enumerate(atoms):
        if j == metal_idx or j in macro_set:
            continue
        if int(z_atomic) == 1:  # ignore H
            continue
        d = float(np.linalg.norm(coords[j] - coords[metal_idx]))
        if d > AXIAL_CUTOFF:
            continue
        n_axial += 1
        # In canonical frame the macrocycle plane is the xy plane and the
        # metal sits on the z axis at z = metal_to_plane. Use the macrocycle
        # mean plane (centroid + R) to determine "above" vs "below":
        atom_in_plane = (coords[j] - centroid) @ R.T
        z_above = float(atom_in_plane[2])
        sym = porph.ELEMENT_SYMBOLS.get(int(z_atomic), f"Z{int(z_atomic)}")
        if z_above >= metal_to_plane:
            if axial_above is None or d < axial_above[1]:
                axial_above = (sym, d, z_above)
        else:
            if axial_below is None or d < axial_below[1]:
                axial_below = (sym, d, z_above)

    # Conformation label (analytic D4h labels, approximate for corrin)
    conf_oop = porph.describe_oop_conformation(amps)

    # In-plane analysis. Class labels for 23-atom corrin macrocycle.
    class_array = (["N"] * 4 + ["Calpha"] * 8 + ["Cmeso"] * 3 + ["Cbeta"] * 8)
    ip = porph.analyse_in_plane(canonical, class_array)
    ip_amps = ip["amps"]
    ip_irrep = ip["irrep_mag"]
    radii = ip["class_radii"]
    conf_ip = porph.describe_ip_conformation(ip_amps)

    result = {
        "file": Path(path).name,
        "metal": metal_sym,
        "n_atoms_total": len(atoms),
        "conformation_oop": conf_oop,
        "conformation_ip":  conf_ip,
        "max_abs_dz_A": max_abs_dz,
        "rms_dz_A": float(np.sqrt(np.mean(dz ** 2))),
        "D_oop_obs_A": d_total_oop,
        "D_oop_minbasis_A": d_minbasis,
        "frac_explained_oop": (d_minbasis / d_total_oop) if d_total_oop > 1e-10 else 1.0,
        "rms_residual_oop_A": rms_dev,
        "d_sad_A": amps["sad"],
        "d_ruf_A": amps["ruf"],
        "d_dom_A": amps["dom"],
        "d_wav_x_A": amps["wav_x"],
        "d_wav_y_A": amps["wav_y"],
        "d_pro_A": amps["pro"],
        "Mag_B2u_sad_A": irrep_mag["B2u (sad)"],
        "Mag_B1u_ruf_A": irrep_mag["B1u (ruf)"],
        "Mag_A2u_dom_A": irrep_mag["A2u (dom)"],
        "Mag_Eg_wav_A":  irrep_mag["Eg (wav)"],
        "Mag_A1u_pro_A": irrep_mag["A1u (pro)"],
        "D_ip_obs_A": ip["D_ip_obs"],
        "D_ip_minbasis_A": ip["D_ip_min"],
        "frac_explained_ip": ip["frac_explained_ip"],
        "rms_residual_ip_A": ip["rms_residual_ip"],
        "d_bre_A": ip_amps["bre"],
        "d_Nstr_A": ip_amps["Nstr"],
        "d_mstr_A": ip_amps["mstr"],
        "d_rot_A": ip_amps["rot"],
        "d_trn_x_A": ip_amps["trn_x"],
        "d_trn_y_A": ip_amps["trn_y"],
        "Mag_A1g_bre_A":  ip_irrep["A1g (bre)"],
        "Mag_B1g_Nstr_A": ip_irrep["B1g (Nstr)"],
        "Mag_B2g_mstr_A": ip_irrep["B2g (mstr)"],
        "Mag_A2g_rot_A":  ip_irrep["A2g (rot)"],
        "Mag_Eu_trn_A":   ip_irrep["Eu (trn)"],
        "M_to_plane_A": metal_to_plane,
        "M_N_mean_A":   M_N_mean,
        "r_M_N_A":     radii["N"],
        "r_Calpha_A":  radii["Calpha"],
        "r_Cmeso_A":   radii["Cmeso"],
        "r_Cbeta_A":   radii["Cbeta"],
        # Axial ligand info (if any non-macrocycle heavy atom within 2.8 A of metal)
        "n_axial": n_axial,
        "axial_above_atom":  axial_above[0]   if axial_above else "",
        "axial_above_dist_A": axial_above[1]   if axial_above else None,
        "axial_below_atom":  axial_below[0]   if axial_below else "",
        "axial_below_dist_A": axial_below[1]   if axial_below else None,
    }

    if verbose:
        _print_report(result, classes, atoms, canonical, dz, recon, ip,
                      source_path=path)

    if report_dir is not None:
        out_path = _save_report(result, classes, atoms, canonical, dz, recon, ip,
                                report_dir, source_path=path)
        if verbose:
            print(f"  Wrote report: {out_path}")

    if plot_dir is not None:
        import nsd_plot
        plot_dir_p = Path(plot_dir)
        plot_dir_p.mkdir(parents=True, exist_ok=True)
        stem = Path(path).stem
        class_labels = ["N"] * 4 + ["Cα"] * 8 + ["Cmeso"] * 3 + ["Cβ"] * 8
        oop_modes = ["sad\n(B2u)", "ruf\n(B1u)", "dom\n(A2u)",
                     "wav(x)\n(Eg)", "wav(y)\n(Eg)", "pro\n(A1u)"]
        oop_amps = [amps["sad"], amps["ruf"], amps["dom"],
                    amps["wav_x"], amps["wav_y"], amps["pro"]]
        ip_modes = ["bre\n(A1g)", "N-str\n(B1g)", "m-str\n(B2g)",
                    "rot\n(A2g)", "trn(x)\n(Eu)", "trn(y)\n(Eu)"]
        ip_vals = [ip_amps["bre"], ip_amps["Nstr"], ip_amps["mstr"],
                   ip_amps["rot"], ip_amps["trn_x"], ip_amps["trn_y"]]
        nsd_plot.plot_per_file_combined(
            canonical[:, :2], dz, dz_recon, class_labels,
            oop_modes, oop_amps,
            f"{stem} -- corrin OOP NSD ({metal_sym})",
            plot_dir_p / f"{stem}_OOP.png")
        nsd_plot.plot_amplitudes(
            ip_modes, ip_vals,
            f"{stem} -- corrin in-plane minimal-basis amplitudes ({metal_sym})",
            plot_dir_p / f"{stem}_IP.png")

    return result


def _print_report(r, classes, atoms, canonical, dz, recon, ip, source_path=None):
    """Backwards-compatible wrapper — calls porph.format_report with corrin
    parameters and prints the result."""
    axial_info = {
        "n_axial":             r.get("n_axial", 0),
        "axial_above_atom":    r.get("axial_above_atom"),
        "axial_above_dist_A":  r.get("axial_above_dist_A"),
        "axial_below_atom":    r.get("axial_below_atom"),
        "axial_below_dist_A":  r.get("axial_below_dist_A"),
    }
    print(porph.format_report(
        r, classes, atoms, canonical, dz, recon, ip,
        source_path=source_path,
        macrocycle_name="corrin",
        non_d4h_caveat=True,
        axial_info=axial_info,
    ))


def _save_report(r, classes, atoms, canonical, dz, recon, ip, report_dir, source_path=None):
    """Write the per-file corrin report as <stem>.report.txt under report_dir."""
    axial_info = {
        "n_axial":             r.get("n_axial", 0),
        "axial_above_atom":    r.get("axial_above_atom"),
        "axial_above_dist_A":  r.get("axial_above_dist_A"),
        "axial_below_atom":    r.get("axial_below_atom"),
        "axial_below_dist_A":  r.get("axial_below_dist_A"),
    }
    from pathlib import Path as _P
    _P(report_dir).mkdir(parents=True, exist_ok=True)
    stem = _P(r['file']).stem
    out = _P(report_dir) / f"{stem}.report.txt"
    text = porph.format_report(
        r, classes, atoms, canonical, dz, recon, ip,
        source_path=source_path,
        macrocycle_name="corrin",
        non_d4h_caveat=True,
        axial_info=axial_info,
    )
    out.write_text(text)
    return out


_BANNER = (
    "================================================================\n"
    " nsd_corrin.py -- GEOMETRY-ONLY corrin NSD analysis\n"
    " Uses analytic D4h-pattern basis on the 23-atom corrin macrocycle.\n"
    " Conformation labels are D4h-approximate (corrin is C1).\n"
    " Geometric quantities (|D_oop|, |D_ip|, M-plane, M-N, class radii)\n"
    " are rigorous. For a freq-derived basis: nsd_corrin_freq.py.\n"
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
            r = analyse_log(f, verbose=not args.quiet, plot_dir=args.plot,
                            report_dir=args.report)
            rows.append(r)
        except Exception as e:
            print(f"!! {f.name}: {e}", file=sys.stderr)

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
        print(" Conformation summary (D4h-approximate labels for corrin):")
        print(f"  {'file':28s} {'metal':>3s}   {'OOP conformation':<35s} {'IP conformation':<25s}")
        for r in rows:
            print(f"  {r['file']:28s} {r['metal']:>3s}   "
                  f"{r['conformation_oop']:<35s} {r['conformation_ip']:<25s}")
        print("\n" + "=" * 100)
        print(" Distortion magnitudes (Angstrom):")
        any_axial = any(r["n_axial"] > 0 for r in rows)
        if any_axial:
            print(" file                         metal max|dz| |D_oop| |D_ip| M-plane M-N    axial(above/below)")
            for r in rows:
                ab = (f"{r['axial_above_atom']}{r['axial_above_dist_A']:.2f}"
                      if r['axial_above_dist_A'] is not None else "—")
                bb = (f"{r['axial_below_atom']}{r['axial_below_dist_A']:.2f}"
                      if r['axial_below_dist_A'] is not None else "—")
                print(f"  {r['file']:28s} {r['metal']:>3s}   "
                      f"{r['max_abs_dz_A']: .3f}  {r['D_oop_obs_A']: .3f}  "
                      f"{r['D_ip_obs_A']: .3f}  "
                      f"{r['M_to_plane_A']:+.2f}  {r['M_N_mean_A']: .3f}  "
                      f"{ab:>7s} / {bb:>7s}")
        else:
            print(" file                         metal   max|dz| |D_oop| |D_ip|  M-plane M-N    %expl")
            for r in rows:
                print(f"  {r['file']:28s} {r['metal']:>3s}    "
                      f"{r['max_abs_dz_A']: .3f}  {r['D_oop_obs_A']: .3f}  "
                      f"{r['D_ip_obs_A']: .3f}  "
                      f"{r['M_to_plane_A']:+.2f}   {r['M_N_mean_A']: .3f}  "
                      f"{100*r['frac_explained_oop']:5.1f}")

    if args.plot and rows:
        import nsd_plot
        plot_dir_p = Path(args.plot)
        plot_dir_p.mkdir(parents=True, exist_ok=True)
        labels = [f"{r['file'].replace('.log','')}\n({r['metal']})" for r in rows]
        oop_modes = ["sad", "ruf", "dom", "wav(x)", "wav(y)", "pro"]
        oop_keys = ["d_sad_A", "d_ruf_A", "d_dom_A",
                    "d_wav_x_A", "d_wav_y_A", "d_pro_A"]
        oop_mat = np.array([[r[k] for k in oop_keys] for r in rows])
        nsd_plot.plot_summary(labels, oop_mat, oop_modes,
                              "Corrin -- OOP minimal-basis amplitudes (D4h-approximate, geometry-only)",
                              plot_dir_p / "_summary_OOP.png")
        ip_modes = ["bre", "N-str", "m-str", "rot", "trn(x)", "trn(y)"]
        ip_keys = ["d_bre_A", "d_Nstr_A", "d_mstr_A",
                   "d_rot_A", "d_trn_x_A", "d_trn_y_A"]
        ip_mat = np.array([[r[k] for k in ip_keys] for r in rows])
        nsd_plot.plot_summary(labels, ip_mat, ip_modes,
                              "Corrin -- in-plane minimal-basis amplitudes (D4h-approximate, geometry-only)",
                              plot_dir_p / "_summary_IP.png")

        # Discriminator scatter plots: saddled vs domed, and saddled vs propellered
        short_labels = [r["file"].replace(".log", "").replace("corrinOH", "")
                        for r in rows]
        metals = [r["metal"] for r in rows]
        sad = [float(r["d_sad_A"]) for r in rows]
        dom = [float(r["d_dom_A"]) for r in rows]
        pro = [float(r["d_pro_A"]) for r in rows]
        nsd_plot.plot_mode_scatter(
            short_labels, sad, dom, "saddled (sad, B2u)", "domed (dom, A2u)",
            "Corrin -- saddled vs domed (Co5 / Ni4 are the high-distortion outliers)",
            plot_dir_p / "_scatter_sad_dom.png", metal_per_file=metals)
        nsd_plot.plot_mode_scatter(
            short_labels, sad, pro, "saddled (sad, B2u)", "propellered (pro, A1u)",
            "Corrin -- saddled vs propellered (sign of pro flips for Co5 / Ni4)",
            plot_dir_p / "_scatter_sad_pro.png", metal_per_file=metals)
        print(f"\nWrote plots to {plot_dir_p}")


if __name__ == "__main__":
    main()
