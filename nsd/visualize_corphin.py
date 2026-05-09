#!/usr/bin/env python3
"""Build a PyMOL visualization bundle for corphin / F430 minimal-basis modes.

Corphin shares porphyrin's 24-atom inner ring (4 N + 8 Cα + 4 Cmeso + 8 Cβ),
so the analytic D4h-pattern minimal basis applies directly. The visual
difference vs. plain metalloporphine is the heavy peripheral substituent
shell (methyl groups, propionamide chains, the F-ring on F430), which we
carry rigidly with the inner ring under each distortion: every peripheral
atom is mapped to its nearest inner-ring ancestor(s) by BFS and inherits
the average of their Δz / Δxy.

Caveat: corphin / F430 is not D4h. The saturation pattern (sp³ carbons in
one or more pyrroles, plus the fused F-ring) breaks four-fold symmetry, so
the irrep labels (B2u, B1u, A2u, …) are approximate classifications. The
distortion patterns Sofia computes are still well-defined geometric
projections onto a fixed analytic basis — they're useful for cross-comparison
across metals and spin states, but they're not pure normal-mode amplitudes.
This is documented in the .pml header and in Manual.md.

Usage (as a module):
    python3 -m nsd.visualize_corphin                 # uses bundled examples/corphin/F430_Ni2.log
    python3 -m nsd.visualize_corphin --log my.log    # custom reference geometry
    python3 -m nsd.visualize_corphin --amp 0.7       # bigger distortion (default 0.5 Å)
    python3 -m nsd.visualize_corphin --out my_viz    # custom output directory

The bundled default reference is F430_Ni2 — the F430-native S=½ Ni state
(the least distorted of our three F430 examples; cleanest planar baseline).
"""

import argparse
import sys
from pathlib import Path

# Ensure sibling modules are on sys.path when run as a script
_SELF_DIR = Path(__file__).resolve().parent
if str(_SELF_DIR) not in sys.path:
    sys.path.insert(0, str(_SELF_DIR))

import nsd_porphyrin as porph
import visualize_common as vc
from visualize_common import _ELEM


_MODES = [
    ("sad",        "B2u",  "salmon",       "saddled — pyrroles tilt alternately up/down"),
    ("ruf",        "B1u",  "skyblue",      "ruffled — meso C alternate up/down"),
    ("dom",        "A2u",  "lightorange",  "domed (deoxy-Hb hybrid: A2u + Tz + metal-out-of-plane)"),
    ("dom_pure",   "A2u",  "wheat",        "domed (pure A2u; Tz-orthogonal — Shelnutt convention)"),
    ("wav_x",      "Eg",   "limegreen",    "waved along x"),
    ("wav_y",      "Eg",   "violet",       "waved along y"),
    ("pro",        "A1u",  "yellow",       "propellered — each pyrrole twists about its own axis"),
]

_IP_MODES = [
    ("bre",     "A1g",  "paleyellow",   "breathing — uniform radial expansion"),
    ("Nstr",    "B1g",  "cyan",         "N-stretched — opposite-N pairs ↔ trans-N–N stretch"),
    ("mstr",    "B2g",  "magenta",      "meso-stretched — opposite-meso pairs ↔ trans-meso stretch"),
    ("rot",     "A2g",  "orange",       "rotated — in-phase rotation of the four pyrroles"),
    ("trn_x",   "Eu",   "red",          "translated along x — macrocycle shifts vs metal"),
    ("trn_y",   "Eu",   "green",        "translated along y — macrocycle shifts vs metal"),
]


_OOP_LAYOUT = [
    ("planar",    0,  0, "gray70",      "planar reference"),
    ("sad",       1,  0, "salmon",      "sad   (B2u)"),
    ("ruf",       2,  0, "skyblue",     "ruf   (B1u)"),
    ("dom",       3,  0, "lightorange", "dom (deoxy-Hb)"),
    ("wav_x",     0, -1, "limegreen",   "wav_x (Eg)"),
    ("wav_y",     1, -1, "violet",      "wav_y (Eg)"),
    ("pro",       2, -1, "yellow",      "pro   (A1u)"),
    ("dom_pure",  3, -1, "wheat",       "dom (A2u pure)"),
]

_IP_LAYOUT = [
    ("planar",    0,  0, "gray70",     "planar reference"),
    ("bre",       1,  0, "paleyellow", "bre   (A1g)"),
    ("Nstr",      2,  0, "cyan",       "N-str (B1g)"),
    ("mstr",      3,  0, "magenta",    "m-str (B2g)"),
    ("rot",       1, -1, "orange",     "rot   (A2g)"),
    ("trn_x",     2, -1, "red",        "trn_x (Eu)"),
    ("trn_y",     3, -1, "green",      "trn_y (Eu)"),
]


_HEADER_EXTRA = (
    "Corphin / F430 is not D4h. Saturation pattern (sp³ pyrrole carbons,\n"
    "fused F-ring) breaks four-fold symmetry, so the irrep labels are\n"
    "D4h-approximate. Peripheral substituents (methyls, propionamides,\n"
    "F-ring) move rigidly with their nearest inner-ring atom under each\n"
    "distortion (BFS-based propagation)."
)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--log", metavar="FILE",
                    help="Reference Gaussian .log "
                         "(default: bundled examples/corphin/F430_Ni2.log)")
    ap.add_argument("--amp", metavar="A", type=float, default=0.5,
                    help="Distortion amplitude in Å (default: 0.5)")
    ap.add_argument("--out", metavar="DIR", default="viz/corphin",
                    help="Output directory (default: viz/corphin)")
    args = ap.parse_args()

    if args.log:
        log_path = Path(args.log)
    else:
        repo_root = _SELF_DIR.parent
        log_path = repo_root / "examples" / "corphin" / "F430_Ni2.log"
    if not log_path.exists():
        sys.exit(f"reference log not found: {log_path}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reference geometry : {log_path}")
    print(f"Distortion amplitude: {args.amp:.2f} Å")
    print(f"Output directory   : {out_dir}")
    print()

    # Corphin shares porphyrin's inner-ring topology, so we use the porphyrin
    # identifier (4 N + 8 Cα + 4 Cmeso + 8 Cβ) directly.
    (atoms, coords_planar, macro_idx, classes, parent_of_H, canonical_xy,
     substituent_parents) = vc.make_planar_reference(log_path,
                                                     porph.identify_macrocycle)
    metal_z = int(atoms[classes['metal'][0]])
    metal_sym = _ELEM.get(metal_z, f"Z{metal_z}")
    print(f"  Inner-ring atoms : 24 ({metal_sym} centre)")
    print(f"  Peripheral atoms : {len(substituent_parents)} "
          f"(carried rigidly via BFS to nearest inner-ring atom)")
    print()

    # Compute bonds once on the planar reference; same CONECT records get
    # attached to every distorted variant.
    bonds = vc.compute_bonds(atoms, coords_planar)

    # 1. Planar reference
    planar_pdb = "corphin_planar.pdb"
    vc.write_pdb(out_dir / planar_pdb, atoms, coords_planar, bonds,
                 f"Planar {metal_sym}-corphin reference (Sofia)")
    print(f"  wrote {planar_pdb}")

    # 2. OOP modes
    print()
    print("Out-of-plane modes:")
    mode_filenames = {}
    for name, irrep, _color, descr in _MODES:
        coords_d = vc.apply_oop_mode(coords_planar, macro_idx, parent_of_H,
                                     canonical_xy, name, args.amp,
                                     metal_idx=classes['metal'][0],
                                     substituent_parents=substituent_parents)
        fname = f"corphin_{name}.pdb"
        vc.write_pdb(out_dir / fname, atoms, coords_d, bonds,
                     f"{name} ({irrep}) -- {descr}; amplitude {args.amp:.2f} A")
        mode_filenames[name] = fname
        print(f"  wrote {fname:<24s} ({irrep}) — {descr}")

    # 3. IP modes
    print()
    print("In-plane modes:")
    ip_mode_filenames = {}
    for name, irrep, _color, descr in _IP_MODES:
        coords_d = vc.apply_ip_mode(coords_planar, macro_idx, parent_of_H,
                                    canonical_xy, name, args.amp,
                                    metal_idx=classes['metal'][0],
                                    substituent_parents=substituent_parents)
        fname = f"corphin_ip_{name}.pdb"
        vc.write_pdb(out_dir / fname, atoms, coords_d, bonds,
                     f"{name} ({irrep}) -- {descr}; amplitude {args.amp:.2f} A")
        ip_mode_filenames[name] = fname
        print(f"  wrote {fname:<24s} ({irrep}) — {descr}")

    # 4. PyMOL scripts
    pml_path = vc.write_oop_pml(out_dir, mode_filenames, planar_pdb, args.amp,
                                prefix="corphin", layout=_OOP_LAYOUT,
                                label_macrocycle="corphin/F430",
                                header_extra=_HEADER_EXTRA)
    ip_pml_path = vc.write_ip_pml(out_dir, ip_mode_filenames, planar_pdb, args.amp,
                                  prefix="corphin", layout=_IP_LAYOUT,
                                  label_macrocycle="corphin/F430",
                                  header_extra=_HEADER_EXTRA)
    print()
    print("Open in PyMOL:")
    print(f"  pymol {pml_path}        # OOP modes (Δz)")
    print(f"  pymol {ip_pml_path}     # IP modes  (Δx, Δy)")


if __name__ == "__main__":
    main()
