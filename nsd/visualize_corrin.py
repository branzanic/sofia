#!/usr/bin/env python3
"""Build a PyMOL visualization bundle for the corrin minimal-basis modes.

Generates a planar metallo-corrin reference plus one distorted variant per
OOP / IP mode under <out>/, and two PyMOL scripts (corrin_oop_modes.pml,
corrin_ip_modes.pml) that load the bundle into a side-by-side grid.

Caveat: corrin is C1 symmetry — its 23-atom inner ring has 4 N, 8 Cα, only
3 Cmeso, and 8 Cβ. Sofia's D4h-pattern analytic basis still produces a
well-defined polynomial Δz = f(x, y) on these 23 atoms, so the modes carry
meaningful (basis-vector) shapes; the irrep labels are D4h-approximate
because corrin is not actually D4h. This is documented in the .pml header
and in Manual.md.

Usage (as a module):
    python3 -m nsd.visualize_corrin                 # uses bundled examples/corrin/CoI_Corrin_S1.log
    python3 -m nsd.visualize_corrin --log my.log    # custom reference geometry
    python3 -m nsd.visualize_corrin --amp 0.7       # bigger distortion (default 0.5 Å)
    python3 -m nsd.visualize_corrin --out my_viz    # custom output directory

The bundled default reference is bare Co(I) corrin (S=0 singlet) — the
canonical Co(I) corrinoid geometry. We use this rather than di-OH corrin
because the bare Co(I) corrin is the chemistry-textbook reference and has
no axial ligands cluttering the viz.
"""

import argparse
import sys
from pathlib import Path

# Ensure sibling modules are on sys.path when run as a script
_SELF_DIR = Path(__file__).resolve().parent
if str(_SELF_DIR) not in sys.path:
    sys.path.insert(0, str(_SELF_DIR))

import nsd_corrin as corrin
import visualize_common as vc
from visualize_common import _ELEM


# Mode tables — same names as porphyrin viz so users navigating between the
# two see consistent terminology. dom_pure variant kept (Shelnutt convention
# vs deoxy-Hb hybrid) — same logic applies to corrin.
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
    ("mstr",    "B2g",  "magenta",      "meso-stretched"),
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
    "Corrin is C1 symmetry (23-atom ring with only 3 Cmeso). Sofia's\n"
    "D4h-pattern analytic minimal basis still produces well-defined\n"
    "polynomial shapes Δz = f(x,y) on these 23 atoms, but the irrep\n"
    "labels (B2u, B1u, A2u, …) are D4h-approximate — corrin is not D4h."
)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--log", metavar="FILE",
                    help="Reference Gaussian .log "
                         "(default: bundled examples/corrin/CoI_Corrin_S1.log)")
    ap.add_argument("--amp", metavar="A", type=float, default=0.5,
                    help="Distortion amplitude in Å (default: 0.5)")
    ap.add_argument("--out", metavar="DIR", default="viz/corrin",
                    help="Output directory (default: viz/corrin)")
    args = ap.parse_args()

    if args.log:
        log_path = Path(args.log)
    else:
        repo_root = _SELF_DIR.parent
        log_path = repo_root / "examples" / "corrin" / "CoI_Corrin_S1.log"
    if not log_path.exists():
        sys.exit(f"reference log not found: {log_path}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reference geometry : {log_path}")
    print(f"Distortion amplitude: {args.amp:.2f} Å")
    print(f"Output directory   : {out_dir}")
    print()

    (atoms, coords_planar, macro_idx, classes, parent_of_H, canonical_xy,
     substituent_parents) = vc.make_planar_reference(log_path,
                                                     corrin.identify_corrin)
    metal_z = int(atoms[classes['metal'][0]])
    metal_sym = _ELEM.get(metal_z, f"Z{metal_z}")

    # Compute bonds once on the planar reference, attach the same CONECT
    # records to every distorted variant. Guarantees PyMOL draws the full
    # corrin skeleton at any amplitude (including across the Cα–Cα closure).
    bonds = vc.compute_bonds(atoms, coords_planar)

    # 1. Planar reference
    planar_pdb = "corrin_planar.pdb"
    vc.write_pdb(out_dir / planar_pdb, atoms, coords_planar, bonds,
                 f"Planar {metal_sym}-corrin reference (Sofia)")
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
        fname = f"corrin_{name}.pdb"
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
        fname = f"corrin_ip_{name}.pdb"
        vc.write_pdb(out_dir / fname, atoms, coords_d, bonds,
                     f"{name} ({irrep}) -- {descr}; amplitude {args.amp:.2f} A")
        ip_mode_filenames[name] = fname
        print(f"  wrote {fname:<24s} ({irrep}) — {descr}")

    # 4. PyMOL scripts
    pml_path = vc.write_oop_pml(out_dir, mode_filenames, planar_pdb, args.amp,
                                prefix="corrin", layout=_OOP_LAYOUT,
                                label_macrocycle="corrin",
                                header_extra=_HEADER_EXTRA)
    ip_pml_path = vc.write_ip_pml(out_dir, ip_mode_filenames, planar_pdb, args.amp,
                                  prefix="corrin", layout=_IP_LAYOUT,
                                  label_macrocycle="corrin",
                                  header_extra=_HEADER_EXTRA)
    print()
    print("Open in PyMOL:")
    print(f"  pymol {pml_path}        # OOP modes (Δz)")
    print(f"  pymol {ip_pml_path}     # IP modes  (Δx, Δy)")


if __name__ == "__main__":
    main()
