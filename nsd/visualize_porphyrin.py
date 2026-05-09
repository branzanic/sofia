#!/usr/bin/env python3
"""Build a PyMOL visualization bundle for the porphyrin OOP minimal-basis modes.

Generates seven .xyz files under <out>/ — one planar reference porphine plus one
distorted variant per mode (sad / ruf / dom / wav_x / wav_y / pro) — and a
porphyrin_oop_modes.pml that loads all seven into PyMOL, arranges them in a grid,
labels each by mode, and sets up a consistent rendering.

The reference geometry is taken from a converged metallo-porphine .log file
(Sofia's example Co1_Pro.log by default). Inner-ring atoms are forced to
z = 0 for a clean planar baseline, and any bonded H atoms are translated
along with their parent C so C–H bonds remain intact under the distortion.

Usage (as a module):
    python3 -m nsd.visualize_porphyrin                # uses bundled example
    python3 -m nsd.visualize_porphyrin --log my.log   # custom reference geometry
    python3 -m nsd.visualize_porphyrin --amp 0.7      # bigger distortion (default 0.5 Å)
    python3 -m nsd.visualize_porphyrin --out my_viz   # custom output dir
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Ensure sibling modules are on sys.path when run as a script
_SELF_DIR = Path(__file__).resolve().parent
if str(_SELF_DIR) not in sys.path:
    sys.path.insert(0, str(_SELF_DIR))

import nsd_porphyrin as porph


_ELEM = {1: "H", 6: "C", 7: "N", 8: "O", 16: "S",
         23: "V", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co",
         28: "Ni", 29: "Cu", 30: "Zn", 45: "Rh"}


def _make_planar_reference(log_path):
    """Read a Gaussian log, extract the inner ring + bonded H, force z=0 on
    the inner ring (and shift each bonded H by the same amount as its parent
    C so C–H bonds stay rigid). Return (atoms, coords, macro_idx, classes)."""
    atoms, coords = porph.parse_last_standard_orientation(str(log_path))
    classes, macro_idx = porph.identify_macrocycle(atoms, coords)

    # Mean-plane fit on the 24 inner-ring atoms — rotate the whole molecule so
    # that the macrocyclic plane coincides with z = 0.
    centroid, normal = porph.fit_mean_plane(coords[macro_idx])
    R = porph.rotation_to_z(normal)
    coords = (coords - centroid) @ R.T

    # Canonicalise xy so N atoms sit on ±x, ±y axes
    canonical_xy = porph.canonicalise_xy(coords[macro_idx], np.arange(4))
    # Re-derive the rotation that maps the 24-atom array onto canonical_xy,
    # apply the same rotation to the WHOLE molecule (including H atoms).
    src = coords[macro_idx][:, :2]
    tgt = canonical_xy[:, :2]
    # Procrustes: 2D rotation, no scaling; assume centroid already at origin
    H_mat = src.T @ tgt
    U, _, Vt = np.linalg.svd(H_mat)
    R2 = Vt.T @ U.T
    if np.linalg.det(R2) < 0:
        Vt[-1] *= -1
        R2 = Vt.T @ U.T
    # Rotate xy of all atoms; leave z untouched
    coords[:, :2] = coords[:, :2] @ R2.T

    # Build adjacency to find which H is bonded to each C-class atom
    adj = porph.build_adjacency(atoms, coords)

    # Determine the planar reference: zero the z of all inner-ring atoms,
    # and shift each bonded H by the same Δz as its parent C (so C–H bonds
    # rotate cleanly with their parent later when we apply distortions).
    parent_of_H = {}  # H_index -> macrocycle_index
    for i in range(len(atoms)):
        if int(atoms[i]) == 1:
            for j in adj[i]:
                if j in macro_idx:
                    parent_of_H[i] = j
                    break

    coords_planar = coords.copy()
    # Subtract the residual z of each inner-ring atom from itself and from its
    # bonded H (so the C and H both move down to the plane together)
    for k_macro, mi in enumerate(macro_idx):
        dz_residual = coords[mi, 2]
        coords_planar[mi, 2] = 0.0
        for h, parent in parent_of_H.items():
            if parent == mi:
                coords_planar[h, 2] -= dz_residual

    return atoms, coords_planar, macro_idx, classes, parent_of_H, canonical_xy


def _apply_oop_mode(coords_planar, macro_idx, parent_of_H, canonical_xy,
                    mode_name, amplitude, metal_idx=None):
    """Add Δz from one OOP mode to the inner ring + its bonded H atoms.
    Returns a new (N, 3) coords array.

    For sad / ruf / wav_x / wav_y / pro: applies Sofia's analytic D4h-pattern
    minimal-basis vector directly. These modes are sign-symmetric (atoms move
    above and below the plane in equal measure), so the result is what a
    chemist would draw.

    For 'dom' (A2u): Sofia's analytic basis uses the +r² generator with the
    rigid-body Tz mode projected out — that's correct for the math (orthogonal
    basis, well-defined amplitude) but produces a counter-intuitive picture
    where N atoms (small r²) dip *below* the plane while Cβ (large r²) rise
    *above*. To match the textbook chemistry picture (e.g. deoxyhemoglobin
    where the metal sits at the apex of a dome and the macrocycle curves
    *toward* the metal, with N atoms tracking the metal motion), we render a
    visualization-only pattern of Δz ∝ (r²_max − r²): all atoms move up, N
    atoms move most, Cβ remain near the original plane. The metal then gets
    an additional lift so it visibly sits at the apex of the dome. The
    analysis pipeline still uses Sofia's analytic basis as-is — this affects
    only the visualization geometry.
    """
    coords_out = coords_planar.copy()

    if mode_name == "dom":
        # Chemistry-textbook "deoxy-Hb dome" picture (NOT a pure normal mode).
        # This is a hybrid distortion: A2u shape + rigid-body Tz on the ring +
        # metal-out-of-plane displacement. Useful for recognising classical
        # doming in real molecules. To see what Shelnutt's pure A2u eigenvector
        # actually looks like, use the 'dom_pure' variant.
        r2 = canonical_xy[:, 0] ** 2 + canonical_xy[:, 1] ** 2
        raw = r2.max() - r2
        pattern = raw / np.linalg.norm(raw)
        dz_pattern = pattern * amplitude

        for k_macro, mi in enumerate(macro_idx):
            dz = float(dz_pattern[k_macro])
            coords_out[mi, 2] += dz
            for h, parent in parent_of_H.items():
                if parent == mi:
                    coords_out[h, 2] += dz

        # Lift the metal so it sits above the N atoms (deoxy-Hb apex).
        if metal_idx is not None:
            n_indices = macro_idx[:4]
            n_max_z = max(coords_out[ni, 2] for ni in n_indices)
            coords_out[metal_idx, 2] = n_max_z + 0.25 * amplitude

        return coords_out

    # All other modes (incl. dom_pure): use Sofia's analytic basis directly.
    # For 'dom_pure' this gives the Tz-orthogonal A2u eigenvector approximation
    # (Cβ above the plane, N below — the genuine A2u shape, faithful to
    # Shelnutt's lowest-A2u normal mode in sign convention).
    basis_key = "dom" if mode_name == "dom_pure" else mode_name
    basis = porph.build_oop_basis(canonical_xy)
    if basis_key not in basis:
        raise KeyError(f"unknown mode {mode_name!r}; "
                       f"choose from {list(basis.keys()) + ['dom_pure']}")
    dz_pattern = basis[basis_key] * amplitude

    for k_macro, mi in enumerate(macro_idx):
        dz = float(dz_pattern[k_macro])
        coords_out[mi, 2] += dz
        for h, parent in parent_of_H.items():
            if parent == mi:
                coords_out[h, 2] += dz

    return coords_out


def _apply_ip_mode(coords_planar, macro_idx, parent_of_H, canonical_xy,
                   mode_name, amplitude, metal_idx=None):
    """Add (Δx, Δy) from one IP mode to the inner ring + bonded H atoms.

    Sofia's analytic IP basis is used as-is for N-str (B1g), m-str (B2g),
    and rot (A2g) — these are sign-symmetric shape modes that look correct
    when applied directly. For bre (A1g) and trn_x / trn_y (Eu), the
    visualization uses chemistry-intuitive patterns rather than Sofia's
    analytic basis:

      • bre: Sofia's analytic amplitude is ≈ 0 by construction (orbit-symmetric
        IP reference), so showing the basis vector verbatim would be invisible.
        For viz we use a uniform radial outward expansion — what 'breathing'
        actually means in chemistry parlance.

      • trn_x / trn_y: Sofia's analytic basis (after Tx/Ty projection) is an
        r²-weighted wobble of the ring relative to the metal. The chemistry
        interpretation of 'macrocycle translation' is just a uniform shift of
        the ring while the metal stays put — that's what we show here.

    Bonded H atoms move with their parent C so C–H bonds stay rigid.
    """
    coords_out = coords_planar.copy()

    if mode_name == "bre":
        # Uniform radial outward — chemistry "breathing"
        for k_macro, mi in enumerate(macro_idx):
            x, y, _ = coords_planar[mi]
            r = (x * x + y * y) ** 0.5
            if r < 1e-9:
                continue
            ux, uy = x / r, y / r
            dx, dy = ux * amplitude, uy * amplitude
            coords_out[mi, 0] += dx
            coords_out[mi, 1] += dy
            for h, parent in parent_of_H.items():
                if parent == mi:
                    coords_out[h, 0] += dx
                    coords_out[h, 1] += dy
        return coords_out

    if mode_name in ("trn_x", "trn_y"):
        # Uniform translation of the ring (and bonded H) — metal stays put.
        if mode_name == "trn_x":
            dx, dy = amplitude, 0.0
        else:
            dx, dy = 0.0, amplitude
        for mi in macro_idx:
            coords_out[mi, 0] += dx
            coords_out[mi, 1] += dy
        for h, parent in parent_of_H.items():
            if parent in set(macro_idx):
                coords_out[h, 0] += dx
                coords_out[h, 1] += dy
        return coords_out

    # N-str, m-str, rot: use Sofia's analytic IP basis directly
    basis = porph.build_ip_basis(canonical_xy)
    if mode_name not in basis:
        raise KeyError(f"unknown IP mode {mode_name!r}; "
                       f"choose from {list(basis.keys()) + ['bre']}")
    flat = basis[mode_name] * amplitude            # length-48: (dx0,dy0,dx1,dy1,...)
    delta_xy = flat.reshape(-1, 2)                 # (24, 2)

    for k_macro, mi in enumerate(macro_idx):
        dx, dy = float(delta_xy[k_macro, 0]), float(delta_xy[k_macro, 1])
        coords_out[mi, 0] += dx
        coords_out[mi, 1] += dy
        for h, parent in parent_of_H.items():
            if parent == mi:
                coords_out[h, 0] += dx
                coords_out[h, 1] += dy
    return coords_out


def _write_xyz(path, atoms, coords, comment):
    """Write a standard XYZ file."""
    with open(path, "w") as f:
        f.write(f"{len(atoms)}\n")
        f.write(comment + "\n")
        for z, (x, y, zc) in zip(atoms, coords):
            sym = _ELEM.get(int(z), f"Z{int(z)}")
            f.write(f"{sym:<3} {x:14.6f} {y:14.6f} {zc:14.6f}\n")


def _write_pdb(path, atoms, coords, bonds, comment):
    """Write a PDB file with explicit CONECT records.

    PyMOL respects CONECT regardless of distance, so the macrocyclic skeleton
    stays drawn even when atoms are pushed far out-of-plane by a large
    distortion amplitude.

    bonds is a list of (i, j) atom-index pairs (0-based).
    """
    with open(path, "w") as f:
        f.write(f"REMARK   1 {comment}\n")
        for k, (z, (x, y, zc)) in enumerate(zip(atoms, coords), start=1):
            sym = _ELEM.get(int(z), f"Z{int(z)}")
            # PDB ATOM record. Element symbol right-justified in cols 77-78.
            # Atom name (cols 13-16): use element + index, e.g. "N   ", "C  4".
            name = f"{sym:<2}{k:>2}"[:4] if len(sym) <= 2 else sym[:4]
            f.write(
                f"HETATM{k:>5d} {name:<4s} MOL A   1    "
                f"{x:8.3f}{y:8.3f}{zc:8.3f}  1.00  0.00          {sym:>2s}\n"
            )
        # CONECT records — PDB spec allows up to 4 bonds per record, multiple
        # records per atom if it has more bonds. Build a per-atom adjacency
        # list, then emit one CONECT per (atom -> sorted neighbours) row.
        adj = {i: [] for i in range(len(atoms))}
        for i, j in bonds:
            adj[i].append(j)
            adj[j].append(i)
        for i in range(len(atoms)):
            nbrs = sorted(set(adj[i]))
            if not nbrs:
                continue
            for chunk_start in range(0, len(nbrs), 4):
                chunk = nbrs[chunk_start : chunk_start + 4]
                row = f"CONECT{i+1:>5d}"
                for j in chunk:
                    row += f"{j+1:>5d}"
                f.write(row + "\n")
        f.write("END\n")


def _compute_bonds(atoms, coords):
    """Return list of (i, j) bond pairs from the planar reference coords,
    using Sofia's covalent_threshold (same rules as the macrocycle identifier)."""
    bonds = []
    n = len(atoms)
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(coords[i] - coords[j]))
            if d < porph.covalent_threshold(int(atoms[i]), int(atoms[j])):
                bonds.append((i, j))
    return bonds


# Mode names + their visual properties (PyMOL colour, geometric blurb)
_MODES = [
    # (name,       irrep,  pymol_color,    description)
    ("sad",        "B2u",  "salmon",       "saddled — pyrroles tilt alternately up/down"),
    ("ruf",        "B1u",  "skyblue",      "ruffled — meso C alternate up/down"),
    ("dom",        "A2u",  "lightorange",  "domed (deoxy-Hb hybrid: A2u + Tz + metal-out-of-plane)"),
    ("dom_pure",   "A2u",  "wheat",        "domed (pure A2u; Tz-orthogonal — Shelnutt convention)"),
    ("wav_x",      "Eg",   "limegreen",    "waved along x"),
    ("wav_y",      "Eg",   "violet",       "waved along y"),
    ("pro",        "A1u",  "yellow",       "propellered — each pyrrole twists about its own axis"),
]


_IP_MODES = [
    # (name,    irrep,  pymol_color,    description)
    ("bre",     "A1g",  "paleyellow",   "breathing — uniform radial expansion"),
    ("Nstr",    "B1g",  "cyan",         "N-stretched — opposite-N pairs ↔ trans-N–N stretch"),
    ("mstr",    "B2g",  "magenta",      "meso-stretched — opposite-meso pairs ↔ trans-meso stretch"),
    ("rot",     "A2g",  "orange",       "rotated — in-phase rotation of the four pyrroles"),
    ("trn_x",   "Eu",   "red",          "translated along x — macrocycle shifts vs metal"),
    ("trn_y",   "Eu",   "green",        "translated along y — macrocycle shifts vs metal"),
]


def _write_pml(out_dir, mode_filenames, planar_filename, amplitude):
    """Write the PyMOL script that loads all 7 structures and arranges them
    in a 4-column grid (planar in the centre, 6 modes around it)."""
    pml = []
    pml.append(f"# Sofia — porphyrin OOP minimal-basis visualization")
    pml.append(f"# Generated by nsd/visualize_porphyrin.py")
    pml.append(f"# Each distortion is the analytic Δz pattern of one mode at")
    pml.append(f"# amplitude = {amplitude:.2f} Å, applied to a planar metallo-porphine.")
    pml.append(f"#")
    pml.append(f"# Open in PyMOL:  pymol porphyrin_oop_modes.pml")
    pml.append("")
    pml.append("reinitialize")
    pml.append("bg_color white")
    pml.append("set ray_opaque_background, on")   # solid white PNG (default has alpha)
    pml.append("set ray_shadows, 0")
    pml.append("set orthoscopic, on")
    pml.append("set stick_radius, 0.12")
    pml.append("")

    # Layout: 4 columns × 2 rows. Top row: planar + 3 sign-symmetric modes
    # (sad, ruf, dom). Bottom row: 3 more (wav_x, wav_y, pro) plus the second
    # dom variant. The two dom variants sit in the same column (col 3), one
    # above the other, so the chemistry-hybrid (top) and pure-A2u (bottom)
    # pictures are directly comparable.
    # Spacing scales gently with amplitude so big distortions don't crowd.
    spacing_x = 14.0 + max(0, amplitude - 0.5) * 4.0
    spacing_y = 14.0 + max(0, amplitude - 0.5) * 4.0
    layout = [
        # (object_name, col, row, mode_color, label_text)
        ("planar",    0,  0, "gray70",      "planar reference"),
        ("sad",       1,  0, "salmon",      "sad   (B2u)"),
        ("ruf",       2,  0, "skyblue",     "ruf   (B1u)"),
        ("dom",       3,  0, "lightorange", "dom (deoxy-Hb)"),
        ("wav_x",     0, -1, "limegreen",   "wav_x (Eg)"),
        ("wav_y",     1, -1, "violet",      "wav_y (Eg)"),
        ("pro",       2, -1, "yellow",      "pro   (A1u)"),
        ("dom_pure",  3, -1, "wheat",       "dom (A2u pure)"),
    ]
    # Absolute paths so the script works no matter where PyMOL is launched
    # from (PyMOL doesn't reliably cd into the .pml's directory).
    out_dir_abs = Path(out_dir).resolve()
    obj_files = {"planar": planar_filename, **mode_filenames}

    for obj_name, col, row, color, _label in layout:
        fname = obj_files[obj_name]
        full = out_dir_abs / fname
        pml.append(f"load {full}, {obj_name}")
        # Colour the carbons of this object with the mode tint; N/O/H/metal
        # follow CPK so the macrocycle structure stays legible.
        pml.append(f"color {color}, {obj_name} and elem C")
        pml.append(f"color blue, {obj_name} and elem N")
        pml.append(f"color white, {obj_name} and elem H")
        # Position in the grid
        dx = col * spacing_x
        dy = row * spacing_y
        pml.append(f"translate [{dx:.1f}, {dy:.1f}, 0], {obj_name}, camera=0")
        pml.append("")

    pml.append("# Render as sticks; metal as a sphere so it's visible")
    pml.append("hide everything")
    pml.append("show sticks")
    pml.append("show spheres, symbol Co+Fe+Mn+Ni+Rh+Cu+Zn")
    pml.append("set sphere_scale, 0.4, symbol Co+Fe+Mn+Ni+Rh+Cu+Zn")
    pml.append("color gray30, symbol Co+Fe+Mn+Ni+Rh+Cu+Zn")
    pml.append("")
    pml.append("# Text labels under each structure. The label text is anchored")
    pml.append("# at the pseudoatom and extends to the right; we offset half the")
    pml.append("# label width to the LEFT so the label visually centres under the")
    pml.append("# molecule. Font size kept small so labels don't overflow into")
    pml.append("# adjacent grid panels.")
    label_offset = 6.5 + max(0, amplitude - 0.5) * 1.5
    for obj_name, col, row, _color, label in layout:
        dx = col * spacing_x - len(label) * 0.20   # rough centring
        dy = row * spacing_y - label_offset
        pml.append(f'pseudoatom lbl_{obj_name}, pos=[{dx:.1f}, {dy:.1f}, 0], '
                   f'label="{label}"')
    pml.append("hide nonbonded, lbl_*")
    pml.append("set label_size, 14")
    pml.append("set label_color, black")
    pml.append("set label_font_id, 7")  # bold sans
    pml.append("")
    pml.append("# Camera tilt to reveal OOP distortion. Wrapped in a Python")
    pml.append("# block because PyMOL's command-mode 'turn' doesn't always")
    pml.append("# persist into subsequent ray/png in headless (-cq) mode;")
    pml.append("# cmd.turn() from Python does.")
    pml.append("python")
    pml.append("from pymol import cmd")
    pml.append("cmd.orient()")
    pml.append("cmd.turn('x', -55)")
    pml.append("cmd.zoom('all', 0)        # zoom AFTER turn so the tilted bbox fills the canvas")
    pml.append("python end")
    pml.append("")
    pml.append("# To render a high-res PNG headlessly:")
    pml.append("#   pymol -cq porphyrin_oop_modes.pml -d 'ray 2000, 1200; png out.png; quit'")

    out_path = Path(out_dir) / "porphyrin_oop_modes.pml"
    out_path.write_text("\n".join(pml) + "\n")
    return out_path


def _write_ip_pml(out_dir, mode_filenames, planar_filename, amplitude):
    """Write the PyMOL script for the IP modes — one planar reference plus
    six in-plane distortions arranged in a grid. Camera is set to top-down
    (looking along z) so xy displacements are clearly visible."""
    pml = []
    pml.append(f"# Sofia — porphyrin IP minimal-basis visualization")
    pml.append(f"# Generated by nsd/visualize_porphyrin.py")
    pml.append(f"# Each distortion is the analytic (Δx, Δy) pattern of one IP")
    pml.append(f"# mode at amplitude = {amplitude:.2f} Å, applied to a planar")
    pml.append(f"# metallo-porphine. The 'bre' (breathing) mode is shown as a")
    pml.append(f"# pure radial expansion (Sofia's analytic basis is ≈ 0 by")
    pml.append(f"# orbit-symmetric reference choice). The 'trn_x' / 'trn_y'")
    pml.append(f"# (Eu) modes are shown as uniform translations of the ring vs")
    pml.append(f"# the metal — the chemistry meaning of 'translation'.")
    pml.append(f"#")
    pml.append(f"# Open in PyMOL:  pymol porphyrin_ip_modes.pml")
    pml.append("")
    pml.append("reinitialize")
    pml.append("bg_color white")
    pml.append("set ray_opaque_background, on")   # solid white PNG (default has alpha)
    pml.append("set ray_shadows, 0")
    pml.append("set orthoscopic, on")
    pml.append("set stick_radius, 0.12")
    pml.append("")

    spacing_x = 14.0 + max(0, amplitude - 0.5) * 4.0
    spacing_y = 14.0 + max(0, amplitude - 0.5) * 4.0
    layout = [
        # (object_name, col, row, mode_color, label_text)
        ("planar",    0,  0, "gray70",     "planar reference"),
        ("bre",       1,  0, "paleyellow", "bre   (A1g)"),
        ("Nstr",      2,  0, "cyan",       "N-str (B1g)"),
        ("mstr",      3,  0, "magenta",    "m-str (B2g)"),
        ("rot",       1, -1, "orange",     "rot   (A2g)"),
        ("trn_x",     2, -1, "red",        "trn_x (Eu)"),
        ("trn_y",     3, -1, "green",      "trn_y (Eu)"),
    ]

    out_dir_abs = Path(out_dir).resolve()
    obj_files = {"planar": planar_filename, **mode_filenames}

    for obj_name, col, row, color, _label in layout:
        fname = obj_files[obj_name]
        full = out_dir_abs / fname
        pml.append(f"load {full}, {obj_name}")
        pml.append(f"color {color}, {obj_name} and elem C")
        pml.append(f"color blue, {obj_name} and elem N")
        pml.append(f"color white, {obj_name} and elem H")
        dx = col * spacing_x
        dy = row * spacing_y
        pml.append(f"translate [{dx:.1f}, {dy:.1f}, 0], {obj_name}, camera=0")
        pml.append("")

    pml.append("# Render as sticks; metal as a sphere so it's visible")
    pml.append("hide everything")
    pml.append("show sticks")
    pml.append("show spheres, symbol Co+Fe+Mn+Ni+Rh+Cu+Zn")
    pml.append("set sphere_scale, 0.4, symbol Co+Fe+Mn+Ni+Rh+Cu+Zn")
    pml.append("color gray30, symbol Co+Fe+Mn+Ni+Rh+Cu+Zn")
    pml.append("")
    pml.append("# Text labels under each structure (centred horizontally)")
    label_offset = 6.5 + max(0, amplitude - 0.5) * 1.5
    for obj_name, col, row, _color, label in layout:
        dx = col * spacing_x - len(label) * 0.20
        dy = row * spacing_y - label_offset
        pml.append(f'pseudoatom lbl_{obj_name}, pos=[{dx:.1f}, {dy:.1f}, 0], '
                   f'label="{label}"')
    pml.append("hide nonbonded, lbl_*")
    pml.append("set label_size, 14")
    pml.append("set label_color, black")
    pml.append("set label_font_id, 7")
    pml.append("")
    pml.append("# IP modes: top-down view (looking down the z-axis) so the")
    pml.append("# xy displacements are clearly visible. Wrapped in a Python")
    pml.append("# block (see OOP .pml note: cmd.* persists where command-mode")
    pml.append("# may not, in headless -cq mode).")
    pml.append("python")
    pml.append("from pymol import cmd")
    pml.append("cmd.orient()")
    pml.append("cmd.zoom('all', 0)")
    pml.append("python end")
    pml.append("")
    pml.append("# To render a high-res PNG headlessly:")
    pml.append("#   pymol -cq porphyrin_ip_modes.pml -d 'ray 2000, 1200; png out.png; quit'")

    out_path = Path(out_dir) / "porphyrin_ip_modes.pml"
    out_path.write_text("\n".join(pml) + "\n")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", metavar="FILE",
                    help="Reference Gaussian .log (default: bundled examples/porphyrin/Co1_Pro.log)")
    ap.add_argument("--amp", metavar="A", type=float, default=0.5,
                    help="Distortion amplitude in Å (default: 0.5)")
    ap.add_argument("--out", metavar="DIR", default="viz/porphyrin",
                    help="Output directory (default: viz/porphyrin)")
    args = ap.parse_args()

    # Locate the reference log
    if args.log:
        log_path = Path(args.log)
    else:
        # Repo root is two levels up from this file (nsd/visualize_porphyrin.py)
        repo_root = _SELF_DIR.parent
        log_path = repo_root / "examples" / "porphyrin" / "Co1_Pro.log"
    if not log_path.exists():
        sys.exit(f"reference log not found: {log_path}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reference geometry : {log_path}")
    print(f"Distortion amplitude: {args.amp:.2f} Å")
    print(f"Output directory   : {out_dir}")
    print()

    atoms, coords_planar, macro_idx, classes, parent_of_H, canonical_xy = \
        _make_planar_reference(log_path)
    metal_z = int(atoms[classes['metal'][0]])
    metal_sym = _ELEM.get(metal_z, f"Z{metal_z}")

    # Compute bonds once on the planar reference, then attach the same
    # CONECT records to every distorted variant. This guarantees PyMOL draws
    # the full macrocyclic skeleton at any amplitude (PDB CONECT is honoured
    # regardless of interatomic distance).
    bonds = _compute_bonds(atoms, coords_planar)

    # 1. Planar reference
    planar_pdb = "porphyrin_planar.pdb"
    _write_pdb(out_dir / planar_pdb, atoms, coords_planar, bonds,
               f"Planar {metal_sym}-porphine reference (Sofia)")
    print(f"  wrote {planar_pdb}")

    # 2. One distorted variant per OOP mode
    print()
    print("Out-of-plane modes:")
    mode_filenames = {}
    for name, irrep, _color, descr in _MODES:
        coords_d = _apply_oop_mode(coords_planar, macro_idx, parent_of_H,
                                   canonical_xy, name, args.amp,
                                   metal_idx=classes['metal'][0])
        fname = f"porphyrin_{name}.pdb"
        _write_pdb(out_dir / fname, atoms, coords_d, bonds,
                   f"{name} ({irrep}) -- {descr}; amplitude {args.amp:.2f} A")
        mode_filenames[name] = fname
        print(f"  wrote {fname:<24s} ({irrep}) — {descr}")

    # 3. One distorted variant per IP mode
    print()
    print("In-plane modes:")
    ip_mode_filenames = {}
    for name, irrep, _color, descr in _IP_MODES:
        coords_d = _apply_ip_mode(coords_planar, macro_idx, parent_of_H,
                                  canonical_xy, name, args.amp,
                                  metal_idx=classes['metal'][0])
        fname = f"porphyrin_ip_{name}.pdb"
        _write_pdb(out_dir / fname, atoms, coords_d, bonds,
                   f"{name} ({irrep}) -- {descr}; amplitude {args.amp:.2f} A")
        ip_mode_filenames[name] = fname
        print(f"  wrote {fname:<24s} ({irrep}) — {descr}")

    # 4. PyMOL scripts (one for OOP, one for IP)
    pml_path = _write_pml(out_dir, mode_filenames, planar_pdb, args.amp)
    ip_pml_path = _write_ip_pml(out_dir, ip_mode_filenames, planar_pdb, args.amp)
    print()
    print(f"Open in PyMOL:")
    print(f"  pymol {pml_path}        # OOP modes (Δz)")
    print(f"  pymol {ip_pml_path}     # IP modes  (Δx, Δy)")


if __name__ == "__main__":
    main()
