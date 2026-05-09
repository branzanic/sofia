#!/usr/bin/env python3
"""Shared helpers for the porphyrin and corrin visualization tools.

The two macrocycle classes use identical analytic minimal-basis modes
(Sofia's `build_oop_basis` / `build_ip_basis` adapt to any atom count), so
most of the geometry-application + file-output logic is shared. This module
holds it; `visualize_porphyrin.py` and `visualize_corrin.py` are thin entry
points that supply the macrocycle-specific parameters (topology identifier,
class layout, default reference log, file prefix, label strings).
"""

import sys
from pathlib import Path

import numpy as np

# Allow importing nsd_porphyrin (sibling module) when running as a script
_SELF_DIR = Path(__file__).resolve().parent
if str(_SELF_DIR) not in sys.path:
    sys.path.insert(0, str(_SELF_DIR))

import nsd_porphyrin as porph


_ELEM = {1: "H", 6: "C", 7: "N", 8: "O", 16: "S",
         23: "V", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co",
         28: "Ni", 29: "Cu", 30: "Zn", 45: "Rh"}


# -------------------- planar reference --------------------

def make_planar_reference(log_path, identifier_fn):
    """Read a Gaussian log, extract the inner ring + bonded H atoms, force
    z=0 on the inner ring (and shift each bonded H by the same Δz so C–H
    bonds stay rigid). Returns:
        (atoms, coords_planar, macro_idx, classes, parent_of_H, canonical_xy)

    `identifier_fn(atoms, coords)` must return either:
        (classes, macro_idx)                — porphyrin convention
        (classes, macro_idx, direct_pair)   — corrin convention
    Only the first two are used; extras are ignored.
    """
    atoms, coords = porph.parse_last_standard_orientation(str(log_path))
    result = identifier_fn(atoms, coords)
    classes, macro_idx = result[0], result[1]

    # Mean-plane fit on the inner ring; rotate so plane normal = +z
    centroid, normal = porph.fit_mean_plane(coords[macro_idx])
    R = porph.rotation_to_z(normal)
    coords = (coords - centroid) @ R.T

    # Canonicalise xy: align N atoms to ±x, ±y axes
    canonical_xy = porph.canonicalise_xy(coords[macro_idx], np.arange(4))
    src = coords[macro_idx][:, :2]
    tgt = canonical_xy[:, :2]
    H_mat = src.T @ tgt
    U, _, Vt = np.linalg.svd(H_mat)
    R2 = Vt.T @ U.T
    if np.linalg.det(R2) < 0:
        Vt[-1] *= -1
        R2 = Vt.T @ U.T
    coords[:, :2] = coords[:, :2] @ R2.T

    # Build adjacency to find which H is bonded to each inner-ring C atom
    adj = porph.build_adjacency(atoms, coords)
    parent_of_H = {}
    macro_set = set(macro_idx)
    for i in range(len(atoms)):
        if int(atoms[i]) == 1:
            for j in adj[i]:
                if j in macro_set:
                    parent_of_H[i] = j
                    break

    # Force inner-ring z to 0; shift bonded H by the same residual.
    coords_planar = coords.copy()
    macro_z_residual = {int(mi): float(coords[mi, 2]) for mi in macro_idx}
    for mi in macro_idx:
        coords_planar[mi, 2] = 0.0
        for h, parent in parent_of_H.items():
            if parent == mi:
                coords_planar[h, 2] -= macro_z_residual[int(mi)]

    # Compute substituent BFS map: each non-macro, non-bonded-H atom (e.g.
    # peripheral methyl C, propionamide chain, F-ring atoms in F430) is mapped
    # to its nearest macro_idx ancestors. Required for corphin/F430 viz so
    # peripherals stay rigid against the inner ring under distortion. For
    # bare metallo-porphine and bare metallo-corrin (no peripheral
    # substituents), this map is empty — no behavior change.
    substituent_parents = compute_substituent_parents(atoms, coords, macro_idx,
                                                      adj=adj, exclude=set(parent_of_H))

    # Planarise peripheral atoms: shift each by the avg Δz_residual of its
    # nearest macro_idx ancestor(s). Keeps substituent bond geometry rigid.
    for atom_idx, parents in substituent_parents.items():
        avg_dz_residual = sum(macro_z_residual[p] for p in parents) / len(parents)
        coords_planar[atom_idx, 2] -= avg_dz_residual

    return (atoms, coords_planar, macro_idx, classes, parent_of_H,
            canonical_xy, substituent_parents)


def compute_substituent_parents(atoms, coords, macro_idx, adj=None, exclude=None):
    """BFS each non-inner-ring atom to find its nearest inner-ring ancestor(s).

    Returns a dict {atom_idx: [list of nearest macro_idx]} for every non-macro
    atom reachable from the inner ring through heavy-atom + H bonds, EXCEPT
    those listed in `exclude` (typically `parent_of_H` keys, which are
    handled separately by apply_*_mode using the parent_of_H map).

    Atoms with multiple equidistant macro_idx ancestors (e.g. F-ring carbons
    bridging two adjacent Cα) get all of them; their displacement is the
    average over those ancestors.
    """
    if adj is None:
        adj = porph.build_adjacency(atoms, coords)
    if exclude is None:
        exclude = set()
    macro_set = set(int(i) for i in macro_idx)

    # BFS layer-by-layer from inner-ring atoms; don't propagate THROUGH
    # other inner-ring atoms (each macro atom is its own BFS root).
    visited = {}  # atom_idx -> (distance, set of macro_idx ancestors)
    for mi in macro_idx:
        visited[int(mi)] = (0, {int(mi)})
    frontier = [int(mi) for mi in macro_idx]
    while frontier:
        next_frontier = []
        for u in frontier:
            d_u, ancestors_u = visited[u]
            for v in adj[u]:
                v = int(v)
                if v in macro_set:
                    continue
                if v not in visited:
                    visited[v] = (d_u + 1, set(ancestors_u))
                    next_frontier.append(v)
                else:
                    d_v, ancestors_v = visited[v]
                    if d_u + 1 == d_v:
                        ancestors_v |= ancestors_u
        frontier = next_frontier

    return {a: sorted(anc) for a, (d, anc) in visited.items()
            if a not in macro_set and a not in exclude}


# -------------------- mode application --------------------

def _propagate_dz_to_substituents(coords_out, dz_pattern, macro_idx,
                                  substituent_parents):
    """Apply Δz to each peripheral atom = avg of its macro_idx ancestors' Δz."""
    if not substituent_parents:
        return
    macro_pos = {int(mi): k for k, mi in enumerate(macro_idx)}
    for atom_idx, parents in substituent_parents.items():
        avg_dz = sum(float(dz_pattern[macro_pos[p]]) for p in parents) / len(parents)
        coords_out[atom_idx, 2] += avg_dz


def _propagate_dxy_to_substituents(coords_out, dxy_pattern, macro_idx,
                                   substituent_parents):
    """Apply (Δx, Δy) to each peripheral atom = avg of its macro_idx ancestors'
    (Δx, Δy). dxy_pattern is shape (N_macro, 2)."""
    if not substituent_parents:
        return
    macro_pos = {int(mi): k for k, mi in enumerate(macro_idx)}
    for atom_idx, parents in substituent_parents.items():
        avg_dx = sum(float(dxy_pattern[macro_pos[p], 0]) for p in parents) / len(parents)
        avg_dy = sum(float(dxy_pattern[macro_pos[p], 1]) for p in parents) / len(parents)
        coords_out[atom_idx, 0] += avg_dx
        coords_out[atom_idx, 1] += avg_dy


def apply_oop_mode(coords_planar, macro_idx, parent_of_H, canonical_xy,
                   mode_name, amplitude, metal_idx=None,
                   substituent_parents=None):
    """Add Δz from one OOP minimal-basis mode to the inner ring + bonded H,
    and (optionally) to the peripheral substituent atoms in
    `substituent_parents` (averaging their parents' Δz). For bare porphine /
    bare corrin pass substituent_parents=None — no peripherals to propagate.

    Mode names:
      sad / ruf / dom_pure / wav_x / wav_y / pro    — Sofia's analytic basis as-is
      dom                                           — chemistry-textbook deoxy-Hb
                                                       hybrid (A2u + Tz + metal-out-of-plane)
                                                       for visualisation only
    """
    coords_out = coords_planar.copy()

    if mode_name == "dom":
        # Chemistry-textbook "deoxy-Hb dome": all atoms above the plane,
        # Δz ∝ (r²_max − r²) so N atoms (small r²) move most, Cβ stay near
        # the rim. Plus a small extra lift on the metal so it sits above
        # the dome apex. NOT a pure A2u eigenvector — see dom_pure for that.
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
        _propagate_dz_to_substituents(coords_out, dz_pattern, macro_idx,
                                      substituent_parents)

        if metal_idx is not None:
            # N atoms are the first 4 entries in canonical class order
            n_indices = macro_idx[:4]
            n_max_z = max(coords_out[ni, 2] for ni in n_indices)
            coords_out[metal_idx, 2] = n_max_z + 0.25 * amplitude
        return coords_out

    # All other modes (incl. dom_pure): use Sofia's analytic basis directly
    basis_key = "dom" if mode_name == "dom_pure" else mode_name
    basis = porph.build_oop_basis(canonical_xy)
    if basis_key not in basis:
        raise KeyError(f"unknown OOP mode {mode_name!r}; "
                       f"choose from {list(basis.keys()) + ['dom_pure']}")
    dz_pattern = basis[basis_key] * amplitude
    for k_macro, mi in enumerate(macro_idx):
        dz = float(dz_pattern[k_macro])
        coords_out[mi, 2] += dz
        for h, parent in parent_of_H.items():
            if parent == mi:
                coords_out[h, 2] += dz
    _propagate_dz_to_substituents(coords_out, dz_pattern, macro_idx,
                                  substituent_parents)
    return coords_out


def apply_ip_mode(coords_planar, macro_idx, parent_of_H, canonical_xy,
                  mode_name, amplitude, metal_idx=None,
                  substituent_parents=None):
    """Add (Δx, Δy) from one IP minimal-basis mode to the inner ring + H,
    and (optionally) to peripheral substituent atoms in `substituent_parents`.

    For 'bre' the analytic A1g amplitude is ≈ 0 by orbit-symmetric reference
    choice, so we render uniform radial outward (chemistry breathing). For
    'trn_x' / 'trn_y' (Eu) the analytic basis is an r²-weighted wobble; we
    render uniform translation of the ring vs the metal — what 'translation'
    means in chemistry parlance. For N-str / m-str / rot we use Sofia's
    analytic basis directly.
    """
    coords_out = coords_planar.copy()

    if mode_name == "bre":
        # Build a (N_macro, 2) Δxy pattern so peripherals can inherit it
        dxy_pattern = np.zeros((len(macro_idx), 2))
        for k_macro, mi in enumerate(macro_idx):
            x, y, _ = coords_planar[mi]
            r = (x * x + y * y) ** 0.5
            if r < 1e-9:
                continue
            ux, uy = x / r, y / r
            dx, dy = ux * amplitude, uy * amplitude
            dxy_pattern[k_macro, 0] = dx
            dxy_pattern[k_macro, 1] = dy
            coords_out[mi, 0] += dx
            coords_out[mi, 1] += dy
            for h, parent in parent_of_H.items():
                if parent == mi:
                    coords_out[h, 0] += dx
                    coords_out[h, 1] += dy
        _propagate_dxy_to_substituents(coords_out, dxy_pattern, macro_idx,
                                       substituent_parents)
        return coords_out

    if mode_name in ("trn_x", "trn_y"):
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
        # Uniform translation: peripherals shift rigidly with the ring
        if substituent_parents:
            for atom_idx in substituent_parents:
                coords_out[atom_idx, 0] += dx
                coords_out[atom_idx, 1] += dy
        return coords_out

    basis = porph.build_ip_basis(canonical_xy)
    if mode_name not in basis:
        raise KeyError(f"unknown IP mode {mode_name!r}; "
                       f"choose from {list(basis.keys()) + ['bre']}")
    flat = basis[mode_name] * amplitude
    delta_xy = flat.reshape(-1, 2)
    for k_macro, mi in enumerate(macro_idx):
        dx, dy = float(delta_xy[k_macro, 0]), float(delta_xy[k_macro, 1])
        coords_out[mi, 0] += dx
        coords_out[mi, 1] += dy
        for h, parent in parent_of_H.items():
            if parent == mi:
                coords_out[h, 0] += dx
                coords_out[h, 1] += dy
    _propagate_dxy_to_substituents(coords_out, delta_xy, macro_idx,
                                   substituent_parents)
    return coords_out


# -------------------- file output --------------------

def write_pdb(path, atoms, coords, bonds, comment):
    """Write a PDB file with explicit CONECT records.

    Bonds are computed once on the planar reference and re-used for every
    distorted variant — PyMOL respects CONECT regardless of distance, so the
    macrocyclic skeleton stays drawn at any amplitude.
    """
    with open(path, "w") as f:
        f.write(f"REMARK   1 {comment}\n")
        for k, (z, (x, y, zc)) in enumerate(zip(atoms, coords), start=1):
            sym = _ELEM.get(int(z), f"Z{int(z)}")
            name = f"{sym:<2}{k:>2}"[:4] if len(sym) <= 2 else sym[:4]
            f.write(
                f"HETATM{k:>5d} {name:<4s} MOL A   1    "
                f"{x:8.3f}{y:8.3f}{zc:8.3f}  1.00  0.00          {sym:>2s}\n"
            )
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


def compute_bonds(atoms, coords):
    """Return list of (i, j) bond pairs from a planar geometry, using
    Sofia's covalent_threshold rules."""
    bonds = []
    n = len(atoms)
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(coords[i] - coords[j]))
            if d < porph.covalent_threshold(int(atoms[i]), int(atoms[j])):
                bonds.append((i, j))
    return bonds


# -------------------- PyMOL script writers --------------------

def _pml_header(amplitude, basis_label, header_extra=""):
    pml = []
    pml.append(f"# Sofia — {basis_label} minimal-basis visualization")
    pml.append(f"# Distortion amplitude = {amplitude:.2f} Å")
    if header_extra:
        for line in header_extra.split("\n"):
            pml.append(f"# {line}")
    pml.append("")
    pml.append("reinitialize")
    pml.append("bg_color white")
    pml.append("set ray_opaque_background, on   # solid white PNG (default has alpha)")
    pml.append("set ray_shadows, 0")
    pml.append("set orthoscopic, on")
    pml.append("set stick_radius, 0.12")
    pml.append("")
    return pml


def write_oop_pml(out_dir, mode_filenames, planar_filename, amplitude,
                  prefix, layout, label_macrocycle, header_extra=""):
    """Write a PyMOL script that loads all OOP variants in a 4×2 grid.

    Args:
      out_dir         — directory containing the .pdb files
      mode_filenames  — dict {mode_name: filename}
      planar_filename — filename of the planar reference .pdb
      amplitude       — distortion amplitude (Å), used for grid spacing
      prefix          — output filename prefix, e.g. "porphyrin" or "corrin"
      layout          — list of (object_name, col, row, color, label_text)
      label_macrocycle — short macrocycle name for the .pml header
      header_extra    — optional multi-line text added as comments at the top
    """
    pml = _pml_header(amplitude, f"{label_macrocycle} OOP", header_extra)

    spacing_x = 14.0 + max(0, amplitude - 0.5) * 4.0
    spacing_y = 14.0 + max(0, amplitude - 0.5) * 4.0

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
    pml.append("# Tilt the camera to make OOP distortion visible. Wrapped in a")
    pml.append("# Python block — PyMOL's command-mode 'turn' doesn't always")
    pml.append("# persist into headless ray; cmd.turn() does.")
    pml.append("python")
    pml.append("from pymol import cmd")
    pml.append("cmd.orient()")
    pml.append("cmd.turn('x', -55)")
    pml.append("cmd.zoom('all', 0)")
    pml.append("python end")
    pml.append("")
    pml.append(f"# Headless render:")
    pml.append(f"#   pymol -cq {prefix}_oop_modes.pml -d 'ray 2000, 1200; png out.png; quit'")

    out_path = Path(out_dir) / f"{prefix}_oop_modes.pml"
    out_path.write_text("\n".join(pml) + "\n")
    return out_path


def write_ip_pml(out_dir, mode_filenames, planar_filename, amplitude,
                 prefix, layout, label_macrocycle, header_extra=""):
    """Write the IP PyMOL script — top-down view, no tilt."""
    pml = _pml_header(amplitude, f"{label_macrocycle} IP", header_extra)

    spacing_x = 14.0 + max(0, amplitude - 0.5) * 4.0
    spacing_y = 14.0 + max(0, amplitude - 0.5) * 4.0

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
    pml.append("# Top-down (no tilt) so xy displacements are clearly visible")
    pml.append("python")
    pml.append("from pymol import cmd")
    pml.append("cmd.orient()")
    pml.append("cmd.zoom('all', 0)")
    pml.append("python end")
    pml.append("")
    pml.append(f"# Headless render:")
    pml.append(f"#   pymol -cq {prefix}_ip_modes.pml -d 'ray 2000, 1200; png out.png; quit'")

    out_path = Path(out_dir) / f"{prefix}_ip_modes.pml"
    out_path.write_text("\n".join(pml) + "\n")
    return out_path
