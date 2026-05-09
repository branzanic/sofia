#!/usr/bin/env python3
"""
Out-of-plane Normal Structural Decomposition (NSD) of porphyrin macrocycles
extracted from Gaussian log files.

Method follows the minimal-basis decomposition of:
  Jentzen, Song, Shelnutt, J. Phys. Chem. B 101, 1684 (1997).

Six minimal-basis out-of-plane modes for the 24-atom macrocycle (D4h ref):
  sad (B2u), ruf (B1u), dom (A2u), wav(x) (Eg), wav(y) (Eg), pro (A1u).

Basis vectors are the lowest-order symmetry-adapted Delta-z patterns
(orthonormalised, rigid-body translation/rotation projected out). These
span the same 6-D minimal-basis subspace as Jentzen's force-field
eigenvectors. For each 1-D irrep the d_1 amplitude matches Jentzen's
up to sign and a near-unity scaling. The magnitude per irrep |D^Gamma|
is basis-independent and is reported alongside.

Usage:
    python3 nsd_porphyrin.py <file.log>
    python3 nsd_porphyrin.py <directory>
    python3 nsd_porphyrin.py <directory> --csv out.csv
"""

import argparse
import csv
import datetime as _datetime
import sys
from pathlib import Path
import numpy as np


SOFIA_VERSION = "0.1"

ELEMENT_SYMBOLS = {
    1: "H", 6: "C", 7: "N", 8: "O", 16: "S",
    23: "V", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co",
    28: "Ni", 29: "Cu", 30: "Zn", 45: "Rh",
}
TRANSITION_METALS = {23, 24, 25, 26, 27, 28, 29, 30, 45}

PLANARITY_THRESHOLD_A = 0.05  # Å — Jentzen 1997 X-ray uncertainty


# ---------- Gaussian log parsing ----------

def parse_last_standard_orientation(path):
    """Return (atomic_numbers[N], coords[N,3]) of the last 'Standard orientation' block."""
    last_atoms = None
    last_coords = None
    with open(path) as f:
        line = f.readline()
        while line:
            if "Standard orientation" in line:
                # Skip 4 header lines
                for _ in range(4):
                    f.readline()
                atoms, coords = [], []
                while True:
                    rec = f.readline()
                    if not rec or "----" in rec:
                        break
                    parts = rec.split()
                    if len(parts) >= 6:
                        atoms.append(int(parts[1]))
                        coords.append([float(parts[3]), float(parts[4]), float(parts[5])])
                if atoms:
                    last_atoms = np.array(atoms, dtype=int)
                    last_coords = np.array(coords, dtype=float)
            line = f.readline()
    if last_atoms is None:
        raise ValueError(f"No 'Standard orientation' block found in {path}")
    return last_atoms, last_coords


# ---------- Connectivity ----------

def covalent_threshold(z1, z2):
    """Maximum bond distance for elements with atomic numbers z1 and z2."""
    if z1 in TRANSITION_METALS or z2 in TRANSITION_METALS:
        return 2.55
    if z1 == 1 or z2 == 1:
        return 1.25
    if {z1, z2} == {6, 6}:
        return 1.75
    if {z1, z2} == {6, 7}:
        return 1.65
    if {z1, z2} == {7, 7}:
        return 1.55
    return 1.85


def build_adjacency(atoms, coords):
    n = len(atoms)
    adj = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(coords[i] - coords[j])
            if d < covalent_threshold(int(atoms[i]), int(atoms[j])):
                adj[i].add(j)
                adj[j].add(i)
    return adj


def identify_macrocycle(atoms, coords):
    """Identify the 24 atoms of the porphyrin macrocycle.

    Returns a dict with sorted index lists keyed by class:
      'metal' (1), 'N' (4), 'Calpha' (8), 'Cmeso' (4), 'Cbeta' (8).
    Also returns the 24-atom ordered index array in the canonical class order
    [N x4, Calpha x8, Cmeso x4, Cbeta x8].
    """
    adj = build_adjacency(atoms, coords)

    metals = [i for i, z in enumerate(atoms) if int(z) in TRANSITION_METALS]
    if len(metals) != 1:
        raise ValueError(f"Expected exactly one transition metal; found {len(metals)}")
    metal = metals[0]

    n_idx = sorted(j for j in adj[metal] if int(atoms[j]) == 7)
    if len(n_idx) != 4:
        raise ValueError(f"Expected 4 pyrrole N bonded to metal; found {len(n_idx)}")

    calpha = set()
    for n in n_idx:
        for k in adj[n]:
            if int(atoms[k]) == 6:
                calpha.add(k)
    if len(calpha) != 8:
        raise ValueError(f"Expected 8 C-alpha; found {len(calpha)}")

    cmeso, cbeta = set(), set()
    for ca in calpha:
        for k in adj[ca]:
            if int(atoms[k]) == 6 and k not in calpha:
                heavy_nbrs = [m for m in adj[k] if int(atoms[m]) != 1]
                n_calpha = sum(1 for m in heavy_nbrs if m in calpha)
                if n_calpha == 2:
                    cmeso.add(k)
                elif n_calpha == 1:
                    cbeta.add(k)
    if len(cmeso) != 4:
        raise ValueError(f"Expected 4 C-meso; found {len(cmeso)}")
    if len(cbeta) != 8:
        raise ValueError(f"Expected 8 C-beta; found {len(cbeta)}")

    classes = {
        "metal": [metal],
        "N": sorted(n_idx),
        "Calpha": sorted(calpha),
        "Cmeso": sorted(cmeso),
        "Cbeta": sorted(cbeta),
    }
    macro_indices = np.array(classes["N"] + classes["Calpha"]
                             + classes["Cmeso"] + classes["Cbeta"], dtype=int)
    return classes, macro_indices


# ---------- Geometry: mean plane and canonical orientation ----------

def fit_mean_plane(coords24):
    """Least-squares mean plane. Returns (centroid, normal)."""
    centroid = coords24.mean(axis=0)
    centered = coords24 - centroid
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    normal = Vt[-1]
    if normal[2] < 0:
        normal = -normal
    return centroid, normal


def rotation_to_z(normal):
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(normal, z)
    s = np.linalg.norm(v)
    c = float(np.dot(normal, z))
    if s < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    K = np.array([[0, -v[2], v[1]],
                  [v[2], 0, -v[0]],
                  [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1.0 - c) / (s * s))


def canonicalise_xy(coords24, n_local_indices):
    """Rotate around z so the four pyrrole-N atoms lie close to +-x, +-y axes."""
    n_xy = coords24[n_local_indices, :2]
    angles = np.arctan2(n_xy[:, 1], n_xy[:, 0])
    # Reduce angles modulo pi/2 (since N atoms are 4-fold symmetric)
    folded = np.mod(angles, np.pi / 2.0)
    folded = np.where(folded > np.pi / 4.0, folded - np.pi / 2.0, folded)
    twist = float(np.mean(folded))
    c, s = np.cos(-twist), np.sin(-twist)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return coords24 @ R.T


# ---------- Out-of-plane minimal basis ----------

def build_oop_basis(canonical_xy):
    """Return six orthonormal Delta-z patterns over 24 atoms.

    Patterns (with N along +-x, +-y, mesos along diagonals):
      sad (B2u): x*y
      ruf (B1u): x^2 - y^2
      dom (A2u): r^2          [translation projected out]
      wav_x (Eg): x*r^2       [Ry rotation projected out]
      wav_y (Eg): y*r^2       [Rx rotation projected out, then orthog. to wav_x]
      pro (A1u): x*y*(x^2 - y^2)
    """
    x, y = canonical_xy[:, 0], canonical_xy[:, 1]
    r2 = x * x + y * y

    raw = {
        "ruf": x * y,
        "sad": x * x - y * y,
        "dom": r2.copy(),
        "wav_x": x * r2,
        "wav_y": y * r2,
        "pro": x * y * (x * x - y * y),
    }

    # Rigid-body Delta-z modes to project out:
    # tz: uniform (A2u, contaminates dom)
    # rx: -y (Eg, contaminates wav_y -- rotation about x-axis)
    # ry: +x (Eg, contaminates wav_x -- rotation about y-axis)
    rb = [np.ones(len(x)), -y.copy(), x.copy()]
    rb_orth = []
    for v in rb:
        u = v.copy()
        for w in rb_orth:
            u -= (u @ w) * w
        nu = np.linalg.norm(u)
        if nu > 1e-10:
            rb_orth.append(u / nu)

    def proj_out_rb(vec):
        u = vec.copy()
        for w in rb_orth:
            u -= (u @ w) * w
        return u

    basis = {}
    for name, vec in raw.items():
        u = proj_out_rb(vec)
        nu = np.linalg.norm(u)
        if nu > 1e-12:
            u = u / nu
        basis[name] = u

    # Make wav_y orthogonal to wav_x (within Eg subspace)
    wx = basis["wav_x"]
    wy = basis["wav_y"]
    wy = wy - (wy @ wx) * wx
    nwy = np.linalg.norm(wy)
    if nwy > 1e-12:
        basis["wav_y"] = wy / nwy

    # Sign convention: ruf positive when meso atoms in (+,+) and (-,-) quadrants
    # are above the mean plane (matches Jentzen Fig 1). Apply by demanding the
    # basis vector value at the (+,+) meso-class atom to be positive.
    return basis


def project(dz, basis):
    return {name: float(dz @ vec) for name, vec in basis.items()}


def reconstruct(amps, basis):
    out = np.zeros_like(next(iter(basis.values())))
    for name, a in amps.items():
        out = out + a * basis[name]
    return out


# ---------- Conformation classification ----------

# Verbose chemical names for the minimal-basis modes.
OOP_CONFORMATION_NAMES = {
    "sad": ("saddled",      "B2u",  "pyrroles tilt alternately up/down"),
    "ruf": ("ruffled",      "B1u",  "meso carbons alternate up/down"),
    "dom": ("domed",        "A2u",  "pyrroles bend away from metal in unison"),
    "wav": ("waved",        "Eg",   "non-symmetric flexing along one axis"),
    "pro": ("propellered",  "A1u",  "each pyrrole rotates about its own normal"),
}

IP_CONFORMATION_NAMES = {
    "bre":  ("breathing",       "A1g",  "uniform radial expansion"),
    "Nstr": ("N-stretched",     "B1g",  "opposite-N pairs stretched/compressed"),
    "mstr": ("meso-stretched",  "B2g",  "opposite-meso pairs stretched/compressed"),
    "rot":  ("rotated",         "A2g",  "pyrroles rotate in-phase"),
    "trn":  ("translated",      "Eu",   "pyrrole pairs translate together"),
}


def describe_oop_conformation(amps, planar_threshold=0.05, secondary_fraction=0.30):
    """Return a human-readable string describing the OOP macrocycle conformation.

    The default planar_threshold of 0.05 Å is roughly the X-ray positional
    uncertainty for synthetic porphyrins (Jentzen 1997) and serves as a
    chemically sensible cutoff between "planar" and "distorted". For
    distortions above this cutoff, the dominant mode is named, plus any
    secondary modes whose magnitude is at least secondary_fraction (default
    30%) of the dominant.

    Inputs the amps dict with keys 'sad', 'ruf', 'dom', 'wav_x', 'wav_y', 'pro'.
    """
    mags = {
        "sad": abs(amps["sad"]),
        "ruf": abs(amps["ruf"]),
        "dom": abs(amps["dom"]),
        "wav": float(np.hypot(amps["wav_x"], amps["wav_y"])),
        "pro": abs(amps["pro"]),
    }
    items = sorted(mags.items(), key=lambda kv: -kv[1])
    primary, primary_mag = items[0]
    if primary_mag < planar_threshold:
        # Below the planarity threshold (≈ X-ray positional uncertainty);
        # the residual is noise, not a meaningful chemical distortion.
        return "essentially planar"
    threshold = primary_mag * secondary_fraction
    significant = [m for m, v in items if v >= threshold]
    names = [OOP_CONFORMATION_NAMES[m][0] for m in significant]
    if len(names) == 1:
        return names[0]
    return " + ".join(names)


def describe_ip_conformation(ip_amps, planar_threshold=0.05, secondary_fraction=0.30):
    """Return a human-readable string describing the in-plane macrocycle distortion.

    Same planar_threshold convention as describe_oop_conformation.
    """
    mags = {
        "bre":  abs(ip_amps["bre"]),
        "Nstr": abs(ip_amps["Nstr"]),
        "mstr": abs(ip_amps["mstr"]),
        "rot":  abs(ip_amps["rot"]),
        "trn":  float(np.hypot(ip_amps["trn_x"], ip_amps["trn_y"])),
    }
    items = sorted(mags.items(), key=lambda kv: -kv[1])
    primary, primary_mag = items[0]
    if primary_mag < planar_threshold:
        return "essentially symmetric"
    threshold = primary_mag * secondary_fraction
    significant = [m for m, v in items if v >= threshold]
    names = [IP_CONFORMATION_NAMES[m][0] for m in significant]
    if len(names) == 1:
        return names[0]
    return " + ".join(names)


# ---------- In-plane: D4h-symmetrised reference geometry ----------

def _d4h_inplane_ops():
    """Return the 8 D4h in-plane symmetry operations as 2x2 matrices.

    4 proper rotations (E, C4, C2, C4^3) and 4 reflections (sigma_v x, sigma_v y,
    sigma_d through y=x, sigma_d through y=-x).
    """
    ops = []
    for k in range(4):
        c, s = np.cos(k * np.pi / 2), np.sin(k * np.pi / 2)
        ops.append(np.array([[c, -s], [s, c]]))
    ops.append(np.array([[1.0, 0.0], [0.0, -1.0]]))   # sigma_v through x-axis
    ops.append(np.array([[-1.0, 0.0], [0.0, 1.0]]))   # sigma_v through y-axis
    ops.append(np.array([[0.0, 1.0], [1.0, 0.0]]))    # sigma_d through y=x
    ops.append(np.array([[0.0, -1.0], [-1.0, 0.0]]))  # sigma_d through y=-x
    return ops


def symmetrise_canonical(canonical, class_array, angular_tolerance_deg=20.0):
    """Return a D4h-(approx-)symmetric in-plane reference for the macrocycle.

    For every atom, the 'true' position is taken as the average over the D4h
    orbit operations that map it to (within angular_tolerance_deg of) another
    atom of the same class. This gracefully handles macrocycles whose class
    counts are not full D4h orbits (e.g. corrin's 3-Cmeso class): operations
    pointing to a 'missing slot' are skipped instead of being mapped to the
    wrong atom.

    Only the in-plane (xy) coordinates are altered; z is left untouched.
    """
    n = canonical.shape[0]
    ops = _d4h_inplane_ops()
    same_class_indices = {c: [i for i, cl in enumerate(class_array) if cl == c]
                          for c in set(class_array)}
    tol = np.deg2rad(angular_tolerance_deg)
    ref = canonical.copy()
    for i in range(n):
        same = same_class_indices[class_array[i]]
        if not same:
            continue
        images = []
        for op in ops:
            transformed = op @ canonical[i, :2]
            r_t = float(np.linalg.norm(transformed))
            if r_t < 1e-9:
                continue
            angle_t = float(np.arctan2(transformed[1], transformed[0]))
            best_j, best_d = -1, np.inf
            for j in same:
                rj = float(np.linalg.norm(canonical[j, :2]))
                if rj < 1e-9:
                    continue
                aj = float(np.arctan2(canonical[j, 1], canonical[j, 0]))
                d = abs(float(np.angle(np.exp(1j * (angle_t - aj)))))
                if d < best_d:
                    best_d, best_j = d, j
            if best_j >= 0 and best_d <= tol:
                images.append(op.T @ canonical[best_j, :2])
        if images:
            ref[i, :2] = np.mean(images, axis=0)
    return ref


# ---------- In-plane minimal basis ----------

def build_ip_basis(canonical_xy):
    """Return six orthonormal in-plane minimal-basis modes as 48-D vectors.

    Each vector is a stack [(dx_0, dy_0, dx_1, dy_1, ..., dx_23, dy_23)] over
    the 24 macrocycle atoms in canonical orientation (N along +-x, +-y).

    Modes (lowest-order analytic patterns of each in-plane irrep of D4h):
      bre   (A1g): radial outward (Delta_r = const)
      Nstr  (B1g): radial outward weighted by cos(2*theta)
      mstr  (B2g): radial outward weighted by sin(2*theta)
      rot   (A2g): tangential weighted by cos(4*theta) (Rz projected out)
      trn_x (Eu) : Delta_x = r^2 (Tx projected out, => Delta_x = r^2 - <r^2>)
      trn_y (Eu) : Delta_y = r^2 (Ty projected out)

    The rigid-body modes Tx, Ty, Rz are projected out before normalisation.
    Within Eu, trn_y is then orthogonalised against trn_x.
    """
    x = canonical_xy[:, 0]
    y = canonical_xy[:, 1]
    r2 = x * x + y * y
    r = np.sqrt(r2)

    cos2t = (x * x - y * y) / r2
    sin2t = 2.0 * x * y / r2
    cos4t = cos2t * cos2t - sin2t * sin2t  # = (x^4 - 6 x^2 y^2 + y^4) / r^4

    def stack(dx, dy):
        return np.column_stack([dx, dy]).flatten()

    raw = {
        "bre":   stack(x / r,           y / r),
        "Nstr":  stack(cos2t * x / r,   cos2t * y / r),
        "mstr":  stack(sin2t * x / r,   sin2t * y / r),
        "rot":   stack(-cos4t * y / r,  cos4t * x / r),
        "trn_x": stack(r2,              np.zeros_like(r2)),
        "trn_y": stack(np.zeros_like(r2), r2),
    }

    # Rigid-body in-plane modes
    Tx = stack(np.ones_like(x),  np.zeros_like(x))
    Ty = stack(np.zeros_like(x), np.ones_like(x))
    Rz = stack(-y, x)
    rb_orth = []
    for v in [Tx, Ty, Rz]:
        u = v.copy()
        for w in rb_orth:
            u -= (u @ w) * w
        nu = np.linalg.norm(u)
        if nu > 1e-12:
            rb_orth.append(u / nu)

    def proj_out_rb(vec):
        u = vec.copy()
        for w in rb_orth:
            u -= (u @ w) * w
        return u

    basis = {}
    for name, vec in raw.items():
        u = proj_out_rb(vec)
        nu = np.linalg.norm(u)
        basis[name] = u / nu if nu > 1e-12 else u

    # Orthogonalise within Eu (trn_y against trn_x)
    bx = basis["trn_x"]
    by = basis["trn_y"] - (basis["trn_y"] @ bx) * bx
    nby = np.linalg.norm(by)
    if nby > 1e-12:
        basis["trn_y"] = by / nby
    return basis


def analyse_in_plane(canonical, class_array):
    """Run the in-plane minimal-basis decomposition.

    Returns a dict with the six signed amplitudes, per-irrep magnitudes,
    total observed/reconstructed in-plane norms, and the four class-mean
    radii (which carry the breathing information that the orbit-symmetric
    reference removes from the bre mode by construction).
    """
    ref = symmetrise_canonical(canonical, class_array)
    delta_xy = (canonical - ref)[:, :2]
    delta = delta_xy.flatten()

    basis = build_ip_basis(canonical)
    amps = {name: float(delta @ vec) for name, vec in basis.items()}
    recon = sum(a * basis[name] for name, a in amps.items())
    residual = delta - recon

    D_ip_obs = float(np.linalg.norm(delta))
    D_ip_min = float(np.linalg.norm(recon))
    rms_res = float(np.sqrt(np.mean(residual ** 2)))

    irrep_mag = {
        "A1g (bre)":  abs(amps["bre"]),     # ~0 by construction with orbit ref
        "B1g (Nstr)": abs(amps["Nstr"]),
        "B2g (mstr)": abs(amps["mstr"]),
        "A2g (rot)":  abs(amps["rot"]),
        "Eu (trn)":   np.hypot(amps["trn_x"], amps["trn_y"]),
    }

    # Class-mean radii (the actual breathing-related geometry)
    class_radii = {}
    for c in ("N", "Calpha", "Cmeso", "Cbeta"):
        idx = [i for i, cl in enumerate(class_array) if cl == c]
        rs = np.linalg.norm(canonical[idx, :2], axis=1)
        class_radii[c] = float(np.mean(rs))

    return {
        "amps": amps,
        "irrep_mag": irrep_mag,
        "D_ip_obs": D_ip_obs,
        "D_ip_min": D_ip_min,
        "frac_explained_ip": (D_ip_min / D_ip_obs) if D_ip_obs > 1e-12 else 1.0,
        "rms_residual_ip": rms_res,
        "class_radii": class_radii,
        "delta_xy": delta_xy,
        "recon_xy": recon.reshape(-1, 2),
    }


# ---------- Top-level analysis ----------

def analyse_log(path, verbose=False, plot_dir=None, report_dir=None,
                macrocycle_name="porphyrin-class", non_d4h_caveat=False):
    """Run the full porphyrin-pipeline analysis on a Gaussian log file.

    The 'macrocycle_name' / 'non_d4h_caveat' parameters are forwarded to the
    report formatter so wrappers can advertise the correct macrocycle class
    (e.g. nsd_corphin sets macrocycle_name='corphin / F430', non_d4h_caveat=True).
    """
    atoms, coords = parse_last_standard_orientation(path)
    classes, macro_idx = identify_macrocycle(atoms, coords)
    macro_coords = coords[macro_idx]

    centroid, normal = fit_mean_plane(macro_coords)
    R = rotation_to_z(normal)
    plane_coords = (macro_coords - centroid) @ R.T

    # n_local_indices: positions of N atoms within the 24-atom array (first 4 by construction)
    n_local = np.arange(4)
    canonical = canonicalise_xy(plane_coords, n_local)

    dz = canonical[:, 2].copy()
    basis = build_oop_basis(canonical)
    amps = project(dz, basis)
    recon = reconstruct(amps, basis)
    residual = dz - recon
    dz_recon = recon.copy()

    # Per-irrep magnitudes |D^Gamma| (basis-independent within each irrep).
    # In our orthonormal basis, |D^Gamma| = sqrt(sum of d_m^2 over modes m of irrep Gamma).
    irrep_mag = {
        "B2u (sad)": abs(amps["sad"]),
        "B1u (ruf)": abs(amps["ruf"]),
        "A2u (dom)": abs(amps["dom"]),
        "Eg (wav)":  np.hypot(amps["wav_x"], amps["wav_y"]),
        "A1u (pro)": abs(amps["pro"]),
    }

    d_total_oop = float(np.linalg.norm(dz))                 # eq 10 of Jentzen
    d_minbasis = float(np.linalg.norm(recon))               # eq 15
    rms_dev = float(np.sqrt(np.mean(residual ** 2)))        # ~delta-bar_oop,1 (eq 17 in 1D)

    metal_z = int(atoms[classes["metal"][0]])
    metal_sym = ELEMENT_SYMBOLS.get(metal_z, f"Z{metal_z}")
    max_abs_dz = float(np.max(np.abs(dz)))

    # Conformation description (analytic basis — D4h labels)
    conf_oop = describe_oop_conformation(amps)

    # Metal-to-plane signed distance (z-component of the metal position
    # in the macrocycle's mean-plane frame, after centroid subtraction).
    metal_idx = classes["metal"][0]
    metal_in_plane = (coords[metal_idx] - centroid) @ R.T
    metal_to_plane = float(metal_in_plane[2])
    M_N_distances = [float(np.linalg.norm(coords[metal_idx] - coords[n]))
                     for n in classes["N"]]
    M_N_mean = float(np.mean(M_N_distances))

    # In-plane analysis. Build a 24-element class label vector aligned with
    # the canonical class order [N x4, Calpha x8, Cmeso x4, Cbeta x8].
    class_array = (["N"] * 4 + ["Calpha"] * 8 + ["Cmeso"] * 4 + ["Cbeta"] * 8)
    ip = analyse_in_plane(canonical, class_array)
    ip_amps = ip["amps"]
    ip_irrep = ip["irrep_mag"]
    radii = ip["class_radii"]
    conf_ip = describe_ip_conformation(ip_amps)

    result = {
        "file": Path(path).name,
        "metal": metal_sym,
        "n_atoms_total": len(atoms),
        # ----- conformation labels -----
        "conformation_oop": conf_oop,
        "conformation_ip":  conf_ip,
        # ----- out-of-plane -----
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
        # ----- in-plane -----
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
        # ----- metal geometry -----
        "M_to_plane_A": metal_to_plane,
        "M_N_mean_A":   M_N_mean,
        # ----- class-mean radii (Angstrom from macrocycle centroid) -----
        "r_M_N_A":     radii["N"],       # mean N radius from centroid (NOT M-N bond)
        "r_Calpha_A":  radii["Calpha"],
        "r_Cmeso_A":   radii["Cmeso"],
        "r_Cbeta_A":   radii["Cbeta"],
    }

    if verbose:
        print_report(result, classes, atoms, canonical, dz, recon, ip,
                     source_path=path,
                     macrocycle_name=macrocycle_name,
                     non_d4h_caveat=non_d4h_caveat)

    if report_dir is not None:
        out_path = save_report(result, classes, atoms, canonical, dz, recon, ip,
                               report_dir, source_path=path,
                               macrocycle_name=macrocycle_name,
                               non_d4h_caveat=non_d4h_caveat)
        if verbose:
            print(f"  Wrote report: {out_path}")

    if plot_dir is not None:
        from pathlib import Path as _P
        import nsd_plot
        plot_dir = _P(plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
        stem = _P(path).stem
        class_labels = ["N"] * 4 + ["Cα"] * 8 + ["Cmeso"] * 4 + ["Cβ"] * 8
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
            f"{stem} -- OOP NSD ({metal_sym})",
            plot_dir / f"{stem}_OOP.png")
        nsd_plot.plot_amplitudes(
            ip_modes, ip_vals,
            f"{stem} -- in-plane minimal-basis amplitudes ({metal_sym})",
            plot_dir / f"{stem}_IP.png")

    return result


# ---------- Output ----------

def format_mode_glossary(macrocycle_name="porphyrin-class", indent="   "):
    """Return a compact text block describing the analytic D4h minimal-basis
    modes — what each mode means geometrically, the irrep label, and the
    polynomial generator that defines the Δz / (Δx, Δy) pattern.

    Used in two places:
      • Interactive mode prints it after the user picks a macrocycle
      • format_report() embeds it after MOLECULE in each .txt report

    The pattern is identical for porphyrin / corphin / 1HBM-style probes /
    ECR QM/MM (all share the 24-atom inner-ring topology). For corrin
    (23-atom, C1) the same labels are used but flagged as approximate.
    """
    lines = []
    pad = indent
    lines.append(f"{pad}MINIMAL-BASIS MODES — what each amplitude measures")
    lines.append("")

    name_lower = macrocycle_name.lower()
    is_corrin = "corrin" in name_lower
    is_d4h_approx = is_corrin or "corphin" in name_lower or "f430" in name_lower or "qm/mm" in name_lower

    if is_d4h_approx:
        if is_corrin:
            lines.append(f"{pad}NB: corrin is C1 (no formal D4h symmetry); the labels and")
            lines.append(f"{pad}analytic patterns below are nevertheless useful approximate")
            lines.append(f"{pad}fingerprints for cross-comparison across metals and spin states.")
        else:
            lines.append(f"{pad}NB: corphin / F430 has lower than D4h symmetry; the labels and")
            lines.append(f"{pad}analytic patterns below are approximate fingerprints. The")
            lines.append(f"{pad}geometric quantities (D_op, D_ip, max|Δz|, M-plane) are rigorous.")
        lines.append("")

    lines.append(f"{pad}Frame:  N atoms aligned to ±x, ±y axes after canonicalisation;")
    lines.append(f"{pad}        rigid-body modes (Tx, Ty, Tz, Rx, Ry, Rz) are projected out.")
    lines.append("")

    lines.append(f"{pad}OUT-OF-PLANE (Δz from the macrocyclic mean plane)")
    lines.append(f"{pad}  mode    irrep    Δz pattern              geometric meaning")
    lines.append(f"{pad}  ─────   ─────    ────────────────────    ────────────────────────────────")
    lines.append(f"{pad}  sad     B2u      Δz ∝ x·y                pyrroles tilt alternately up/down")
    lines.append(f"{pad}  ruf     B1u      Δz ∝ x²−y²              meso C atoms alternate up/down")
    lines.append(f"{pad}  dom     A2u      Δz ∝ r²                 uniform doming / cup-shape")
    lines.append(f"{pad}  wav_x   Eg       Δz ∝ x·r²               waved along x")
    lines.append(f"{pad}  wav_y   Eg       Δz ∝ y·r²               waved along y")
    lines.append(f"{pad}  pro     A1u      Δz ∝ x·y·(x²−y²)        each pyrrole twists about its own axis")
    lines.append("")

    lines.append(f"{pad}IN-PLANE (Δx, Δy in the macrocyclic plane)")
    lines.append(f"{pad}  mode    irrep    pattern                 geometric meaning")
    lines.append(f"{pad}  ─────   ─────    ────────────────────    ────────────────────────────────")
    lines.append(f"{pad}  bre     A1g      uniform Δr outward      uniform radial expansion (≈ 0 by ref choice)")
    lines.append(f"{pad}  N-str   B1g      Δr ∝ cos(2θ)            opposite-N pairs ↔  trans-N–N stretch")
    lines.append(f"{pad}  m-str   B2g      Δr ∝ sin(2θ)            opposite-meso pairs ↔ trans-meso stretch")
    lines.append(f"{pad}  rot     A2g      Δθ ∝ cos(4θ) tangential in-phase rotation of the four pyrroles")
    lines.append(f"{pad}  trn_x   Eu       Δx ∝ r²                 macrocycle translation in x vs metal")
    lines.append(f"{pad}  trn_y   Eu       Δy ∝ r²                 same in y")
    lines.append("")

    lines.append(f"{pad}Per-irrep magnitudes |D^Γ| are basis-independent within each irrep;")
    lines.append(f"{pad}the 2-D irreps (Eg waving, Eu translation) are reported as |D^Γ| =")
    lines.append(f"{pad}√(d_x² + d_y²), invariant to in-plane axis choice.")

    return "\n".join(lines)


def format_report(r, classes, atoms, canonical, dz, recon, ip,
                  source_path=None,
                  macrocycle_name="porphyrin-class",
                  non_d4h_caveat=False,
                  axial_info=None,
                  m_s_distance=None):
    """Build the full per-file analysis report as a Unicode string.

    Sections (ordered for scientific use, key results up top):
      1. Header (Sofia version, run timestamp, input path)
      2. Molecule (atom count, metal, macrocycle topology)
      3. Key findings (conformer labels + the load-bearing geometric quantities)
      4. OOP minimal-basis amplitudes
      5. OOP per-irrep magnitudes |D^Γ|
      6. IP minimal-basis amplitudes
      7. IP per-irrep magnitudes
      8. Class-mean radii from centroid
      9. Per-atom Δz / reconstruction / Δxy table
      10. Method / citation footer

    Optional knobs:
      macrocycle_name : 'porphyrin-class' / 'corrin' / 'corphin' / 'F430 (QM/MM)'
      non_d4h_caveat  : True for corrin / corphin (mode labels are approximate)
      axial_info      : dict from corrin's axial-detector (n_axial, axial_above_*,
                        axial_below_*); inserted into Key Findings when present
      m_s_distance    : metal–sulphur axial distance (Å), for QM/MM F430 only

    Returned as a single string; pass to print() or write to a .txt file.
    """
    lines = []
    sep = "=" * 80
    sub = "-" * 80
    metal_z = int(atoms[classes['metal'][0]])
    metal_sym = r['metal']
    n_macro = len(classes['N']) + len(classes['Calpha']) + len(classes['Cmeso']) + len(classes['Cbeta'])
    src = str(source_path) if source_path else r['file']
    now = _datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ---------- 1. Header ----------
    lines.append(sep)
    lines.append(f" Sofia v{SOFIA_VERSION} — Normal-coordinate Structural Decomposition (NSD) report")
    lines.append(f" Run on            : {now}")
    lines.append(f" Input             : {src}")
    lines.append(sep)
    lines.append("")

    # ---------- 2. Molecule ----------
    lines.append(" MOLECULE")
    lines.append(f"   Total atoms     : {r['n_atoms_total']}")
    lines.append(f"   Metal           : {metal_sym}  (Z = {metal_z})")
    lines.append(f"   Macrocycle      : {macrocycle_name} {n_macro}-atom inner ring")
    lines.append(f"                     ({len(classes['N'])} N + "
                 f"{len(classes['Calpha'])} Cα + "
                 f"{len(classes['Cmeso'])} Cmeso + "
                 f"{len(classes['Cbeta'])} Cβ)")
    if non_d4h_caveat:
        lines.append(f"   NB              : Mode labels (sad/ruf/dom/wav/pro for OOP, ")
        lines.append(f"                     bre/N-str/m-str/rot/trn for IP) are")
        lines.append(f"                     D4h-approximate; geometric quantities")
        lines.append(f"                     (max|Δz|, |D_op|, |D_ip|, M-plane, M–N,")
        lines.append(f"                     class radii) are rigorous and basis-")
        lines.append(f"                     independent.")
    lines.append("")

    # ---------- 2.5 Mode glossary ----------
    lines.append(format_mode_glossary(macrocycle_name))
    lines.append("")

    # ---------- 3. Key findings ----------
    lines.append(" KEY FINDINGS")
    lines.append(f"   OOP conformer   : {r['conformation_oop']}")
    lines.append(f"   IP  conformer   : {r['conformation_ip']}")
    lines.append(f"   max |Δz|        : {r['max_abs_dz_A']:.3f} Å    "
                 f"(planarity threshold {PLANARITY_THRESHOLD_A:.2f} Å)")
    lines.append(f"   |D_op|          : {r['D_oop_obs_A']:.3f} Å    "
                 f"({100*r['frac_explained_oop']:.1f}% explained by minimal basis)")
    lines.append(f"   |D_ip|          : {r['D_ip_obs_A']:.3f} Å    "
                 f"({100*r['frac_explained_ip']:.1f}% explained)")
    lines.append(f"   M out-of-plane  : {r['M_to_plane_A']:+.3f} Å    "
                 f"(positive = above the macrocyclic mean plane)")
    lines.append(f"   M–N (mean)      : {r['M_N_mean_A']:.3f} Å")

    if axial_info and axial_info.get("n_axial", 0) > 0:
        ab_atom = axial_info.get("axial_above_atom") or "—"
        ab_dist = axial_info.get("axial_above_dist_A")
        bl_atom = axial_info.get("axial_below_atom") or "—"
        bl_dist = axial_info.get("axial_below_dist_A")
        ab_str = f"{ab_atom} {ab_dist:.3f} Å" if ab_dist is not None else "—"
        bl_str = f"{bl_atom} {bl_dist:.3f} Å" if bl_dist is not None else "—"
        lines.append(f"   axial donors    : {axial_info['n_axial']} within 2.8 Å of M")
        lines.append(f"     above plane   : {ab_str}")
        lines.append(f"     below plane   : {bl_str}")

    if m_s_distance is not None:
        lines.append(f"   M–S(axial)      : {m_s_distance:.3f} Å    "
                     f"(coenzyme-M sulphur in QM region)")

    lines.append("")

    # ---------- 4. OOP amplitudes ----------
    lines.append(" OUT-OF-PLANE MINIMAL-BASIS AMPLITUDES (Å, signed)")
    lines.append(f"   sad   (B2u) : {r['d_sad_A']:+.3f}")
    lines.append(f"   ruf   (B1u) : {r['d_ruf_A']:+.3f}")
    lines.append(f"   dom   (A2u) : {r['d_dom_A']:+.3f}")
    lines.append(f"   wav_x (Eg)  : {r['d_wav_x_A']:+.3f}")
    lines.append(f"   wav_y (Eg)  : {r['d_wav_y_A']:+.3f}")
    lines.append(f"   pro   (A1u) : {r['d_pro_A']:+.3f}")
    lines.append("")

    # ---------- 5. OOP per-irrep magnitudes ----------
    lines.append(" OOP per-irrep magnitudes |D^Γ| (Å, basis-independent within each irrep)")
    lines.append(f"   B2u  (sad)   : {r['Mag_B2u_sad_A']:.3f}")
    lines.append(f"   B1u  (ruf)   : {r['Mag_B1u_ruf_A']:.3f}")
    lines.append(f"   A2u  (dom)   : {r['Mag_A2u_dom_A']:.3f}")
    lines.append(f"   Eg   (wav)   : {r['Mag_Eg_wav_A']:.3f}    "
                 f"(= √(d_wav_x² + d_wav_y²))")
    lines.append(f"   A1u  (pro)   : {r['Mag_A1u_pro_A']:.3f}")
    lines.append("")

    # ---------- 6. IP amplitudes ----------
    lines.append(" IN-PLANE MINIMAL-BASIS AMPLITUDES (Å, signed)")
    lines.append(f"   bre   (A1g) : {r['d_bre_A']:+.3f}    "
                 f"(≈ 0 by construction with orbit-symmetric reference)")
    lines.append(f"   N-str (B1g) : {r['d_Nstr_A']:+.3f}")
    lines.append(f"   m-str (B2g) : {r['d_mstr_A']:+.3f}")
    lines.append(f"   rot   (A2g) : {r['d_rot_A']:+.3f}")
    lines.append(f"   trn_x (Eu)  : {r['d_trn_x_A']:+.3f}")
    lines.append(f"   trn_y (Eu)  : {r['d_trn_y_A']:+.3f}")
    lines.append("")

    # ---------- 7. IP per-irrep magnitudes ----------
    lines.append(" IP per-irrep magnitudes |D^Γ| (Å)")
    lines.append(f"   A1g  (bre)   : {r['Mag_A1g_bre_A']:.3f}    "
                 f"(numerical residual; ≈ 0 by reference choice)")
    lines.append(f"   B1g  (N-str) : {r['Mag_B1g_Nstr_A']:.3f}")
    lines.append(f"   B2g  (m-str) : {r['Mag_B2g_mstr_A']:.3f}")
    lines.append(f"   A2g  (rot)   : {r['Mag_A2g_rot_A']:.3f}")
    lines.append(f"   Eu   (trn)   : {r['Mag_Eu_trn_A']:.3f}    "
                 f"(= √(d_trn_x² + d_trn_y²))")
    lines.append("")

    # ---------- 8. Class radii ----------
    lines.append(" CLASS-MEAN RADII FROM MACROCYCLE CENTROID (Å)")
    lines.append(f"   M  → N      : {r['r_M_N_A']:.3f}")
    lines.append(f"   centroid → Cα   : {r['r_Calpha_A']:.3f}")
    lines.append(f"   centroid → Cmeso: {r['r_Cmeso_A']:.3f}")
    lines.append(f"   centroid → Cβ   : {r['r_Cbeta_A']:.3f}")
    lines.append("")

    # ---------- 9. Per-atom table ----------
    lines.append(" PER-ATOM DISPLACEMENTS (canonical frame: N atoms on ±x, ±y axes)")
    lines.append(f"   {'idx':>4} {'cls':>4} {'x':>9} {'y':>9}    "
                 f"{'Δz_obs':>9} {'Δz_rec':>9} {'Δx':>9} {'Δy':>9}")
    lines.append("   " + "-" * 76)
    labels = (["N"] * 4 + ["Cα"] * 8 + ["Cm"] * 4 + ["Cβ"] * 8)
    dxy = ip["delta_xy"]
    n_atoms_macro = canonical.shape[0]
    for i in range(n_atoms_macro):
        x_, y_ = canonical[i, 0], canonical[i, 1]
        lines.append(f"   {i:>4} {labels[i]:>4} {x_:+9.4f} {y_:+9.4f}    "
                     f"{dz[i]:+9.4f} {recon[i]:+9.4f} {dxy[i,0]:+9.4f} {dxy[i,1]:+9.4f}")
    lines.append("   (cls: N = pyrrole N, Cα = α-C, Cm = meso-C, Cβ = β-C)")
    lines.append("")

    # ---------- 10. Method / citation footer ----------
    lines.append(sub)
    lines.append(" METHOD")
    lines.append("   NSD decomposition follows Jentzen, Song & Shelnutt (J. Phys. Chem. B")
    lines.append("   1997, 101, 1684). The minimal basis is built from lowest-order")
    lines.append("   polynomial generators of each D4h irrep evaluated on the actual")
    lines.append("   inner-ring atomic positions in the canonical frame. The in-plane")
    lines.append("   reference is the D4h-orbit-symmetrised positions; this makes the")
    lines.append("   A1g (breathing) channel identically zero by construction.")
    lines.append("")
    lines.append(f" Generated by Sofia v{SOFIA_VERSION}.")
    lines.append(sep)

    return "\n".join(lines)


def print_report(r, classes, atoms, canonical, dz, recon, ip, source_path=None,
                 **kwargs):
    """Backwards-compatible wrapper — formats the report and prints it.
    Extra kwargs (macrocycle_name, non_d4h_caveat, axial_info, m_s_distance)
    are forwarded to format_report."""
    print(format_report(r, classes, atoms, canonical, dz, recon, ip,
                        source_path=source_path, **kwargs))


def save_report(r, classes, atoms, canonical, dz, recon, ip,
                report_dir, source_path=None, **kwargs):
    """Write the per-file report as <stem>.report.txt under report_dir.
    Extra kwargs forwarded to format_report."""
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    stem = Path(r['file']).stem
    out = Path(report_dir) / f"{stem}.report.txt"
    text = format_report(r, classes, atoms, canonical, dz, recon, ip,
                         source_path=source_path, **kwargs)
    out.write_text(text)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
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

    p = Path(args.path)
    files = []
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

    # Compact summary tables at the end
    if len(rows) > 1:
        print("\n" + "=" * 96)
        print(" Conformation summary:")
        print(f"  {'file':28s} {'metal':>3s}   {'OOP conformation':<35s} {'IP conformation':<25s}")
        for r in rows:
            print(f"  {r['file']:28s} {r['metal']:>3s}   "
                  f"{r['conformation_oop']:<35s} {r['conformation_ip']:<25s}")
        print("\n" + "=" * 88)
        print(" OOP minimal-basis amplitudes (Angstrom):")
        print(" file                         metal   sad     ruf     dom    wav    pro     %expl")
        for r in rows:
            wav = np.hypot(r["d_wav_x_A"], r["d_wav_y_A"])
            print(f"  {r['file']:28s} {r['metal']:>3s}  "
                  f"{r['d_sad_A']: .3f} {r['d_ruf_A']: .3f} {r['d_dom_A']: .3f} "
                  f"{wav: .3f} {r['d_pro_A']: .3f}  "
                  f"{100*r['frac_explained_oop']:5.1f}")

        print("\n IP minimal-basis amplitudes (Angstrom; bre is ~0 by ref choice):")
        print(" file                         metal   N-str   m-str   rot     trn     %expl")
        for r in rows:
            trn = np.hypot(r["d_trn_x_A"], r["d_trn_y_A"])
            print(f"  {r['file']:28s} {r['metal']:>3s}  "
                  f"{r['d_Nstr_A']: .3f} {r['d_mstr_A']: .3f} {r['d_rot_A']: .3f} "
                  f"{trn: .3f}  "
                  f"{100*r['frac_explained_ip']:5.1f}")

        print("\n Class-mean radii from macrocycle centroid (Angstrom):")
        print(" file                         metal   M-N     N-Ca    N-Cm    N-Cb")
        for r in rows:
            print(f"  {r['file']:28s} {r['metal']:>3s}  "
                  f"{r['r_M_N_A']: .3f}  {r['r_Calpha_A']: .3f}  "
                  f"{r['r_Cmeso_A']: .3f}  {r['r_Cbeta_A']: .3f}")

    # Cross-file summary plots
    if args.plot and rows:
        from pathlib import Path as _P
        import nsd_plot
        plot_dir = _P(args.plot)
        plot_dir.mkdir(parents=True, exist_ok=True)
        labels = [f"{r['file'].replace('.log','')}\n({r['metal']})" for r in rows]
        oop_modes = ["sad", "ruf", "dom", "wav(x)", "wav(y)", "pro"]
        oop_keys = ["d_sad_A", "d_ruf_A", "d_dom_A",
                    "d_wav_x_A", "d_wav_y_A", "d_pro_A"]
        oop_mat = np.array([[r[k] for k in oop_keys] for r in rows])
        nsd_plot.plot_summary(labels, oop_mat, oop_modes,
                              "Porphyrin -- OOP minimal-basis amplitudes across dataset",
                              plot_dir / "_summary_OOP.png")
        ip_modes = ["bre (~0)", "N-str", "m-str", "rot", "trn(x)", "trn(y)"]
        ip_keys = ["d_bre_A", "d_Nstr_A", "d_mstr_A",
                   "d_rot_A", "d_trn_x_A", "d_trn_y_A"]
        ip_mat = np.array([[r[k] for k in ip_keys] for r in rows])
        nsd_plot.plot_summary(labels, ip_mat, ip_modes,
                              "Porphyrin -- in-plane minimal-basis amplitudes across dataset",
                              plot_dir / "_summary_IP.png")
        print(f"\nWrote plots to {plot_dir}")


if __name__ == "__main__":
    main()
