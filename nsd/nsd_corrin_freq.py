#!/usr/bin/env python3
"""
NSD-style decomposition of corrin macrocycles in Gaussian opt+freq logs.

Uses the actual lowest-frequency vibrational normal modes of a chosen reference
calculation as the in-plane (IP) and out-of-plane (OOP) basis. Default reference:
Co1corrinOH (S=0 Co(I), the cleanest electronic structure per Onija et al.).

Each input log must include a frequency calculation (opt+freq route).

Pipeline (per input file):
  1. Parse last 'Standard orientation' geometry.
  2. Identify the 23-atom corrin macrocycle (4 N + 8 Calpha + 3 Cmeso + 8 Cbeta)
     by graph topology -- the corrin closure is a direct Calpha-Calpha bond.
  3. Kabsch-align the observation's macrocycle to the reference macrocycle.
  4. Compute per-macrocycle-atom displacement Delta_r in the reference's
     mean-plane frame; split into Delta_z (OOP) and Delta_xy (IP).
  5. Project Delta_r onto the 12 reference normal-mode basis vectors
     (restricted to the 23 macrocycle atoms and renormalised).

Usage:
    python3 nsd_corrin.py <file.log>
    python3 nsd_corrin.py <directory>
    python3 nsd_corrin.py <directory> --ref CO1corrinOH.log --csv out.csv
"""

import argparse
import csv
import sys
from pathlib import Path
import numpy as np


ELEMENT_SYMBOLS = {1: "H", 6: "C", 7: "N", 8: "O", 16: "S",
                   23: "V", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co",
                   28: "Ni", 29: "Cu", 30: "Zn", 45: "Rh"}
TRANSITION_METALS = {23, 24, 25, 26, 27, 28, 29, 30, 45}


# ============================================================
# Gaussian log parsing
# ============================================================

def parse_last_standard_orientation(path):
    last_atoms = last_coords = None
    with open(path) as f:
        for line in f:
            if "Standard orientation" in line:
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
    if last_atoms is None:
        raise ValueError(f"No 'Standard orientation' block found in {path}")
    return last_atoms, last_coords


def parse_normal_modes(path):
    """Return frequencies (K,), symms (list of K str), displacements (K, N, 3).

    Reads non-mass-weighted Cartesian normal-mode displacement vectors from
    the 'Frequencies --' blocks of a Gaussian opt+freq log.
    """
    freqs, symms, disps = [], [], []
    with open(path) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        if lines[i].lstrip().startswith("Frequencies --"):
            freq_vals = [float(x) for x in lines[i].split("--")[1].split()]
            symm_line = lines[i - 1].split()
            j = i + 1
            while j < len(lines) and "Atom" not in lines[j]:
                j += 1
            j += 1  # past "Atom AN  X Y Z ..." header
            atom_rows = []
            while j < len(lines):
                parts = lines[j].split()
                if len(parts) < 2 + 3 * len(freq_vals):
                    break
                try:
                    int(parts[0]); int(parts[1])
                except ValueError:
                    break
                atom_rows.append([float(x) for x in parts[2:2 + 3 * len(freq_vals)]])
                j += 1
            block = np.array(atom_rows)
            for m in range(len(freq_vals)):
                freqs.append(freq_vals[m])
                symms.append(symm_line[m] if m < len(symm_line) else "?")
                disps.append(block[:, m * 3:(m + 1) * 3])
            i = j
        else:
            i += 1
    if not freqs:
        raise ValueError(f"No 'Frequencies --' blocks found in {path}")
    return np.array(freqs), symms, np.array(disps)


# ============================================================
# Connectivity & corrin macrocycle identification
# ============================================================

def covalent_threshold(z1, z2):
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
    if {z1, z2} == {6, 8}:
        return 1.55
    return 1.85


def build_adjacency(atoms, coords):
    n = len(atoms)
    adj = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(coords[i] - coords[j])
            if d < covalent_threshold(int(atoms[i]), int(atoms[j])):
                adj[i].add(j); adj[j].add(i)
    return adj


def identify_corrin(atoms, coords):
    """Identify the 23-atom corrin macrocycle by graph topology.

    Returns (classes, macro_indices, direct_pair).
      classes: dict with keys 'metal' (1), 'N' (4), 'Calpha' (8),
               'Cmeso' (3), 'Cbeta' (8).
      macro_indices: 23-element ordered array in class order
                     [N x4, Calpha x8, Cmeso x3, Cbeta x8].
      direct_pair: tuple of indices of the directly-bonded Calpha pair
                   (the corrin closure replacing the porphyrin meso).
    """
    adj = build_adjacency(atoms, coords)

    metals = [i for i, z in enumerate(atoms) if int(z) in TRANSITION_METALS]
    if len(metals) != 1:
        raise ValueError(f"Expected 1 transition metal; found {len(metals)}")
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
                n_ca = sum(1 for m in heavy_nbrs if m in calpha)
                if n_ca == 2:
                    cmeso.add(k)
                elif n_ca == 1:
                    cbeta.add(k)
    if len(cmeso) != 3:
        raise ValueError(f"Expected 3 C-meso (corrin); found {len(cmeso)}")
    if len(cbeta) != 8:
        raise ValueError(f"Expected 8 C-beta; found {len(cbeta)}")

    direct_pairs = [(a, b) for a in calpha for b in calpha
                    if a < b and b in adj[a]]
    if len(direct_pairs) != 1:
        raise ValueError(f"Expected exactly 1 Calpha-Calpha direct bond; "
                         f"found {len(direct_pairs)}")

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


# ============================================================
# Geometry: mean plane, Kabsch alignment
# ============================================================

def fit_mean_plane(coords):
    centroid = coords.mean(axis=0)
    _, _, Vt = np.linalg.svd(coords - centroid, full_matrices=False)
    normal = Vt[-1]
    if normal[2] < 0:
        normal = -normal
    return centroid, normal


def rotation_to_z(normal):
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(normal, z); s = np.linalg.norm(v); c = float(np.dot(normal, z))
    if s < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / (s * s))


def kabsch(source_centred, target_centred):
    """Rotation R such that target ≈ source @ R.T (least-squares).

    Both inputs (N,3) must already be mean-centred.
    """
    H = source_centred.T @ target_centred
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


# ============================================================
# Reference object: holds geometry + chosen basis modes
# ============================================================

class CorrinReference:
    def __init__(self, path, n_oop=6, n_ip=6,
                 macro_threshold=0.30, oop_cutoff=0.7, ip_cutoff=0.3):
        self.path = Path(path)
        self.atoms, self.coords = parse_last_standard_orientation(path)
        self.frequencies, self.symms, self.displacements = parse_normal_modes(path)
        self.classes, self.macro_idx, self.direct_pair = identify_corrin(
            self.atoms, self.coords)

        # Mean-plane frame from the 23 macrocycle atoms (centroid + plane normal)
        macro_coords = self.coords[self.macro_idx]
        self.macro_centroid, self.normal = fit_mean_plane(macro_coords)
        self.R_plane = rotation_to_z(self.normal)
        self.macro_in_plane = (macro_coords - self.macro_centroid) @ self.R_plane.T

        # Per-mode characteristics
        K = len(self.frequencies)
        self.macro_char = np.zeros(K)
        self.oop_frac = np.zeros(K)
        for k in range(K):
            full2 = float(np.sum(self.displacements[k] ** 2))
            d_macro = self.displacements[k][self.macro_idx] @ self.R_plane.T
            macro2 = float(np.sum(d_macro ** 2))
            z2 = float(np.sum(d_macro[:, 2] ** 2))
            self.macro_char[k] = np.sqrt(macro2 / full2) if full2 > 0 else 0.0
            self.oop_frac[k] = z2 / macro2 if macro2 > 1e-12 else 0.0

        # Pick the lowest n_oop OOP-dominant and n_ip IP-dominant
        # macrocycle-localised real (positive-frequency) modes.
        sorted_idx = np.argsort(self.frequencies)
        self.oop_basis_idx = []
        self.ip_basis_idx = []
        for k in sorted_idx:
            if self.frequencies[k] <= 0:
                continue
            if self.macro_char[k] < macro_threshold:
                continue
            if self.oop_frac[k] > oop_cutoff and len(self.oop_basis_idx) < n_oop:
                self.oop_basis_idx.append(int(k))
            elif self.oop_frac[k] < ip_cutoff and len(self.ip_basis_idx) < n_ip:
                self.ip_basis_idx.append(int(k))
            if (len(self.oop_basis_idx) >= n_oop
                    and len(self.ip_basis_idx) >= n_ip):
                break

        # Build the macrocycle-restricted, unit-norm basis vectors
        # in the mean-plane frame.
        self.basis_idx = self.oop_basis_idx + self.ip_basis_idx
        self.basis_kind = (["OOP"] * len(self.oop_basis_idx)
                           + ["IP"] * len(self.ip_basis_idx))
        n_basis = len(self.basis_idx)
        self.basis_macro = np.zeros((n_basis, len(self.macro_idx), 3))
        self.basis_norm_full = np.zeros(n_basis)
        for i, k in enumerate(self.basis_idx):
            d = self.displacements[k][self.macro_idx] @ self.R_plane.T
            nrm = float(np.linalg.norm(d))
            self.basis_norm_full[i] = nrm
            self.basis_macro[i] = d / nrm if nrm > 1e-12 else d

        # Gram matrix of basis on macrocycle subspace (basis vectors are
        # orthonormal in the full 3N-6 space, but not necessarily after
        # restriction to 23 atoms). Reported for diagnostics.
        flat = self.basis_macro.reshape(n_basis, -1)
        self.gram = flat @ flat.T
        self.gram_cond = (np.linalg.cond(self.gram) if n_basis > 0 else 1.0)

    def basis_summary(self):
        rows = []
        for slot, kind in enumerate(self.basis_kind):
            k = self.basis_idx[slot]
            rows.append((slot + 1, kind, k + 1, float(self.frequencies[k]),
                         self.symms[k], float(self.macro_char[k]),
                         float(self.oop_frac[k])))
        return rows


# ============================================================
# Per-file analysis
# ============================================================

def describe_corrin_conformation(ref, amps_array, threshold_A=0.05, planar_threshold=0.05):
    """Return strings describing dominant OOP and IP modes for corrin.

    Modes are identified by their basis-slot label and reference frequency,
    since C1 corrin lacks a clean sad/ruf/dom/etc. labelling.
    """
    n_oop = len(ref.oop_basis_idx)
    oop_descriptors, ip_descriptors = [], []
    for i in range(n_oop):
        a = float(amps_array[i])
        if abs(a) >= threshold_A:
            k = ref.oop_basis_idx[i]
            nu = float(ref.frequencies[k])
            oop_descriptors.append((abs(a), f"OOP{i+1} ({nu:.0f} cm-1, {a:+.2f} A)"))
    for i in range(len(ref.ip_basis_idx)):
        a = float(amps_array[n_oop + i])
        if abs(a) >= threshold_A:
            k = ref.ip_basis_idx[i]
            nu = float(ref.frequencies[k])
            ip_descriptors.append((abs(a), f"IP{i+1} ({nu:.0f} cm-1, {a:+.2f} A)"))
    oop_descriptors.sort(key=lambda kv: -kv[0])
    ip_descriptors.sort(key=lambda kv: -kv[0])
    if not oop_descriptors:
        oop_str = "essentially undistorted vs. reference"
    else:
        oop_str = "; ".join(d[1] for d in oop_descriptors[:3])
        if len(oop_descriptors) > 3:
            oop_str += f" (+ {len(oop_descriptors)-3} more)"
    if not ip_descriptors:
        ip_str = "essentially undistorted vs. reference"
    else:
        ip_str = "; ".join(d[1] for d in ip_descriptors[:3])
        if len(ip_descriptors) > 3:
            ip_str += f" (+ {len(ip_descriptors)-3} more)"
    return oop_str, ip_str


def project_observation(obs_coords, obs_macro_idx, ref):
    """Align the observation to the reference macrocycle, decompose."""
    obs_macro = obs_coords[obs_macro_idx]
    obs_centroid = obs_macro.mean(axis=0)
    obs_centred = obs_macro - obs_centroid
    ref_centred = ref.coords[ref.macro_idx] - ref.macro_centroid
    R_align = kabsch(obs_centred, ref_centred)

    obs_macro_aligned = obs_centred @ R_align.T
    obs_in_plane = obs_macro_aligned @ ref.R_plane.T

    delta = obs_in_plane - ref.macro_in_plane  # (23, 3)

    # Raw projections (modes are unit-norm on the macrocycle)
    n_basis = len(ref.basis_macro)
    amps = np.zeros(n_basis)
    for i in range(n_basis):
        amps[i] = float(np.sum(delta * ref.basis_macro[i]))

    recon = np.zeros_like(delta)
    for i in range(n_basis):
        recon = recon + amps[i] * ref.basis_macro[i]
    residual = delta - recon

    return {
        "delta": delta,
        "amps": amps,
        "recon": recon,
        "residual": residual,
        "R_align": R_align,
        "obs_in_plane": obs_in_plane,
    }


def analyse_log(path, ref, verbose=False, plot_dir=None):
    atoms, coords = parse_last_standard_orientation(path)
    classes, macro_idx, _ = identify_corrin(atoms, coords)
    proj = project_observation(coords, macro_idx, ref)

    delta = proj["delta"]
    recon = proj["recon"]
    residual = proj["residual"]

    D_obs = float(np.linalg.norm(delta))
    D_oop_obs = float(np.linalg.norm(delta[:, 2]))
    D_ip_obs = float(np.linalg.norm(delta[:, :2]))
    D_min = float(np.linalg.norm(recon))
    D_oop_min = float(np.linalg.norm(recon[:, 2]))
    D_ip_min = float(np.linalg.norm(recon[:, :2]))
    rms_residual = float(np.sqrt(np.mean(residual ** 2)))
    max_abs_dz = float(np.max(np.abs(delta[:, 2])))
    rms_dz = float(np.sqrt(np.mean(delta[:, 2] ** 2)))

    metal_z = int(atoms[classes["metal"][0]])
    metal_sym = ELEMENT_SYMBOLS.get(metal_z, f"Z{metal_z}")

    # Conformation description (corrin: index-based mode labels)
    conf_oop, conf_ip = describe_corrin_conformation(ref, proj["amps"])

    # Geometric quantities of the observation itself (from its own atoms,
    # before alignment): metal-N, metal-plane displacement, class-mean radii.
    obs_macro = coords[macro_idx]
    obs_centroid_macro, obs_normal = fit_mean_plane(obs_macro)
    metal_pos = coords[classes["metal"][0]]
    metal_to_plane = float(np.dot(metal_pos - obs_centroid_macro, obs_normal))
    M_N_distances = [float(np.linalg.norm(metal_pos - coords[n]))
                     for n in classes["N"]]
    M_N_mean = float(np.mean(M_N_distances))

    # Class-mean radial distances from the centroid of the macrocycle in its
    # own mean-plane frame
    R_obs_plane = rotation_to_z(obs_normal)
    obs_macro_in_own_plane = (obs_macro - obs_centroid_macro) @ R_obs_plane.T
    radii = {}
    bounds = {"N": (0, 4), "Calpha": (4, 12), "Cmeso": (12, 15), "Cbeta": (15, 23)}
    for cname, (lo, hi) in bounds.items():
        rs = np.linalg.norm(obs_macro_in_own_plane[lo:hi, :2], axis=1)
        radii[cname] = float(np.mean(rs))

    result = {
        "file": Path(path).name,
        "metal": metal_sym,
        "n_atoms_total": len(atoms),
        # Conformation labels
        "conformation_oop": conf_oop,
        "conformation_ip":  conf_ip,
        # OOP / IP totals
        "max_abs_dz_A": max_abs_dz,
        "rms_dz_A": rms_dz,
        "D_obs_A": D_obs,
        "D_oop_obs_A": D_oop_obs,
        "D_ip_obs_A": D_ip_obs,
        "D_minbasis_A": D_min,
        "D_oop_minbasis_A": D_oop_min,
        "D_ip_minbasis_A": D_ip_min,
        "frac_explained": (D_min / D_obs) if D_obs > 1e-10 else 1.0,
        "frac_explained_oop": (D_oop_min / D_oop_obs) if D_oop_obs > 1e-10 else 1.0,
        "frac_explained_ip": (D_ip_min / D_ip_obs) if D_ip_obs > 1e-10 else 1.0,
        "rms_residual_A": rms_residual,
        # geometry
        "M_to_plane_A": metal_to_plane,
        "M_N_mean_A": M_N_mean,
        "r_N_A": radii["N"],
        "r_Calpha_A": radii["Calpha"],
        "r_Cmeso_A": radii["Cmeso"],
        "r_Cbeta_A": radii["Cbeta"],
    }
    # Per-mode amplitudes: name them by basis slot and mode frequency
    for i, (kind, k) in enumerate(zip(ref.basis_kind, ref.basis_idx)):
        nu = float(ref.frequencies[k])
        col = f"d_{kind}{i+1 if i < len(ref.oop_basis_idx) else i+1-len(ref.oop_basis_idx)}_v{int(round(nu))}_A"
        result[col] = float(proj["amps"][i])

    if verbose:
        print_report(result, classes, atoms, coords, macro_idx, ref, proj)

    if plot_dir is not None:
        from pathlib import Path as _P
        import nsd_plot
        plot_dir = _P(plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
        stem = _P(path).stem
        class_labels = ["N"] * 4 + ["Cα"] * 8 + ["Cmeso"] * 3 + ["Cβ"] * 8
        # Mode labels: "OOPn ν cm⁻¹" / "IPn ν cm⁻¹"
        mode_labels = []
        for slot, kind in enumerate(ref.basis_kind):
            k = ref.basis_idx[slot]
            nu = float(ref.frequencies[k])
            local_n = (slot + 1 if slot < len(ref.oop_basis_idx)
                       else slot + 1 - len(ref.oop_basis_idx))
            mode_labels.append(f"{kind}{local_n}\n{nu:.0f} cm⁻¹")
        amplitudes = list(proj["amps"])
        dz_obs = proj["delta"][:, 2]
        dz_recon = proj["recon"][:, 2]
        macro_xy = ref.macro_in_plane[:, :2]
        nsd_plot.plot_per_file_combined(
            macro_xy, dz_obs, dz_recon, class_labels,
            mode_labels, amplitudes,
            f"{stem} -- corrin NSD ({metal_sym})",
            plot_dir / f"{stem}.png")

    return result, proj


# ============================================================
# Output
# ============================================================

def print_basis_table(ref):
    print("Reference: " + str(ref.path.name))
    print(f"  Atoms total      : {len(ref.atoms)}")
    print(f"  Macrocycle atoms : 23 (4 N, 8 Calpha, 3 Cmeso, 8 Cbeta)")
    print(f"  Direct Calpha-Calpha pair (corrin closure): atoms "
          f"{ref.direct_pair[0]+1}, {ref.direct_pair[1]+1}")
    print(f"  Macrocycle z-RMSD at equilibrium: "
          f"{float(np.std(ref.macro_in_plane[:, 2])):.4f} A")
    print(f"  Macrocycle max |z| at equilibrium: "
          f"{float(np.max(np.abs(ref.macro_in_plane[:, 2]))):.4f} A")
    print(f"  Total normal modes parsed: {len(ref.frequencies)}")
    print(f"  Basis Gram condition number: {ref.gram_cond:.2f}")
    print()
    print("Selected basis modes (lowest-frequency macrocycle-dominant modes):")
    print(" slot  kind  mode#   freq(cm-1)  symm   macro_char  oop_frac")
    for slot, kind, k1, freq, symm, mc, oof in ref.basis_summary():
        print(f"   {slot:>3}  {kind:>3}    {k1:>4}    {freq:>9.2f}   "
              f"{symm:>3}     {mc:>5.2f}      {oof:>5.2f}")


def print_report(r, classes, atoms, coords, macro_idx, ref, proj):
    print("=" * 78)
    print(f" File              : {r['file']}")
    print(f" Metal             : {r['metal']}  "
          f"(atomic number {int(atoms[classes['metal'][0]])})")
    print(f" Atoms in molecule : {r['n_atoms_total']}")
    print(f" Macrocycle atoms  : 23  (4 N, 8 Calpha, 3 Cmeso, 8 Cbeta)")
    print()
    print(f" Conformation (vs. Co1corrinOH reference):")
    print(f"   out-of-plane : {r['conformation_oop']}")
    print(f"   in-plane     : {r['conformation_ip']}")
    print()
    print(f" Geometry of the macrocycle:")
    print(f"   metal-to-plane     : {r['M_to_plane_A']: .4f} A")
    print(f"   M-N mean distance  : {r['M_N_mean_A']: .4f} A")
    print(f"   r_N   (centroid)   : {r['r_N_A']: .4f} A")
    print(f"   r_Calpha           : {r['r_Calpha_A']: .4f} A")
    print(f"   r_Cmeso            : {r['r_Cmeso_A']: .4f} A")
    print(f"   r_Cbeta            : {r['r_Cbeta_A']: .4f} A")
    print()
    print(f" Distortion (after Kabsch alignment to reference):")
    print(f"   max |dz| on macrocycle  : {r['max_abs_dz_A']: .4f} A")
    print(f"   rms dz                  : {r['rms_dz_A']: .4f} A")
    print(f"   |D_obs| total           : {r['D_obs_A']: .4f} A")
    print(f"   |D_obs| out-of-plane    : {r['D_oop_obs_A']: .4f} A")
    print(f"   |D_obs| in-plane        : {r['D_ip_obs_A']: .4f} A")
    print(f"   |D_minbasis| total      : {r['D_minbasis_A']: .4f} A   "
          f"({100*r['frac_explained']:5.1f}% explained)")
    print(f"   |D_minbasis| OOP / IP   : {r['D_oop_minbasis_A']: .4f} / "
          f"{r['D_ip_minbasis_A']: .4f} A   "
          f"({100*r['frac_explained_oop']:.1f}% / "
          f"{100*r['frac_explained_ip']:.1f}%)")
    print(f"   rms residual            : {r['rms_residual_A']: .4f} A")
    print()
    print(f" Mode amplitudes (Angstrom, signed):")
    print(f"   slot  kind  mode#  freq(cm-1)   amplitude")
    for i, (kind, k) in enumerate(zip(ref.basis_kind, ref.basis_idx)):
        nu = float(ref.frequencies[k])
        a = float(proj["amps"][i])
        print(f"   {i+1:>3}   {kind:>3}   {k+1:>4}   {nu:>9.2f}   {a: .4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="Gaussian .log file or directory of .log files")
    ap.add_argument("--ref",
                    default="/Users/adrianbranzanic/Downloads/Calcule_articol/CorrinOH/Logs/Co1corrinOH.log",
                    help="Reference Gaussian opt+freq log (default: Co1corrinOH.log)")
    ap.add_argument("--csv", help="Write per-file results to this CSV")
    ap.add_argument("-q", "--quiet", action="store_true",
                    help="Suppress per-file detailed report")
    ap.add_argument("--n-oop", type=int, default=6, help="Number of OOP basis modes")
    ap.add_argument("--n-ip", type=int, default=6, help="Number of IP basis modes")
    ap.add_argument("--plot", metavar="DIR",
                    help="Write per-file PNG plots and a cross-file summary to DIR")
    args = ap.parse_args()

    print("Building corrin reference and basis ...")
    ref = CorrinReference(args.ref, n_oop=args.n_oop, n_ip=args.n_ip)
    print()
    print_basis_table(ref)
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
            r, _ = analyse_log(f, ref, verbose=not args.quiet, plot_dir=args.plot)
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
        print(" Conformation summary (dominant modes vs. Co1corrinOH reference):")
        for r in rows:
            print(f"  {r['file']:30s} {r['metal']:>3s}")
            print(f"      OOP : {r['conformation_oop']}")
            print(f"      IP  : {r['conformation_ip']}")
        print("\n" + "=" * 90)
        print(" Distortion magnitudes (Angstrom):")
        print(" file                           metal   |D_oop|  |D_ip|  M-plane  M-N    %expl")
        for r in rows:
            print(f"  {r['file']:30s} {r['metal']:>3s}    "
                  f"{r['D_oop_obs_A']: .4f}  {r['D_ip_obs_A']: .4f}  "
                  f"{r['M_to_plane_A']: .3f}  {r['M_N_mean_A']: .3f}  "
                  f"{100*r['frac_explained']:5.1f}")

    # Cross-file summary plot
    if args.plot and rows:
        from pathlib import Path as _P
        import nsd_plot
        plot_dir = _P(args.plot)
        plot_dir.mkdir(parents=True, exist_ok=True)
        labels = [f"{r['file'].replace('.log','')}\n({r['metal']})" for r in rows]
        # Collect amplitudes from rows; per-mode column names follow the template
        # used in analyse_log: f"d_{kind}{local_n}_v{nu_int}_A"
        mode_keys = [k for k in rows[0].keys() if k.startswith("d_OOP") or k.startswith("d_IP")]
        # Sort: OOP modes first by local index, then IP modes
        oop_keys = sorted([k for k in mode_keys if k.startswith("d_OOP")])
        ip_keys  = sorted([k for k in mode_keys if k.startswith("d_IP")])
        ordered = oop_keys + ip_keys
        # Build mode labels from key names: "d_OOP1_v50_A" -> "OOP1\n50 cm-1"
        def lbl(k):
            parts = k[2:-2].split("_")  # strip 'd_' and '_A'
            kind_n = parts[0]; nu = parts[1].lstrip("v")
            return f"{kind_n}\n{nu} cm$^{{-1}}$"
        mode_labels = [lbl(k) for k in ordered]
        mat = np.array([[r[k] for k in ordered] for r in rows])
        nsd_plot.plot_summary(labels, mat, mode_labels,
                              "Corrin -- minimal-basis amplitudes across the dataset (Co1corrinOH reference)",
                              plot_dir / "_summary.png")
        print(f"\nWrote plots to {plot_dir}")


if __name__ == "__main__":
    main()
