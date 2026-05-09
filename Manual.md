# Sofia — theory and implementation manual

In-depth reference for the macrocyclic NSD pipeline that backs Sofia.
Companion to **[README.md](README.md)** (quickstart).

The CLI entry point is **`sofia.py`**; underneath it dispatches to the
single-purpose analysis modules described in §9.

---

## Table of contents

1. [Background](#1-background)
2. [Macrocycle topology and graph identification](#2-macrocycle-topology-and-graph-identification)
3. [Mean-plane fit and canonical orientation](#3-mean-plane-fit-and-canonical-orientation)
4. [Out-of-plane minimal basis](#4-out-of-plane-minimal-basis)
5. [In-plane minimal basis and the breathing mode](#5-in-plane-minimal-basis-and-the-breathing-mode)
6. [Conformer-label syntax](#6-conformer-label-syntax)
7. [Basis-construction variants](#7-basis-construction-variants)
8. [Validation](#8-validation)
9. [Per-script implementation notes](#9-per-script-implementation-notes)
10. [Output formats](#10-output-formats)
11. [References](#11-references)

---

## 1. Background

Macrocyclic tetrapyrrole cofactors (porphyrin in heme, corrin in B12, corphin
in F430) deviate from planarity in characteristic ways. Jentzen, Song and
Shelnutt (1997) showed that for porphyrin the deviation can be expanded in a
basis of low-frequency normal modes of the planar D4h-symmetric reference, and
that six modes — saddled (B2u), ruffled (B1u), domed (A2u), waved (Eg, two
components), propellered (A1u) — capture > 90 % of the out-of-plane (OOP)
distortion in synthetic and protein-bound porphyrins.

The pipeline in this directory implements the Jentzen–Song–Shelnutt
decomposition with the following design choices:

- **Analytic minimal basis.** Instead of force-field eigenvectors, the basis
  vectors are the lowest-order polynomial generators of each D4h irrep,
  evaluated on the actual macrocyclic geometry (Section 4). This makes the
  construction methodologically uniform across porphyrin / corrin / corphin
  and across different program packages (Gaussian, ORCA QM/MM).
- **In-plane decomposition.** A parallel in-plane (IP) decomposition is
  carried out using the lowest-order generators of the gerade and ungerade
  in-plane irreps (Section 5).
- **Geometry-only.** No frequency calculation is required. A separate tool
  (`nsd_corrin_freq.py`) implements the Gaussian-freq variant for cross-check.
- **Topology-aware.** The macrocycle inner ring is identified by graph
  walking (Section 2), so peripheral substituents (full cobalamin side
  chains, the DMB nucleotide, axial ligands, F430 F-ring saturation) are
  rejected automatically.
- **Closed-form basis.** The minimal basis is unambiguous and reproducible —
  no dependence on functional, basis set, or solver settings, and no need
  for an additional freq job (cf. Variants iii in Section 7).

---

## 2. Macrocycle topology and graph identification

### 2.1 Adjacency

`nsd_porphyrin.build_adjacency(atoms, coords)` builds an undirected graph
from interatomic distances. The covalent threshold depends on element pair
(see `covalent_threshold`):

| pair | threshold (Å) |
|---|---|
| any–metal (Z ∈ {23–30, 45}) | 2.55 |
| any–H | 1.25 |
| C–C | 1.75 |
| C–N | 1.65 |
| N–N | 1.55 |
| other | 1.85 |

These are conservative upper bounds chosen to handle DFT geometries with
dispersion-stretched bonds while not bridging through-space contacts. The
graph is built once per file in O(N²) time (negligible for N ≤ 200).

### 2.2 Porphyrin (`identify_macrocycle`)

The 24-atom inner ring is identified by:

1. Locate the unique transition metal.
2. Take its 4 bonded N atoms as the pyrrole nitrogens.
3. The 8 carbons bonded to those N atoms are Cα.
4. For each non-Cα carbon bonded to ≥ 1 Cα: count its **non-hydrogen**
   neighbours that are themselves Cα.
   - If 2 → it bridges two pyrroles → **Cmeso**.
   - If 1 → it's part of a pyrrole 5-ring → **Cβ**.

This handles bare metalloporphine, porphine + axial ligands (1HBM model),
and tetrahydroporphyrin (THP / bacteriochlorin) since the 24-atom inner-ring
topology is identical in all three.

The macrocycle index array is stored in canonical class order
`[N×4, Cα×8, Cmeso×4, Cβ×8]`.

### 2.3 Corrin (`identify_corrin`)

Corrin has 23 inner-ring atoms (one direct Cα–Cα closure replacing one
meso bridge). The naive porphyrin identifier breaks for two reasons:

1. **Cobalamin's DMB nitrogen** sits within bond range of the metal but is
   not a pyrrole-N. We need to distinguish pyrrole N's from non-pyrrole.
2. **Peripheral side chains** in the full cobalamin model add C atoms whose
   degree pattern looks like Cβ to a naive identifier.

The corrin identifier solves both by **walking pyrrole 5-rings explicitly**.
A pyrrole N is one for which there exist two C neighbours `Cα1, Cα2` such
that the cycle `N – Cα1 – Cβ1 – Cβ2 – Cα2 – N` exists in the graph (all
carbons, no hydrogens). For each such 5-ring, the two Cβ's are recorded as
macrocycle Cβ.

Cmeso atoms are then identified as carbons bonded to two Cα's from
**different** pyrroles. Finally a sanity check verifies exactly one Cα–Cα
direct bond (the corrin closure).

The result is a `classes` dict with `metal/N/Calpha/Cmeso/Cbeta` index lists
plus `direct_pair = (Cα_i, Cα_j)` recording the corrin closure.

### 2.4 Corphin / F430 (`nsd_corphin.py`)

Corphin's inner ring has the same 24-atom porphyrin-class topology
(4 N + 8 Cα + 4 Cmeso + 8 Cβ); only the peripheral saturation pattern and
the fused F-ring distinguish it. The corphin pipeline therefore reuses
`nsd_porphyrin.identify_macrocycle` — the F-ring atoms are simply outside
the inner-ring graph walk.

### 2.5 QM/MM (`nsd_qmmm.parse_xyz`)

`*.QMRegion.xyz` is a standard XYZ file written by ORCA QM/MM containing
the QM-region atoms. Element symbols are mapped to atomic numbers via a
lookup table, and the same `identify_macrocycle` routine handles topology
identification.

---

## 3. Mean-plane fit and canonical orientation

### 3.1 SVD mean-plane fit

`fit_mean_plane(coords24)` computes the centroid of the macrocycle atoms,
centres the coordinates, runs `np.linalg.svd`, and takes the right singular
vector with the smallest singular value as the plane normal. The sign of
the normal is chosen so its z-component is positive (so the global +z
direction is consistent across files).

This is the standard least-squares plane fit: minimising
∑ᵢ (rᵢ · n̂)² over unit n̂ yields the smallest right-singular-vector. The
residual ‖Δz_obs − Δz_recon‖ / √N is reported as the mean-plane RMSD.

### 3.2 Rotation to z

`rotation_to_z(normal)` returns a 3×3 rotation matrix that maps the plane
normal onto +z, computed by the Rodrigues formula:

```
R = I + K + K² (1 − cosθ) / sin²θ
```

where K is the skew-symmetric cross-product matrix of `normal × ẑ`, and θ
is the angle between them. After this rotation, the macrocycle z-coordinate
is the signed displacement from the mean plane (the OOP coordinate Δz).

### 3.3 In-plane canonicalisation

`canonicalise_xy(coords24, n_local_indices)` rotates the macrocycle in the
xy plane so that the four pyrrole-N atoms lie close to the ±x, ±y axes. The
twist angle is determined by averaging the four `arctan2(y_N, x_N)` angles
modulo π/2 (the four-fold symmetry of the N orbit). After canonicalisation,
the OOP and IP basis vectors below have a fixed phase relative to the
molecular geometry, enabling consistent sign conventions across files.

---

## 4. Out-of-plane minimal basis

The OOP basis is built from the lowest-order polynomial generators of the
D4h irreps that act on the z-coordinate (B2u, B1u, A2u, Eg, A1u; the only
gerade representation for an axial mode is A2u – more precisely, the A2u of
the D4h *out-of-plane* sector). Geometrically, on the canonical (x, y)
positions of the 24 inner-ring atoms:

| mode | irrep | analytic generator (Δz pattern) |
|---|---|---|
| sad | B2u | x² − y² |
| ruf | B1u | xy |
| dom | A2u | r² (= x² + y²) — uniform z projected out |
| wav_x | Eg | x · r² — Ry projected out |
| wav_y | Eg | y · r² — Rx projected out |
| pro | A1u | xy(x² − y²) |

(See `build_oop_basis` in `nsd_porphyrin.py`.)

### 4.1 Why these generators

Each entry is the **lowest-order polynomial in (x, y)** that transforms as
the named irrep under D4h. Higher-order generators of the same irrep are
linearly dependent on these (within the 24-atom subspace) and are therefore
omitted. With six independent generators for the six 1-D + 1-D + 1-D + 2-D +
1-D = 6 lowest-order modes, the basis spans the full 6-dimensional minimal
NSD subspace.

### 4.2 Rigid-body projection

The plain polynomial patterns `r²` (dom), `x · r²` (wav_x), `y · r²`
(wav_y) overlap with the rigid-body modes:

| rigid body | Δz pattern | overlaps with |
|---|---|---|
| translation Tz | constant 1 | dom |
| rotation Rx (about x-axis) | −y | wav_y |
| rotation Ry (about y-axis) | x | wav_x |

These three rigid-body z-displacements are projected out before
normalisation, by Gram–Schmidt against an orthonormal basis of the
rigid-body subspace. The resulting basis vectors describe **internal**
deformations of the macrocycle independent of how the molecule is
positioned in space.

### 4.3 Eg orthogonalisation

The Eg representation is two-dimensional. After rigid-body projection,
wav_x and wav_y can have non-zero overlap (the geometric Eg basis is not
automatically orthogonal in the 24-atom L² inner product). A second
Gram–Schmidt step orthogonalises wav_y against wav_x within the Eg
subspace, yielding two orthogonal Eg basis vectors. The total Eg distortion
magnitude `|D_Eg| = √(d_wav_x² + d_wav_y²)` is rotationally invariant and
basis-independent.

### 4.4 Projection and signed amplitudes

For an observed Δz vector on the 24 inner-ring atoms, the signed amplitude
in mode `m` is `d_m = Δz · ê_m`, where ê_m is the orthonormalised mode
vector. The reconstruction is `Δz_recon = Σ_m d_m ê_m`, and the
fraction-explained metric is:

```
frac_explained = 1 − ‖Δz_obs − Δz_recon‖² / ‖Δz_obs‖²
```

For the Jentzen-equivalent quantity `D_oop = ‖Δz_obs‖` (eq. 10 of the 1997
paper), and the minimal-basis distortion `D_oop_minbasis = ‖Δz_recon‖`
(eq. 15).

### 4.5 Sign convention

For the 1-D irreps (sad, ruf, dom, pro) the mode is unique up to sign. The
geometric sign of each basis vector follows the convention "positive value
at the (+,+) atom of that class" (e.g. ruf is positive at the meso atom in
the (+,+) quadrant). Under reflection z → −z, all signed amplitudes flip
sign while per-irrep magnitudes are invariant — exactly as required.

---

## 5. In-plane minimal basis and the breathing mode

The IP decomposition uses the same canonical (x, y) coordinates as the OOP
decomposition, but treats the 48-dimensional displacement vector
`(Δx_0, Δy_0, …, Δx_{23}, Δy_{23})`. The minimal basis comprises six modes
spanning the lowest-order analytic patterns of the IP irreps (`build_ip_basis`):

| mode | irrep | analytic pattern (Δx, Δy) |
|---|---|---|
| bre  | A1g | (x/r, y/r) — radial outward |
| Nstr | B1g | cos2θ · (x/r, y/r) |
| mstr | B2g | sin2θ · (x/r, y/r) |
| rot  | A2g | cos4θ · (−y/r, x/r) — tangential |
| trn_x | Eu | (r², 0) |
| trn_y | Eu | (0, r²) |

with `cos2θ = (x²−y²)/r²`, `sin2θ = 2xy/r²`, `cos4θ = cos²2θ − sin²2θ`.

The rigid-body modes Tx, Ty (uniform in-plane translation) and Rz
(in-plane rotation about z) are projected out before normalisation, and
trn_y is orthogonalised against trn_x within Eu (analogous to the Eg
treatment for OOP).

### 5.1 The orbit-symmetric reference

The IP "observation" is `Δxy = canonical_xy − reference_xy`, where the
reference is constructed by the **orbit-symmetric** rule: for every atom,
take the average over the 8 D4h operations that map it to (within an
angular tolerance of 20°) another atom of the same class, then set the
reference position to that average (`symmetrise_canonical`). For corrin
(where one meso slot is missing) the operations pointing to the missing
slot are skipped instead of being mapped to the wrong atom.

This convention has an important consequence for the breathing mode:
because the reference radii are the orbit-mean radii of each class, the
A1g (uniform-radial) channel is identically zero by construction. The
reported `d_bre` values are residual numerical roundoff (typically < 0.01
Å) and should not be interpreted as a physical breathing amplitude.

The same orbit-symmetric reference also makes the IP analysis
basis-rigorous within the remaining 5 IP irreps: the orbit-mean radii are
the unique set of class-symmetric radii compatible with D4h, so the Δxy
residual lives in the rigorous 5-dimensional IP-non-A1g subspace. This is
the right trade-off when comparing across metals: the bre channel is then
a free parameter (set to zero) that absorbs no information, while the
remaining 5 channels carry well-defined geometric meaning.

If a non-zero bre amplitude is desired (e.g. for comparison against
crystallographic porphine bond lengths), a fixed external reference would
have to be introduced. We did not pursue this — the orbit-symmetric choice
gives a clean, basis-rigorous IP decomposition that is comparable across
all macrocycles.

### 5.2 Class-mean radii

The IP analysis additionally reports `r_M_N`, `r_Calpha`, `r_Cmeso`,
`r_Cbeta` — the orbit-mean distances of each macrocycle class from the
mean-plane centroid. These are the geometric quantities that the
orbit-symmetric reference choice removes from the bre mode by construction.
Together with the bre = 0 channel, they fully characterise the radial
component of the macrocyclic geometry without confounding it with the
A1g IP "amplitude".

---

## 6. Conformer-label syntax

`describe_oop_conformation(amps)` and `describe_ip_conformation(ip_amps)`
return a string that compactly describes the dominant distortion modes:

1. Compute the magnitude of every mode (`|d_m|` for 1-D irreps,
   `√(d_x² + d_y²)` for 2-D irreps).
2. The largest is the **dominant mode**.
3. If the dominant magnitude is below `planar_threshold = 0.05 Å`, return
   `"essentially planar"` (OOP) or `"essentially symmetric"` (IP). If the
   dominant magnitude is between 0.001 Å and `planar_threshold`, the
   dominant mode is named in parentheses, e.g.
   `"essentially planar (saddled-proned, 4 mÅ)"`.
4. Otherwise list every mode whose magnitude is at least
   `secondary_fraction = 30 %` of the dominant, in decreasing order,
   joined by `" + "`.

The 0.05 Å planarity threshold is roughly the X-ray positional uncertainty
for synthetic porphyrins (Jentzen 1997). The 30 % secondary cutoff is
chosen to capture chemically meaningful mixtures (e.g. saddled + ruffled
in 1HBM_Rh) while filtering out small residuals.

### Examples

| amplitudes | label |
|---|---|
| sad = 1.4 Å, others < 0.1 Å | `saddled` |
| sad = 1.4 Å, ruf = 0.5 Å (36 % of sad) | `saddled + ruffled` |
| ruf = 0.36 Å, sad = 0.16 Å (44 % of ruf) | `ruffled + saddled` |
| all < 0.05 Å, max sad = 4 mÅ | `essentially planar (saddled-proned, 4 mÅ)` |
| all < 1 mÅ | `essentially planar` |

---

## 7. Basis-construction variants

Two basis variants are supported. The structural conclusions are
basis-independent (per-irrep magnitudes are uniquely defined within each
1-D irrep, and rotationally invariant within each 2-D irrep), so the
choice between them affects only the assignment of signed amplitudes to
mode labels.

| variant | eigenvector source | applied to | role |
|---|---|---|---|
| (i) analytic D4h | low-order polynomial generators (Section 4) | porphyrin / corrin / corphin / QM/MM | **MAIN** — used throughout |
| (ii) Gaussian freq | Gaussian-computed normal modes (opt+freq job) | corrin only (e.g. Co1corrinOH reference) | optional; `nsd_corrin_freq.py` |

### Why Variant (i) is the default

- **Methodologically uniform:** the same construction works for every
  macrocycle Sofia handles, including ECR QM/MM cofactors.
- **No DFT freq cost:** built from any optimised geometry without an
  additional freq job.
- **Closed-form:** unambiguous, reproducible, no functional/basis
  dependence.

### Variant (ii): Gaussian freq basis

`nsd_corrin_freq.py` parses the `Frequencies` blocks of a Gaussian opt+freq
log, identifies the 12 lowest-frequency macrocycle-localised real-positive
normal modes, splits them into 6 OOP-dominant + 6 IP-dominant by
out-of-plane fraction (z² / total² of the displacement vector), and uses
those as the basis. The macrocycle-localisation criterion is `‖d_macro‖ /
‖d_full‖ ≥ 0.30`. After construction, the macrocycle-restricted Gram matrix
is computed and its condition number reported; values < 10 indicate the
basis is well-conditioned on the macrocycle subspace.

This variant is more rigorous in principle (the basis is the actual
molecular eigenvector, not an analytic pattern) but requires opt+freq logs
and is reference-specific. We applied it to Co1corrinOH only as a
cross-check on the corrin analytic basis.

---

## 8. Validation

Five internal checks are applied during development:

1. **Topology.** Each input passes the four-pyrrole / one-metal / four-N
   check (porphyrin) or the three-meso + one-Cα–Cα-closure check (corrin).
   Cobalamin-class structures pass after the DMB pyrrole-5-ring filter.
2. **Mean-plane RMSD residual.** ‖Δz_obs − Δz_recon‖ / √N is ≤ 0.10 Å for
   porphyrin / corphin minimal bases, ≤ 0.30 Å for corrin (which is C1, so
   the D4h minimal basis explains less of the total distortion).
3. **Reflection sanity.** Flipping z → −z flips the sign of all OOP
   amplitudes, leaves all per-irrep magnitudes invariant, and leaves D_oop
   invariant.
4. **Translation sanity.** Rigid translation in (x, y) leaves all
   D4h-derived amplitudes invariant.
5. **Numerical bre.** With orbit-symmetric IP reference, |d_bre| < 0.02 Å
   on every input.

---

## 9. Per-script implementation notes

All `nsd_*.py` modules live in the **`nsd/`** subdirectory of the repo;
`sofia.py` adds that directory to `sys.path` at startup so the modules
import each other by their flat names (`import nsd_porphyrin as porph`)
without needing package-style relative imports.

### `nsd_porphyrin.py` — core implementation (1045 lines)

The reference implementation; all other scripts import from it. Key modules:

- `parse_last_standard_orientation(path)` — reads the last `Standard
  orientation` block of a Gaussian log (so partially-converged jobs still
  produce usable geometry).
- `build_adjacency`, `identify_macrocycle` — graph construction and
  porphyrin topology identification.
- `fit_mean_plane`, `rotation_to_z`, `canonicalise_xy` — coordinate frame.
- `build_oop_basis`, `project`, `reconstruct` — analytic OOP basis +
  projection.
- `build_ip_basis`, `symmetrise_canonical`, `analyse_in_plane` — IP basis +
  orbit-symmetric reference + projection.
- `describe_oop_conformation`, `describe_ip_conformation` — conformer-label
  syntax.
- `format_report`, `print_report`, `save_report` — scientific .txt report
  builders (used by `--report DIR`).
- `analyse_log(path, verbose, plot_dir)` — top-level per-file analysis.
- `print_report` — verbose stdout report (suppressed by `--quiet`).
- `main()` — argparse + directory iteration + CSV writing.

### `nsd_corphin.py` — wrapper (156 lines)

Imports `nsd_porphyrin` and calls `analyse_log` directly. Adds the
corphin-specific banner/caveat (corphin is C1, mode labels approximate) and
a corphin cross-file summary table (D_oop, ruf, M-plane, etc. across files).
Otherwise identical to `nsd_porphyrin.py`.

### `nsd_corrin.py` — corrin pipeline (531 lines)

Imports `nsd_porphyrin` for everything except the topology identifier.
`identify_corrin` is corrin-specific (Section 2.3) with the pyrrole-5-ring
walk and the DMB filter. The class array is 23 atoms
(`["N"]*4 + ["Calpha"]*8 + ["Cmeso"]*3 + ["Cbeta"]*8`), and
`symmetrise_canonical` handles the missing meso slot via its
angular-tolerance filter.

The script also detects axial coordination: every heavy atom (Z ≥ 6) within
2.8 Å of the metal that is **not** one of the four pyrrole N's is reported
as an axial donor, with separate `axial_above` / `axial_below` slots
distinguishing the two faces of the macrocycle by signed canonical-frame z
of the donor. This handles cobalamin's H₂O above and DMB-N below cleanly.

### `nsd_corrin_freq.py` — freq-based corrin (661 lines)

Self-contained (does not import from `nsd_porphyrin` for the basis logic).
Parses both `Standard orientation` and `Frequencies / NAtoms × 3 …` blocks
of a Gaussian opt+freq log. The `CorrinReference` class encapsulates a
reference geometry plus the selected normal-mode basis.

For each observation, the inner-ring geometry is Kabsch-aligned to the
reference (translation + rotation, no scale) before projection. This is
necessary because the Gaussian-computed normal modes are tied to the
specific reference orientation — unlike the analytic basis, which is
rebuilt per file.

### `nsd_qmmm.py` — F430-in-QM/MM (323 lines)

Reads `*.QMRegion.xyz` (standard XYZ, the QM-region geometry of an ORCA
QM/MM job) and the corresponding `*.out` (for multiplicity readout). Calls
`nsd_porphyrin.identify_macrocycle` (corphin-class topology) on the QM
geometry. Adds two QM/MM-specific outputs:

- `M_S_axial_A` — distance from the metal to the nearest sulphur atom in
  the QM region (the F430 coenzyme M ligand).
- `spin` — read from the ORCA `Mult ....  N` line via a regex.

### `nsd_plot.py` — shared plotting (248 lines)

Five plot functions, all using matplotlib with the non-interactive `Agg`
backend so the scripts can run on headless systems:

- `plot_dz_profile` — observed vs reconstructed Δz vs atom index.
- `plot_amplitudes` — signed-amplitude bar chart for the minimal basis.
- `plot_per_file_combined` — single PNG combining the above two.
- `plot_mode_scatter` — 2D scatter of two mode amplitudes across files
  (used for the planarity-quadrant figures and the analytic-vs-DFT scatter).
- `plot_summary` — grid of bar charts, one panel per mode, bars across
  files (used for the cross-dataset summary panels).

---

## 10. Output formats

### CSV row (per file)

The columns are listed in [README.md](README.md#csv-columns-common-to-all-tools).
Briefly: `file`, `metal`, `n_atoms_total`, conformer labels, geometric
quantities (`max_abs_dz_A`, `D_oop_obs_A`, `D_ip_obs_A`, `M_to_plane_A`,
`M_N_mean_A`), signed amplitudes (`d_<mode>_A`), per-irrep magnitudes
(`Mag_<irrep>_A`), and class radii (`r_<class>_A`). The corrin axial fields
(`n_axial`, `axial_above_atom`, `axial_above_dist_A`, `axial_below_atom`,
`axial_below_dist_A`) and QM/MM fields (`spin`, `multiplicity`,
`M_S_axial_A`) are appended where applicable.

All amplitudes are in Ångström; the units convention follows Jentzen 1997:
each per-atom Δz contributes `(Δz_atom)²` to `D_oop² = Σ Δz²`, so D_oop is
the Euclidean norm of the Δz vector (NOT a per-atom RMS).

### Per-file plots

For each input log, two PNGs are written:
- `<stem>_OOP.png` — Δz profile across the 24 (or 23) inner-ring atoms,
  observed vs minimal-basis reconstruction, with class boundaries marked.
- `<stem>_IP.png` — same for the IP analysis.

A combined `<stem>_combined.png` (Δz profile + amplitude bar chart) is also
written by some tools.

### Cross-dataset summary plots

In each plot directory:
- `_summary_OOP.png` — grid of bar charts, one panel per OOP mode, bars
  across files.
- `_summary_IP.png` — same for IP modes.

### Verbose stdout report

Without `--quiet`, each per-file analysis prints a multi-section report:

1. Banner line with file name, total atoms, metal, multiplicity if known.
2. Topology check — confirms 4 N + 8 Cα + 4 Cmeso + 8 Cβ and prints the
   atom indices in the canonical order.
3. Mean-plane fit — centroid, normal, RMSD residual.
4. OOP amplitudes — six signed `d_<mode>` and five per-irrep |D^Γ|.
5. IP amplitudes — six signed `d_<mode>` and five per-irrep |D^Γ|.
6. Conformer labels.
7. Geometric quantities — M-plane, M-N, class radii.
8. Per-atom Δz table — `idx cls x y dz_obs dz_rec dx dy`.

---

## 11. References

1. Jentzen, W.; Song, X.-Z.; Shelnutt, J. A. *Structural Characterization of
   Synthetic and Protein-Bound Porphyrins in Terms of the Lowest-Frequency
   Normal Coordinates of the Macrocycle.* J. Phys. Chem. B **1997**, *101*,
   1684–1699. — The original NSD method paper.

2. Onija, R.; Cosma, S.-R.; Brânzanic, A. M. V.; Silaghi-Dumitrescu, R.
   *Rationales for the choice of metals for super-reduced biological metal
   centers: cobalt in cobalamin vs. nickel in F430.* — The first manuscript
   using Sofia.
