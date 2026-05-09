# Sofia

**Sofia** is a Normal-coordinate Structural Decomposition (NSD) program for
metalloporphyrin / metallocorrin / metallocorphin macrocycles, with support
for ORCA QM/MM output. It analyses geometric distortions of tetrapyrrole
cofactors from Gaussian opt logs and writes per-file reports, CSV summaries,
and PNG figures.

The name nods to the **dome** — both Hagia Sofia's, and the *dom* (A2u)
mode of the NSD minimal basis.

For a guided tutorial on what NSD is, what Sofia decomposes, and how to
read the output, see **[docs/Sofia_lesson.html](docs/Sofia_lesson.html)**
(open in any browser). New to group theory? Start with the prerequisite
primer at **[docs/group_theory_lesson.html](docs/group_theory_lesson.html)**
— it covers symmetry operations, point groups, irreducible
representations, and the D<sub>4h</sub> character table at exactly the
depth needed for the main lesson. For the theory, basis construction,
validation strategy, and per-script implementation notes, see
**[Manual.md](Manual.md)**.

---

## Requirements

- Python ≥ 3.9
- `numpy` — core
- `matplotlib` — only needed for `--plot` (per-file figures + cross-dataset summary plots)
- **PyMOL** — only needed if you want to render the `viz` bundles. Sofia
  always *generates* the `.pdb` + `.pml` files; PyMOL is the program that
  opens them.

```bash
pip install -r requirements.txt          # numpy + matplotlib

# PyMOL (optional, install separately if you want viz rendering):
brew install --cask pymol                # macOS
sudo apt-get install pymol               # Linux (Debian/Ubuntu)
conda install -c conda-forge pymol-open-source   # any OS
```

No compiled extensions, no Gaussian/ORCA binaries needed at run time —
Sofia reads text logs / `.xyz` files only.

---

## Quick start

```bash
# Single file
python3 sofia.py porphyrin examples/porphyrin/Co1_Pro.log

# Whole directory + CSV + plots (recursive — picks up .log files in subdirs)
python3 sofia.py porphyrin path/to/dir --csv out.csv --plot plots/

# Don't know which macrocycle you have? Let Sofia decide.
python3 sofia.py auto examples/corrin/Co1corrinOH.log
```

That's it — Sofia identifies the macrocycle inner ring by graph walking,
fits the mean plane, and projects displacements onto the analytic D4h
minimal basis.

---

## The Sofia CLI

```
python3 sofia.py <subcommand> <path> [--csv FILE] [--plot DIR] [-q]
```

| subcommand | macrocycle | input | typical use |
|---|---|---|---|
| `porphyrin` | porphyrin-class 24-atom inner ring | Gaussian `.log` | metallo-porphine, tetrahydroporphyrin (chlorin / bacteriochlorin), porphine + peripheral substitution, corphin / F430 — they all share the same inner-ring topology |
| `corphin` | corphin / F430 | Gaussian `.log` | thin wrapper around `porphyrin` with corphin-specific banner and caveats |
| `corrin` | corrin 23-atom inner ring | Gaussian `.log` | bare Metal(I)-corrin, di-OH corrin, full cobalamin (decoordinated and axially coordinated) |
| `corrin-freq` | corrin (Variant iii basis) | Gaussian opt+freq `.log` | rigorous corrin analysis using DFT-computed normal modes as the basis |
| `qmmm` | F430 in ORCA QM/MM | `.QMRegion.xyz` + `.out` | F430 in ECR enzyme |
| `auto` | any | Gaussian `.log` | auto-detects topology and dispatches to the right pipeline |

Internally Sofia delegates to single-purpose modules (`nsd_porphyrin.py`,
`nsd_corrin.py`, …) that are also runnable on their own — calling
`sofia.py porphyrin foo.log` is identical to `python3 nsd_porphyrin.py
foo.log`.

---

## Output

### CSV columns (per file)

| column | meaning |
|---|---|
| `file` | log file stem |
| `metal` | element symbol of the macrocycle metal |
| `n_atoms_total` | atom count of the input geometry |
| `conformation_oop` / `conformation_ip` | conformer-label fingerprint (mode abbreviations joined by `+`) |
| `max_abs_dz_A` | maximum absolute z-displacement of any inner-ring atom (Å) |
| `D_oop_obs_A`, `D_ip_obs_A` | total observed OOP / IP distortion magnitudes |
| `D_oop_minbasis_A`, `D_ip_minbasis_A` | reconstruction from the minimal basis |
| `frac_explained_oop`, `frac_explained_ip` | fraction of total distortion captured |
| `d_<mode>_A` | signed amplitude (Å) of `sad/ruf/dom/wav_x/wav_y/pro` (OOP) and `bre/Nstr/mstr/rot/trn_x/trn_y` (IP) |
| `Mag_<irrep>_A` | per-irrep magnitude \|D^Γ\|, basis-independent within each irrep |
| `M_to_plane_A` | signed metal-to-mean-plane distance |
| `M_N_mean_A` | mean metal–N distance |
| `r_<class>_A` | mean centroid-to-class distance |

`corrin` adds: `n_axial`, `axial_above_atom`, `axial_above_dist_A`,
`axial_below_atom`, `axial_below_dist_A`.

`qmmm` adds: `spin`, `multiplicity`, `M_S_axial_A`.

### Per-file plots

`<stem>_OOP.png` (Δz profile + minimal-basis reconstruction) and
`<stem>_IP.png` for each input.

### Cross-dataset summary plots

`_summary_OOP.png` and `_summary_IP.png` per directory: a grid of bar charts,
one panel per minimal-basis mode, with bars across files in the dataset.

---

## Visualizing the minimal-basis modes (PyMOL)

Sofia ships a generator that builds two PyMOL sessions showing what each
minimal-basis mode actually looks like — a planar metallo-macrocycle plus
distorted variants for the OOP modes (sad, ruf, dom, dom_pure, wav_x,
wav_y, pro) and the IP modes (bre, N-str, m-str, rot, trn_x, trn_y),
arranged side by side for direct comparison. Available for three
macrocycle classes:

| `--type` | inner ring | reference geometry |
|---|---|---|
| `porphyrin` (default) | 24-atom (4 N + 8 Cα + 4 Cmeso + 8 Cβ) | bare metalloporphine |
| `corrin` | 23-atom (4 N + 8 Cα + 3 Cmeso + 8 Cβ) | bare Co(I) corrin |
| `corphin` | 24-atom + F430 peripheral shell | F430-Ni²⁺ |

Three ways to invoke it:

```bash
# 1. Top-level Sofia subcommand
python3 sofia.py viz                          # porphyrin (default), small amplitude
python3 sofia.py viz --type corrin  --amp 1.0 # corrin, bigger amplitude
python3 sofia.py viz --type corphin --amp 0.7 # corphin / F430 with peripheral shell
python3 sofia.py viz --amp 1.0 --render       # porphyrin + headless PyMOL PNG previews

# 2. Inside the interactive prompt: after you pick a macrocycle, Sofia
#    asks "Generate the visualization bundle now?" — say yes and answer
#    a few prompts for type / amplitude / output dir / PNG render. The
#    default viz type tracks the analysis cmd (corrin → corrin viz,
#    corphin/qmmm → corphin viz, everything else → porphyrin viz), but
#    all three are always offered.
python3 sofia.py

# 3. Direct call to the underlying module
python3 -m nsd.visualize_porphyrin --amp 1.0
python3 -m nsd.visualize_corrin    --amp 1.0
python3 -m nsd.visualize_corphin   --amp 0.7
```

Output lands in `viz/<type>/`: planar reference + seven OOP variants +
six IP variants as `.pdb` files (with `CONECT` records so PyMOL keeps
the macrocyclic skeleton — including corrin's Cα–Cα closure and
corphin's F-ring lactam — drawn at any amplitude), plus two `.pml`
scripts:

```bash
pymol viz/porphyrin/porphyrin_oop_modes.pml  # OOP modes (Δz, tilted view)
pymol viz/porphyrin/porphyrin_ip_modes.pml   # IP modes  (Δx, Δy, top-down)
pymol viz/corrin/corrin_oop_modes.pml        # corrin OOP grid
pymol viz/corrin/corrin_ip_modes.pml         # corrin IP grid
pymol viz/corphin/corphin_oop_modes.pml      # F430 OOP grid (with peripheral shell)
pymol viz/corphin/corphin_ip_modes.pml       # F430 IP grid
```

For corrin and corphin, the irrep labels (B2u, B1u, A2u, …) are
D4h-approximate — corrin is C1 (only 3 mesos) and corphin / F430 has
sp³ pyrrole carbons + the fused F-ring breaking four-fold symmetry.
Sofia's analytic basis still produces well-defined polynomial shapes
on the inner ring; the labels carry their porphyrin-derived meaning
by analogy.

For corphin specifically, peripheral substituent atoms (methyl groups,
propionamide chains, F-ring atoms) are mapped to their nearest
inner-ring ancestor by BFS and inherit the average displacement of
those parents — so the peripheral shell stays rigidly attached to the
inner ring under each distortion.

If you used `--render` (or answered "yes" to "Also render PNG previews"
in the interactive prompt), `porphyrin_oop.png` and `porphyrin_ip.png`
will be in the same directory.

Each OOP distortion is the analytic Δz pattern of the corresponding mode
applied to the planar reference; bonded H atoms move with their parent C
so C–H bonds stay rigid.

The dom (A2u) mode is shown twice: `dom` is the chemistry-textbook
deoxy-Hb hybrid picture (metal at apex, all atoms above the plane —
combines pure A2u shape + Tz translation + metal-out-of-plane), and
`dom_pure` is Sofia's analytic basis applied directly (Tz-orthogonal
saucer, atoms above and below the plane — faithful to the lowest A2u
normal mode of Jentzen–Song–Shelnutt 1997).

For IP modes, bre is shown as a uniform radial expansion (Sofia's
analytic A1g amplitude is ≈ 0 by orbit-symmetric reference choice, so
showing the basis verbatim would be invisible) and trn_x / trn_y are
shown as uniform translations of the ring vs the metal — the chemistry
meaning of "translation".

---

## Conformer-label syntax

Every conformer string lists modes by amplitude: largest first, then any
secondary mode whose amplitude is ≥ 30 % of the dominant, joined by ` + `.
If the dominant amplitude is below 0.05 Å the label is `essentially planar`
(OOP) or `essentially symmetric` (IP).

Mode abbreviations:

| OOP | | IP | |
|---|---|---|---|
| `sad` | saddled (B2u) | `bre` | breathing (A1g, ≈ 0 by reference choice) |
| `ruf` | ruffled (B1u) | `N-str` | N-stretched (B1g) |
| `dom` | domed (A2u) | `m-str` | meso-stretched (B2g) |
| `wav` | waved (Eg, x and y) | `rot` | rotated (A2g) |
| `pro` | propellered (A1u) | `trn` | translated (Eu, x and y) |

---

## Repo layout

```
sofia-macrocycle/
├── sofia.py              # master CLI — entry point
├── nsd/                  # analysis modules (Sofia auto-loads from here)
│   ├── nsd_porphyrin.py     # porphyrin-class core implementation
│   ├── nsd_corphin.py       # corphin / F430 wrapper
│   ├── nsd_corrin.py        # corrin pipeline
│   ├── nsd_corrin_freq.py   # Variant (ii) freq-based corrin
│   ├── nsd_qmmm.py          # ORCA QM/MM pipeline
│   ├── nsd_plot.py          # shared plotting helpers
│   ├── visualize_common.py     # shared geometry + PML helpers (used by corrin/corphin viz)
│   ├── visualize_porphyrin.py  # PyMOL viz bundle for the porphyrin minimal basis
│   ├── visualize_corrin.py     # PyMOL viz bundle for the corrin minimal basis
│   └── visualize_corphin.py    # PyMOL viz bundle for the corphin / F430 minimal basis
├── examples/             # one stripped log per macrocycle for smoke tests
├── docs/                 # HTML lessons — open docs/Sofia_lesson.html in a browser
│   ├── Sofia_lesson.html
│   ├── group_theory_lesson.html  # prerequisite group theory primer
│   └── figures/          # rendered viz grids embedded in the lesson
├── Manual.md             # detailed theory + implementation reference
├── README.md             # this file
├── LICENSE               # MIT
└── requirements.txt
```

---

## Common errors

| error | cause | fix |
|---|---|---|
| `Expected exactly one transition metal; found 0` | log has no metal in the supported set (Z = 23–30, 45) | extend `ELEMENT_SYMBOLS` in `nsd_porphyrin.py` |
| `Expected 4 pyrrole N bonded to metal; found 5` | a 5th N is coordinated (e.g. cobalamin DMB) | use `corrin` subcommand — it has the pyrrole-5-ring walk that filters DMB |
| `Expected 1 Calpha-Calpha direct bond (corrin closure); found 0` | the input is porphyrin, not corrin | use `porphyrin` subcommand instead |
| `No 'Standard orientation' block found` | log truncated before any geometry was written, or it is from an ORCA / NWChem run | check the file is a Gaussian log; for ORCA QM/MM use the `qmmm` subcommand |

---

## Citing Sofia

If you use Sofia in published work, please cite the Onija / Cosma /
Brânzanic / Silaghi-Dumitrescu manuscript and the Jentzen–Song–Shelnutt
1997 paper that defines the underlying NSD method:

> Jentzen, W.; Song, X.-Z.; Shelnutt, J. A. *Structural Characterization
> of Synthetic and Protein-Bound Porphyrins in Terms of the
> Lowest-Frequency Normal Coordinates of the Macrocycle.*
> J. Phys. Chem. B **1997**, *101*, 1684–1699.
