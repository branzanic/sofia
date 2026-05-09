#!/usr/bin/env python3
"""
Sofia — Normal-coordinate Structural Decomposition of macrocyclic distortions.

Named after the dome of Hagia Sofia (and after the dom mode of the NSD basis).

Sofia analyses metalloporphyrin / metallocorrin / metallocorphin / F430 macrocycles
from Gaussian opt logs and ORCA QM/MM output. Pick the right subcommand for your
macrocycle, or use 'auto' to let Sofia pick by graph-walking the inner ring.

Usage:
    sofia.py <subcommand> <path> [options]

Subcommands:
    interactive  — step-by-step prompts for macrocycle / path / options
                   (default when run with no arguments).
    porphyrin    — porphyrin-class 24-atom inner ring (Mn/Fe/Co/Ni/Rh-porphine,
                   tetrahydroporphyrin, 1HBM-style amide-substituted porphyrinoids,
                   corphin / F430 — all share the same inner-ring topology).
    corrin       — corrin 23-atom inner ring (bare Metal(I)-corrin, di-OH corrin,
                   full cobalamin decoordinated and axially-coordinated).
    corphin      — corphin / F430 (thin wrapper around the porphyrin pipeline,
                   with corphin-specific banner / caveats).
    qmmm         — F430 in ORCA QM/MM output (reads .QMRegion.xyz + .out).
    corrin-freq  — Variant (iii): corrin analysis using Gaussian-derived
                   normal modes (requires opt+freq log + --ref).
    auto         — auto-detect the macrocycle type from graph topology.
    viz          — generate a PyMOL visualization bundle of the minimal-basis
                   modes (planar reference + distorted variants for each OOP
                   and IP mode). Useful for teaching / publishing what each
                   amplitude means geometrically.

Each subcommand accepts:
    <path>          a single .log/.xyz file or a directory of files
    --csv FILE      write per-file results to FILE (one row per input)
    --plot DIR      write per-file PNG figures and cross-dataset summary plots
    -q, --quiet     suppress per-file detailed report
    -h, --help      show subcommand-specific help

Examples:
    sofia.py                                      # interactive mode (recommended for first-time users)
    sofia.py porphyrin /path/to/porphyrins/      --csv out.csv --plot plots/
    sofia.py corrin    Co1corrinOH.log
    sofia.py qmmm      /path/to/QMMM_articol/    --csv qmmm.csv
    sofia.py auto      Co1_Pro.log               # detects → porphyrin pipeline
    sofia.py viz                                  # porphyrin minimal-basis viz (default)
    sofia.py viz       --type corrin --amp 1.0    # corrin minimal-basis viz + bigger amp
    sofia.py viz       --type corphin             # corphin / F430 viz with peripheral shell
    sofia.py viz       --amp 1.0 --render         # build viz + PNG previews via headless PyMOL

Documentation:
    docs/Sofia_lesson.html         — guided tutorial: what NSD is, what Sofia
                                     decomposes, how to read the output, and
                                     a SALC-analogy bridge for chemists.
    docs/group_theory_lesson.html  — prerequisite primer on D4h symmetry,
                                     irreducible representations, character
                                     reduction, and basis construction (with
                                     fully worked examples). Read this first
                                     if A1g / B2u / Eg feel like hieroglyphics.
    Manual.md                      — theory + implementation reference.
    README.md                      — quick reference and worked CLI examples.
"""

import argparse
import sys
from pathlib import Path

# Allow running without installing: ensure the bundled nsd/ module dir is on
# sys.path so 'import nsd_porphyrin' (etc.) resolves to the local sources.
_SELF_DIR = Path(__file__).resolve().parent
_NSD_DIR = _SELF_DIR / "nsd"
if not _NSD_DIR.is_dir():
    raise SystemExit(
        f"sofia.py: expected the analysis modules at {_NSD_DIR} but the "
        f"directory is missing. Re-clone the repo or restore the nsd/ folder."
    )
if str(_NSD_DIR) not in sys.path:
    sys.path.insert(0, str(_NSD_DIR))


def _eternal_return_tree():
    """ASCII tree evoking Irfan's "The Eternal Return" (2015) album cover —
    together with Eliade's Myth of the Eternal Return (archaic time as
    cyclical, the axis mundi binding heaven and earth) and Heidegger's
    Freiburg lectures reading Nietzsche's eternal recurrence as the
    affirmation of being in becoming. A small visible reminder that
    every metalloporphyrin Sofia measures is a ritual of departure from
    and return to its D4h ideal — the negentropic cycle made geometric.
    The crown and roots mirror through the horizon line; the trunk is
    the axis. Compositional choices follow Irfan's bare-tree-against-
    pale-sky aesthetic.
    """
    B, F, T = "\\", "/", 30          # backslash, forward slash, trunk column
    pad = lambda left_chars: " " * (T - left_chars)
    rows = [
        " " * T + "*",
        pad(6)  + ".   " + B*2 + "|" + F*2 + "   .",
        pad(6)  + B*2 + "  " + B*2 + "|" + F*2 + "  " + F*2,
        pad(5)  + B*5 + "|" + F*5,
        pad(4)  + B*4 + "|" + F*4,
        pad(3)  + B*3 + "|" + F*3,
        pad(2)  + B*2 + "|" + F*2,
        pad(1)  + B   + "|" + F,
        pad(0)  + "|",
        pad(11) + "─"*11 + "┼" + "─"*11,
        pad(0)  + "|",
        pad(1)  + F   + "|" + B,
        pad(2)  + F*2 + "|" + B*2,
        pad(3)  + F*3 + "|" + B*3,
        pad(4)  + F*4 + "|" + B*4,
        pad(6)  + F*3 + "   |   " + B*3,
        pad(5)  + ".   " + F + "|" + B + "   .",
        " " * T + "*",
    ]
    return "\n".join(rows)


_BANNER = (
    _eternal_return_tree() + "\n\n"
    "==============================================================\n"
    " Sofia — NSD analysis of macrocyclic distortions\n"
    "==============================================================\n"
)


# Map each subcommand to its underlying module.
_SUBMODS = {
    "porphyrin":   "nsd_porphyrin",
    "corrin":      "nsd_corrin",
    "corphin":     "nsd_corphin",
    "qmmm":        "nsd_qmmm",
    "corrin-freq": "nsd_corrin_freq",
}


def _delegate(cmd, argv):
    """Hand off the remaining argv to the underlying module's main()."""
    import importlib
    mod = importlib.import_module(_SUBMODS[cmd])
    # The submodule's argparse reads sys.argv. Construct a minimal argv so
    # the subcommand sees only its own arguments.
    sys.argv = [_SUBMODS[cmd]] + list(argv)
    return mod.main()


def _detect_macrocycle(path: Path) -> str:
    """Peek at one log/xyz file and decide which pipeline to dispatch.

    Returns one of: 'porphyrin', 'corrin', 'corphin', 'qmmm'.
    Falls back to 'porphyrin' if topology is ambiguous.
    """
    import nsd_porphyrin as porph
    # Pick a representative file from a directory
    p = Path(path)
    if p.is_dir():
        # Recursive: prefer a .log anywhere under the directory, fall back to
        # .QMRegion.xyz. Used only to peek at one file for topology detection.
        candidates = sorted(p.rglob("*.log")) or sorted(p.rglob("*.QMRegion.xyz"))
        if not candidates:
            raise SystemExit(f"auto-detect: no .log or .QMRegion.xyz files in {p}")
        p = candidates[0]
    print(f"[auto-detect] inspecting {p.name}")

    # ORCA QM/MM .QMRegion.xyz dispatches to qmmm
    if p.suffix == ".xyz" or p.name.endswith(".QMRegion.xyz"):
        return "qmmm"

    # Gaussian log: try the porphyrin identifier, then corrin
    atoms, coords = porph.parse_last_standard_orientation(str(p))
    try:
        porph.identify_macrocycle(atoms, coords)
        # Heuristic: if metal has 4 bonded N and 24-atom inner ring is found,
        # this is porphyrin-class. Distinguish corphin vs porphyrin by atom
        # count: corphin (and its F430 variants) typically has > 50 atoms
        # because of the F-ring and saturation; bare metalloporphine ≤ 41.
        n_total = len(atoms)
        if n_total >= 50:
            return "corphin"
        return "porphyrin"
    except ValueError:
        pass

    # Try corrin
    try:
        import nsd_corrin
        nsd_corrin.identify_corrin(atoms, coords)
        return "corrin"
    except (ImportError, ValueError):
        pass

    raise SystemExit(
        f"auto-detect: could not identify macrocycle topology in {p.name}.\n"
        "  Try a specific subcommand: sofia.py porphyrin|corrin|corphin|qmmm <path>"
    )


# -------------------- interactive mode --------------------

# Macrocycle choice menu shown to the user. Order matches ergonomic frequency.
_INTERACTIVE_CHOICES = [
    ("auto",        "auto-detect (recommended — Sofia walks the graph for you)"),
    ("porphyrin",   "porphyrin (incl. THP/bacteriochlorin and 1HBM-style probes)"),
    ("corrin",      "corrin (bare, di-OH, full cobalamin)"),
    ("corphin",     "corphin / F430 (free cofactor)"),
    ("qmmm",        "F430 in ORCA QM/MM (.QMRegion.xyz + .out)"),
    ("corrin-freq", "corrin with Gaussian-freq basis (Variant iii; needs opt+freq + --ref)"),
]


def _ask(prompt, default=None, choices=None):
    """Prompt user for input. Re-prompt on invalid choice. Returns string.

    - default: shown in [brackets] and used if user just hits Enter.
    - choices: optional list of valid responses (case-insensitive).
    """
    if default is not None:
        prompt_str = f"{prompt} [{default}]: "
    else:
        prompt_str = f"{prompt}: "
    while True:
        try:
            ans = input(prompt_str).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[cancelled]")
            sys.exit(130)
        if not ans and default is not None:
            ans = default
        if not ans:
            print("  (please enter a value)")
            continue
        if choices and ans.lower() not in [c.lower() for c in choices]:
            print(f"  (not one of: {', '.join(choices)})")
            continue
        return ans


def _ask_yes_no(prompt, default=False):
    """Yes/no prompt. Default is the value used on Enter."""
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        try:
            ans = input(f"{prompt} {suffix}: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n[cancelled]")
            sys.exit(130)
        if not ans:
            return default
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("  (please answer y/yes or n/no)")


def _path_completer(text, state):
    """readline completer for filesystem paths.

    Lists files / directories matching the typed prefix; appends '/' to
    directories so the user can chain tabs to drill in. Honours leading '~'.
    """
    import glob
    import os
    # Expand ~ for matching, then translate back to keep the user's input style
    expanded = os.path.expanduser(text)
    raw_matches = glob.glob(expanded + "*")
    matches = []
    home = os.path.expanduser("~")
    for m in raw_matches:
        # Append slash to directories
        if os.path.isdir(m) and not m.endswith(os.sep):
            m = m + os.sep
        # If the user typed a ~-relative path, present matches the same way
        if text.startswith("~") and m.startswith(home):
            m = "~" + m[len(home):]
        matches.append(m)
    # Directories first (so cycling lands on the most-likely-next-step), then files,
    # alphabetical within each group.
    matches.sort(key=lambda x: (not x.endswith(os.sep), x.lower()))
    if state < len(matches):
        return matches[state]
    return None


def _enable_path_completion():
    """Enable tab-completion for filesystem paths on the next input() call.

    Returns a function that restores the previous readline state. Safe to call
    even if readline is unavailable (e.g. exotic Python builds without it) —
    falls back to a no-op so prompts still work.
    """
    try:
        import readline
    except ImportError:
        return lambda: None

    prev_completer = readline.get_completer()
    prev_delims = readline.get_completer_delims()

    readline.set_completer(_path_completer)
    # Strip path-meaningful characters from the word-break set so the completer
    # sees the whole partial path, not just the last component.
    readline.set_completer_delims(" \t\n")

    # macOS Python is usually linked against libedit, which has different
    # bindings than GNU readline. Try GNU first; if its binding throws or
    # libedit is detected, use libedit's syntax.
    is_libedit = "libedit" in getattr(readline, "__doc__", "") or ""
    try:
        if is_libedit:
            readline.parse_and_bind("bind ^I rl_complete")
        else:
            readline.parse_and_bind("tab: complete")
    except Exception:
        # Best-effort: completion may simply not work, but prompt still does.
        pass

    def restore():
        try:
            readline.set_completer(prev_completer)
            readline.set_completer_delims(prev_delims)
        except Exception:
            pass

    return restore


def _ask_path(prompt, must_exist=True, allow_empty=False):
    """Prompt for a filesystem path with tab-completion and ~ expansion.

    - must_exist=True: re-prompt if the path doesn't exist.
    - allow_empty=True: an empty answer (Enter) returns None instead of
      re-prompting; use this when the path is genuinely optional.
    Tab cycles through matching files / directories in the current word.
    """
    restore = _enable_path_completion()
    try:
        while True:
            # Allow empty input here; re-implement the inner loop because _ask
            # rejects empty values without a default.
            try:
                if allow_empty:
                    raw = input(f"{prompt}: ").strip()
                else:
                    raw = _ask(prompt)
            except (EOFError, KeyboardInterrupt):
                print("\n[cancelled]")
                sys.exit(130)
            if allow_empty and not raw:
                return None
            # Strip surrounding quotes the user may have copy-pasted from a shell
            if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ("'", '"'):
                raw = raw[1:-1]
            p = Path(raw).expanduser()
            if must_exist and not p.exists():
                print(f"  (path does not exist: {p})")
                continue
            return p
    finally:
        restore()


# -------------------- visualization helpers --------------------

def _run_viz(amplitude, out_dir, render_png=False, log_path=None,
             macrocycle="porphyrin"):
    """Run the visualization tool for porphyrin or corrin.

    Generates the .pdb + .pml bundle under `out_dir` (planar reference + 7 OOP
    + 6 IP distorted variants + two .pml scripts). If `render_png=True` and
    PyMOL is on PATH, also produces .png previews of both grids via headless
    PyMOL.

    `macrocycle` is "porphyrin" (24-atom ring; reference defaults to
    examples/porphyrin/Co1_Pro.log) or "corrin" (23-atom ring; reference
    defaults to examples/corrin/Co1corrinOH.log).
    """
    import subprocess
    import shutil

    if macrocycle == "corrin":
        module = "nsd.visualize_corrin"
        prefix = "corrin"
    elif macrocycle == "corphin":
        module = "nsd.visualize_corphin"
        prefix = "corphin"
    else:
        module = "nsd.visualize_porphyrin"
        prefix = "porphyrin"

    cmd = [sys.executable, "-m", module,
           "--amp", str(amplitude), "--out", str(out_dir)]
    if log_path is not None:
        cmd += ["--log", str(log_path)]
    print()
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        print(f"  visualization failed (exit {rc})", file=sys.stderr)
        return False

    if not render_png:
        return True

    pymol_bin = shutil.which("pymol")
    if pymol_bin is None:
        print(f"\n  pymol not found in PATH — skipping PNG render.")
        print(f"  Open the .pml files in PyMOL manually to view the grid.")
        return True

    print()
    print(f"  Rendering PNGs via headless PyMOL …")
    for pml_file, png_file in [
        (f"{prefix}_oop_modes.pml", f"{prefix}_oop.png"),
        (f"{prefix}_ip_modes.pml",  f"{prefix}_ip.png"),
    ]:
        pml_path = Path(out_dir) / pml_file
        png_path = Path(out_dir) / png_file
        if not pml_path.exists():
            continue
        result = subprocess.run(
            [pymol_bin, "-cq", str(pml_path),
             "-d", f"ray 1600, 1000; png {png_path}, dpi=120; quit"],
            capture_output=True)
        if png_path.exists():
            print(f"    wrote {png_path}")
        else:
            print(f"    failed to render {png_file} (PyMOL exit {result.returncode})")
    return True


def _interactive_viz_prompt(cmd_macrocycle):
    """Optional viz step in the interactive flow. If the user opts in, prompt
    for type / amplitude / output dir / render-PNG and run the visualizer.

    The viz type is chosen independently of the analysis macrocycle so the
    user can always reach the corrin viz (or porphyrin viz). The default is
    keyed to the analysis subcommand: 'corrin' analysis defaults to corrin
    viz, everything else defaults to porphyrin viz.

    Returns True if the visualization was generated, False otherwise."""
    print()
    print(" Optional — visualize the minimal-basis modes in PyMOL")
    print(" (one .pdb per mode + .pml scripts showing each OOP/IP distortion")
    print("  applied to a planar metallo-macrocycle, side-by-side for comparison)")

    if not _ask_yes_no("  Generate the visualization bundle now?", default=False):
        return False

    # Pick which macrocycle to visualise. Default tracks the analysis cmd so
    # users who picked 'corrin'/'corphin' get the matching viz by default;
    # other paths default to porphyrin. All three options are always offered
    # so any viz is reachable from any analysis path.
    if cmd_macrocycle == "corrin":
        default_type = "corrin"
    elif cmd_macrocycle in ("corphin", "qmmm"):
        default_type = "corphin"
    else:
        default_type = "porphyrin"
    print()
    print("   Available viz types:")
    print("     1) porphyrin  — 24-atom inner ring (porphine, THP, 1HBM-style probes)")
    print("     2) corrin     — 23-atom inner ring with 3 mesos (corrin, cobalamin)")
    print("     3) corphin    — 24-atom inner ring + F430 peripheral shell")
    raw_type = _ask("  Visualization type (number or name)",
                    default=default_type,
                    choices=["porphyrin", "corrin", "corphin", "1", "2", "3"])
    if raw_type == "1":
        viz_macrocycle = "porphyrin"
    elif raw_type == "2":
        viz_macrocycle = "corrin"
    elif raw_type == "3":
        viz_macrocycle = "corphin"
    else:
        viz_macrocycle = raw_type.lower()

    if viz_macrocycle == "corrin":
        print()
        print("   Note: corrin is C1 symmetry (23-atom ring, 3 mesos only).")
        print("   The D4h-pattern minimal basis still produces well-defined")
        print("   shapes on these 23 atoms, but the irrep labels are")
        print("   D4h-approximate.")
        default_out = "viz/corrin"
    elif viz_macrocycle == "corphin":
        print()
        print("   Note: corphin / F430 is not D4h. Saturation pattern (sp³")
        print("   pyrrole carbons + fused F-ring) breaks four-fold symmetry,")
        print("   so the irrep labels are D4h-approximate. Peripheral atoms")
        print("   (methyls, propionamides, F-ring) move rigidly with their")
        print("   nearest inner-ring atom under each distortion.")
        default_out = "viz/corphin"
    else:
        default_out = "viz/porphyrin"

    amp_str = _ask("  Distortion amplitude (Å)", default="1.0")
    try:
        amplitude = float(amp_str)
    except ValueError:
        print(f"  invalid amplitude {amp_str!r}; skipping viz")
        return False
    out_dir = _ask("  Output directory", default=default_out)
    render = _ask_yes_no("  Also render PNG previews via headless PyMOL?",
                         default=False)
    return _run_viz(amplitude, out_dir, render_png=render,
                    macrocycle=viz_macrocycle)


def interactive_mode():
    """Walk the user through macrocycle / path / options and dispatch."""
    print(_BANNER)
    print(" Interactive mode. Press Ctrl-C to abort at any prompt.")
    print(" Default values are shown in [brackets] — hit Enter to accept.\n")

    # 1. Macrocycle type
    print(" Step 1/4 — Which macrocycle type?")
    for i, (key, desc) in enumerate(_INTERACTIVE_CHOICES, 1):
        print(f"   {i}) {key:<12} — {desc}")
    print()
    valid_keys = [k for k, _ in _INTERACTIVE_CHOICES]
    valid_nums = [str(i) for i in range(1, len(_INTERACTIVE_CHOICES) + 1)]
    raw = _ask("  Choice (number or name)", default="auto",
               choices=valid_keys + valid_nums)
    if raw.isdigit():
        cmd = _INTERACTIVE_CHOICES[int(raw) - 1][0]
    else:
        cmd = raw.lower()
    print(f"  → {cmd}\n")

    # 1b. Mode glossary — what each amplitude that Sofia will compute means.
    # Skip for corrin-freq (its basis is per-file Gaussian normal modes, not
    # the analytic D4h patterns) and for the auto-detect placeholder (the
    # actual macrocycle isn't known until after Step 2 reads a file).
    if cmd not in ("corrin-freq", "auto"):
        # Map the subcommand to a human-readable macrocycle name for the
        # glossary's tone-setting (corrin gets the C1 caveat, etc.)
        glossary_name = {
            "porphyrin": "porphyrin-class",
            "corphin":   "corphin / F430",
            "corrin":    "corrin",
            "qmmm":      "F430 / corphin (QM/MM)",
        }.get(cmd, "porphyrin-class")
        try:
            import nsd_porphyrin as porph  # nsd/ already on sys.path
            print(porph.format_mode_glossary(glossary_name, indent="  "))
            print()
        except Exception:
            pass  # never let a glossary error block the run

    # 1c. Optional visualization. Independent of the analysis flow — the user
    # can opt in regardless of which macrocycle they picked, then continue to
    # the analysis prompts. If they only wanted the viz, they can decline the
    # path prompt below to bail out.
    if cmd != "corrin-freq":
        _interactive_viz_prompt(cmd)

    # 2. Input path. Allow an empty answer to mean "skip the analysis" (useful
    # when the user only wanted the visualization above). Tab-completion is
    # active throughout — _ask_path wires up readline.
    print()
    print(" Step 2/4 — Path to a single .log file or a directory of files")
    print("            (leave blank to exit without running the analysis)")
    path = _ask_path("  Path", allow_empty=True)
    if path is None:
        print("[no path given — exiting after visualization]")
        return 0
    n_files = 1 if path.is_file() else len(list(path.rglob("*.log"))) + len(list(path.rglob("*.QMRegion.xyz")))
    print(f"  → {path}  ({'directory' if path.is_dir() else 'single file'}, {n_files} input(s))\n")

    # 3. Optional outputs
    print(" Step 3/4 — Output options")
    write_csv = _ask_yes_no("  Write CSV summary?", default=False)
    csv_path = None
    if write_csv:
        default_csv = "sofia_results.csv"
        csv_path = _ask("  CSV filename", default=default_csv)
    write_plots = _ask_yes_no("  Write per-file PNG plots and cross-dataset summary?",
                              default=False)
    plot_dir = None
    if write_plots:
        default_dir = "plots/"
        plot_dir = _ask("  Plot directory", default=default_dir)
    write_report = _ask_yes_no("  Write per-file scientific .txt report?",
                               default=False)
    report_dir = None
    if write_report:
        default_dir = "reports/"
        report_dir = _ask("  Report directory", default=default_dir)
    quiet = _ask_yes_no("  Quiet mode (suppress per-file detailed report)?",
                        default=False)
    print()

    # 4. corrin-freq needs --ref
    ref_path = None
    if cmd == "corrin-freq":
        print(" Step 3b — corrin-freq requires a reference opt+freq log")
        ref_path = str(_ask_path("  Reference opt+freq log"))
        print()

    # 5. Confirmation
    argv = [str(path)]
    if csv_path:
        argv += ["--csv", csv_path]
    if plot_dir:
        argv += ["--plot", plot_dir]
    if report_dir:
        argv += ["--report", report_dir]
    if quiet:
        argv += ["-q"]
    if ref_path:
        argv += ["--ref", ref_path]
    equiv = "sofia.py " + cmd + " " + " ".join(argv)
    print(" Step 4/4 — Confirm and run")
    print(f"  Equivalent non-interactive command:")
    print(f"    {equiv}")
    print()
    if not _ask_yes_no("  Run now?", default=True):
        print("[cancelled]")
        return 0

    print()
    print("=" * 64)

    # Dispatch
    if cmd == "auto":
        detected = _detect_macrocycle(path)
        print(f"[auto-detect] dispatching to: {detected}\n")
        cmd = detected
    return _delegate(cmd, argv)


# -------------------- main --------------------

def main():
    # We use a permissive top-level parser so subcommand-specific flags
    # (passed via argv[2:]) are forwarded as-is to the underlying module.
    if len(sys.argv) < 2:
        # No args → interactive mode
        return interactive_mode()

    if sys.argv[1] in ("-h", "--help", "help"):
        print(__doc__)
        return 0

    cmd = sys.argv[1]
    rest = sys.argv[2:]

    if cmd in ("interactive", "-i", "--interactive"):
        return interactive_mode()

    if cmd == "viz":
        # Forward to the visualizer module (porphyrin or corrin). We support a
        # thin --render flag here that's not in the underlying tool (it
        # post-processes via headless PyMOL); other flags are passed through.
        import argparse as _ap
        ap = _ap.ArgumentParser(prog="sofia.py viz", description=(
            "Generate the minimal-basis visualization (planar reference + 7 "
            "OOP + 6 IP distorted variants + 2 .pml scripts for PyMOL). "
            "Pick --type porphyrin (default), corrin, or corphin."))
        ap.add_argument("--type", choices=("porphyrin", "corrin", "corphin"),
                        default="porphyrin",
                        help="Macrocycle class for the viz bundle "
                             "(default: porphyrin)")
        ap.add_argument("--amp", type=float, default=0.5,
                        help="Distortion amplitude in Å (default: 0.5)")
        ap.add_argument("--out", default=None,
                        help="Output directory "
                             "(default: viz/<type>)")
        ap.add_argument("--log", help="Reference Gaussian .log "
                        "(default: bundled examples/<type>/<reference>.log)")
        ap.add_argument("--render", action="store_true",
                        help="Also produce .png previews via headless PyMOL")
        a = ap.parse_args(rest)
        out_dir = a.out if a.out is not None else f"viz/{a.type}"
        ok = _run_viz(a.amp, out_dir, render_png=a.render, log_path=a.log,
                      macrocycle=a.type)
        return 0 if ok else 1

    if cmd == "auto":
        # auto-detect needs the path argument; everything after gets forwarded
        if not rest or rest[0].startswith("-"):
            print("usage: sofia.py auto <path> [--csv FILE] [--plot DIR] [-q]",
                  file=sys.stderr)
            return 2
        detected = _detect_macrocycle(Path(rest[0]))
        print(f"[auto-detect] dispatching to: sofia.py {detected} {' '.join(rest)}\n")
        return _delegate(detected, rest)

    if cmd in _SUBMODS:
        return _delegate(cmd, rest)

    print(f"sofia.py: unknown subcommand {cmd!r}\n", file=sys.stderr)
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main() or 0)
