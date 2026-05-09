"""Plotting helpers for NSD output (porphyrin and corrin).

All functions write PNG to disk and close the figure. Uses the non-interactive
'Agg' matplotlib backend so the scripts can run on headless systems.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -------------------- per-file plots --------------------

def plot_dz_profile(canonical_xy, dz_obs, dz_recon, class_labels,
                    title, out_path):
    """Linear plot of Delta-z vs atom index, observed + reconstruction.

    canonical_xy: (N, 2) -- only used for legend annotation, not plotted
    dz_obs, dz_recon: (N,) signed displacements in Å
    class_labels: length-N list of class strings (e.g. ['N','N','N','N','Ca',...])
    """
    n = len(dz_obs)
    fig, ax = plt.subplots(figsize=(9, 3.5))
    x = np.arange(n)

    ax.plot(x, dz_obs, "o-", color="C0", label="observed",
            linewidth=1.5, markersize=5)
    ax.plot(x, dz_recon, "s--", color="C3", label="minimal-basis reconstruction",
            linewidth=1.0, markersize=4, alpha=0.8)
    ax.axhline(0, color="gray", linewidth=0.5)

    # Class boundaries
    boundaries = []
    cur = class_labels[0]
    for i in range(1, n):
        if class_labels[i] != cur:
            boundaries.append(i - 0.5)
            cur = class_labels[i]
    for b in boundaries:
        ax.axvline(b, color="lightgray", linewidth=0.7, linestyle=":")

    # Class-name labels at top
    starts = [0] + [int(b + 0.5) for b in boundaries] + [n]
    ymax = max(abs(dz_obs.max()), abs(dz_obs.min())) * 1.25 if n else 1.0
    if ymax < 1e-4:
        ymax = 1e-4
    ax.set_ylim(-ymax, ymax)
    for i in range(len(starts) - 1):
        mid = (starts[i] + starts[i + 1]) / 2 - 0.5
        ax.text(mid, 0.92 * ymax, class_labels[starts[i]],
                ha="center", fontsize=10, fontweight="bold", color="gray")

    ax.set_xlabel("atom index in canonical macrocycle order")
    ax.set_ylabel(r"$\Delta z$ (Å)")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.85)
    ax.set_xlim(-0.5, n - 0.5)

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_amplitudes(mode_labels, amplitudes, title, out_path,
                    annotate_threshold=1e-4):
    """Signed-amplitude bar chart for the minimal-basis modes."""
    n = len(mode_labels)
    fig, ax = plt.subplots(figsize=(max(6, 0.7 * n), 3.5))
    x = np.arange(n)
    colors = ["C0" if a >= 0 else "C3" for a in amplitudes]
    bars = ax.bar(x, amplitudes, color=colors, edgecolor="black", linewidth=0.4)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(mode_labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("amplitude (Å)")
    ax.set_title(title, fontsize=11)

    ymax = max(abs(min(amplitudes, default=0)),
               abs(max(amplitudes, default=0))) * 1.20 or 1e-4
    ax.set_ylim(-ymax, ymax)

    for i, a in enumerate(amplitudes):
        if abs(a) >= annotate_threshold:
            ax.text(i, a + (0.04 * ymax * (1 if a >= 0 else -1)),
                    f"{a:+.4f}", ha="center",
                    va="bottom" if a >= 0 else "top",
                    fontsize=7, color="black")

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_per_file_combined(canonical_xy, dz_obs, dz_recon, class_labels,
                           mode_labels, amplitudes,
                           title, out_path):
    """Single PNG combining the Δz profile and the amplitude bar chart."""
    n_atoms = len(dz_obs)
    n_modes = len(mode_labels)

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9, 6.5),
                                         gridspec_kw={"height_ratios": [3, 3]})

    # ---- top: Δz profile ----
    x = np.arange(n_atoms)
    ax_top.plot(x, dz_obs, "o-", color="C0", label="observed",
                linewidth=1.5, markersize=5)
    ax_top.plot(x, dz_recon, "s--", color="C3", label="minimal-basis recon.",
                linewidth=1.0, markersize=4, alpha=0.8)
    ax_top.axhline(0, color="gray", linewidth=0.5)
    boundaries = []
    cur = class_labels[0]
    for i in range(1, n_atoms):
        if class_labels[i] != cur:
            boundaries.append(i - 0.5)
            cur = class_labels[i]
    for b in boundaries:
        ax_top.axvline(b, color="lightgray", linewidth=0.7, linestyle=":")
    starts = [0] + [int(b + 0.5) for b in boundaries] + [n_atoms]
    ymax = max(abs(dz_obs.max()), abs(dz_obs.min())) * 1.30 if n_atoms else 1.0
    if ymax < 1e-4:
        ymax = 1e-4
    ax_top.set_ylim(-ymax, ymax)
    for i in range(len(starts) - 1):
        mid = (starts[i] + starts[i + 1]) / 2 - 0.5
        ax_top.text(mid, 0.92 * ymax, class_labels[starts[i]],
                    ha="center", fontsize=10, fontweight="bold", color="gray")
    ax_top.set_xlabel("atom index in canonical macrocycle order")
    ax_top.set_ylabel(r"$\Delta z$ (Å)")
    ax_top.legend(loc="lower right", fontsize=9, framealpha=0.85)
    ax_top.set_xlim(-0.5, n_atoms - 0.5)
    ax_top.set_title(title, fontsize=11)

    # ---- bottom: amplitudes ----
    xb = np.arange(n_modes)
    colors = ["C0" if a >= 0 else "C3" for a in amplitudes]
    ax_bot.bar(xb, amplitudes, color=colors, edgecolor="black", linewidth=0.4)
    ax_bot.axhline(0, color="black", linewidth=0.5)
    ax_bot.set_xticks(xb)
    ax_bot.set_xticklabels(mode_labels, rotation=45, ha="right", fontsize=9)
    ax_bot.set_ylabel("amplitude (Å)")
    yb = max(abs(min(amplitudes, default=0)),
             abs(max(amplitudes, default=0))) * 1.20 or 1e-4
    ax_bot.set_ylim(-yb, yb)
    for i, a in enumerate(amplitudes):
        if abs(a) >= 1e-4:
            ax_bot.text(i, a + 0.04 * yb * (1 if a >= 0 else -1),
                        f"{a:+.3f}", ha="center",
                        va="bottom" if a >= 0 else "top", fontsize=7)

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# -------------------- summary plot across files --------------------

def plot_mode_scatter(file_labels, x_amps, y_amps, x_label, y_label,
                      title, out_path, metal_per_file=None,
                      x_planar=0.05, y_planar=0.05):
    """2D scatter of two mode amplitudes across the dataset.

    Each point is a single file. Points are coloured by metal (if
    metal_per_file given). Lines at x=0, y=0 visualise sign quadrants;
    optional dashed boxes at ±planar_threshold mark the chemical-significance
    cutoff. Each point is labelled with file_labels.
    """
    fig, ax = plt.subplots(figsize=(7.0, 6.5))

    # Distinct colour per metal
    if metal_per_file is None:
        metal_per_file = [None] * len(file_labels)
    metals = list(dict.fromkeys(metal_per_file))
    cmap = {m: f"C{i}" for i, m in enumerate(metals)}
    seen = set()
    for label, x, y, metal in zip(file_labels, x_amps, y_amps, metal_per_file):
        c = cmap.get(metal, "C0")
        plot_label = metal if metal not in seen else None
        seen.add(metal)
        ax.scatter(x, y, color=c, s=80, edgecolor="black", linewidth=0.6,
                   zorder=3, label=plot_label)
        ax.annotate(label, xy=(x, y), xytext=(6, 6), textcoords="offset points",
                    fontsize=8, color="black", zorder=4)

    # Reference lines
    ax.axhline(0, color="gray", linewidth=0.7, zorder=1)
    ax.axvline(0, color="gray", linewidth=0.7, zorder=1)

    # Planarity / chemical-significance band (dashed grey rectangle)
    if x_planar is not None and y_planar is not None:
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle((-x_planar, -y_planar),
                               2 * x_planar, 2 * y_planar,
                               edgecolor="lightgray", facecolor="none",
                               linestyle="--", linewidth=0.8, zorder=1))
        ax.text(x_planar * 1.05, -y_planar * 0.95,
                f"±{x_planar:.2f} Å\nplanarity band",
                fontsize=7, color="gray", va="top")

    ax.set_xlabel(f"{x_label} amplitude (Å)")
    ax.set_ylabel(f"{y_label} amplitude (Å)")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="best", fontsize=9, framealpha=0.85, title="metal")
    ax.set_aspect("equal", adjustable="datalim")

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_summary(file_labels, amplitudes_matrix, mode_labels, title, out_path):
    """Grid of bar charts: one panel per mode, bars across files.

    amplitudes_matrix: (n_files, n_modes)
    """
    n_files, n_modes = amplitudes_matrix.shape
    ncols = 3
    nrows = (n_modes + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 2.2 * nrows),
                             sharex=False)
    axes = np.atleast_2d(axes)

    x = np.arange(n_files)
    for j, mode in enumerate(mode_labels):
        r, c = j // ncols, j % ncols
        ax = axes[r, c]
        amps = amplitudes_matrix[:, j]
        colors = ["C0" if a >= 0 else "C3" for a in amps]
        ax.bar(x, amps, color=colors, edgecolor="black", linewidth=0.3)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(mode, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(file_labels, rotation=70, ha="right", fontsize=7)
        # symmetric y-axis with small padding
        m = max(abs(amps).max() if amps.size else 0, 1e-4)
        ax.set_ylim(-1.2 * m, 1.2 * m)

    # Hide unused panels
    for j in range(n_modes, nrows * ncols):
        r, c = j // ncols, j % ncols
        axes[r, c].axis("off")

    fig.suptitle(title, fontsize=12, y=1.0)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
