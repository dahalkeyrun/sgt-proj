#!/usr/bin/env python3
"""
Render the NCut comparison-paper figures from the CSVs in stats/csv/.

Each figure is saved as high-DPI PNG + SVG into stats/figures/.
Run collect_data.py first to produce the CSVs.

    python stat_gen_scripts/plot_figures.py [--png-dpi 300] [--figs 1..10]

Figures:
    fig01_end_to_end_runtime   CPU vs GPU total time vs image size
    fig02_stage_breakdown      graph/eigen/partition stacked bars CPU vs GPU
    fig03_speedup              CPU/GPU speedup factor vs image size
    fig04_graph_build          old loops vs vectorized CPU graph builder
    fig05_fiedler_correlation  |Pearson| CPU vs GPU Fiedler vectors
    fig06_ncut_stability       NCut score CPU vs GPU across sizes
    fig07_graph_complexity     edges vs nodes, sparsity vs nodes
    fig08_segmentation_panel   qualitative output on a real image
    fig09_kway                 k-way NCut vs number of clusters k
    fig10_memory               peak memory CPU vs GPU vs image size
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

CSV_DIR = REPO_ROOT / "stats" / "csv"
FIG_DIR = REPO_ROOT / "stats" / "figures"

PAPER_WIDTH_IN = 6.8          # single-column IEEE-ish width
GOLDEN = 1.618

CPU_COLOR = "#1f77b4"
GPU_COLOR = "#d62728"
REF_COLOR = "#7f7f7f"
FAST_COLOR = "#2ca02c"
GRID_ALPHA = 0.3

MARKERS = {"cpu": "o", "gpu": "s"}


# ── CSV readers ─────────────────────────────────────────────────────────────

def _read_csv(name: str) -> list[dict]:
    path = CSV_DIR / name
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _f(rows, key) -> np.ndarray:
    return np.asarray([float(r[key]) for r in rows])


def _save(fig, name: str, png_dpi: int) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=png_dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {name}.png / .svg")


def _style_axis(ax, xlabel, ylabel, title=None, logx=False, logy=False):
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, which="both", alpha=GRID_ALPHA, linewidth=0.5)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.tick_params(labelsize=9)


# ── Fig 1: end-to-end runtime ──────────────────────────────────────────────

def fig01_end_to_end_runtime(png_dpi):
    rows = _read_csv("timing.csv")
    size = _f(rows, "size")
    cpu = _f(rows, "cpu_total_ms")
    gpu = _f(rows, "gpu_total_ms")

    fig, ax = plt.subplots(figsize=(PAPER_WIDTH_IN, PAPER_WIDTH_IN / GOLDEN))
    ax.plot(size, cpu, marker=MARKERS["cpu"], color=CPU_COLOR,
            label="CPU (SciPy ARPACK)", linewidth=1.6)
    ax.plot(size, gpu, marker=MARKERS["gpu"], color=GPU_COLOR,
            label="GPU (CuPy / CUDA)", linewidth=1.6)
    _style_axis(ax, "Image resolution (pixels/side)",
                "End-to-end time (ms)", title="Fig. 1 — End-to-end runtime",
                logx=True, logy=True)
    ax.legend(fontsize=10)
    _save(fig, "fig01_end_to_end_runtime", png_dpi)


# ── Fig 2: stage breakdown ─────────────────────────────────────────────────

def fig02_stage_breakdown(png_dpi):
    rows = _read_csv("stages.csv")
    sizes = sorted({int(r["size"]) for r in rows})
    stages = ["graph_ms", "eigensolver_ms", "partition_ms"]
    labels = ["Graph build", "Eigensolver", "Partition"]
    colors = ["#4c72b0", "#dd8452", "#55a868"]
    backend_colors = {"cpu": "#4878d0", "gpu": "#d65f5f"}

    fig, axes = plt.subplots(
        1, 2, figsize=(PAPER_WIDTH_IN, PAPER_WIDTH_IN / GOLDEN / 2),
        sharey=True,
    )
    for ax, size in zip(axes, sizes):
        cpu_rows = [r for r in rows if int(r["size"]) == size and r["backend"] == "cpu"]
        gpu_rows = [r for r in rows if int(r["size"]) == size and r["backend"] == "gpu"]
        if not cpu_rows or not gpu_rows:
            continue
        cpu_vals = [float(cpu_rows[0][s]) for s in stages]
        gpu_vals = [float(gpu_rows[0][s]) for s in stages]

        bottom_cpu, bottom_gpu = 0.0, 0.0
        for i, (stage, label) in enumerate(zip(stages, labels)):
            ax.bar(0, cpu_vals[i], bottom=bottom_cpu, width=0.5,
                   color=colors[i], label=label if size == sizes[0] else None)
            ax.bar(1, gpu_vals[i], bottom=bottom_gpu, width=0.5,
                   color=colors[i])
            bottom_cpu += cpu_vals[i]
            bottom_gpu += gpu_vals[i]

        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"CPU\n{sum(cpu_vals):.0f} ms",
                            f"GPU\n{sum(gpu_vals):.0f} ms"])
        ax.set_title(f"Resolution {size}×{size}")
        ax.grid(True, which="major", axis="y", alpha=GRID_ALPHA)
        ax.tick_params(labelsize=9)

    axes[0].set_ylabel("Stage time (ms)", fontsize=11)
    axes[0].legend(fontsize=9)
    fig.suptitle("Fig. 2 — Per-stage time breakdown (stacked)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, "fig02_stage_breakdown", png_dpi)


# ── Fig 3: speedup ─────────────────────────────────────────────────────────

def fig03_speedup(png_dpi):
    rows = _read_csv("speedup.csv")
    size = _f(rows, "size")
    total = _f(rows, "speedup_total")

    fig, ax = plt.subplots(figsize=(PAPER_WIDTH_IN, PAPER_WIDTH_IN / GOLDEN))
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0,
               label="parity (1×)")
    ax.plot(size, total, marker="D", color="#9467bd", linewidth=1.8,
            label="CPU / GPU speedup")
    _style_axis(ax, "Image resolution (pixels/side)",
                "Speedup factor (CPU time ÷ GPU time)",
                title="Fig. 3 — GPU speedup vs image size", logx=True)
    ax.legend(fontsize=10)
    _save(fig, "fig03_speedup", png_dpi)


# ── Fig 4: graph build ─────────────────────────────────────────────────────

def fig04_graph_build(png_dpi):
    rows = _read_csv("graph_build.csv")
    size = _f(rows, "size")
    ref = _f(rows, "ref_loops_ms")
    fast = _f(rows, "vectorized_ms")
    speedup = ref / fast

    fig, ax = plt.subplots(figsize=(PAPER_WIDTH_IN, PAPER_WIDTH_IN / GOLDEN))
    ax.plot(size, ref, marker=MARKERS["cpu"], color=REF_COLOR, linewidth=1.6,
            label="Nested Python loops (original)")
    ax.plot(size, fast, marker=MARKERS["gpu"], color=FAST_COLOR, linewidth=1.6,
            label="Vectorized (NumPy offset-grid)")
    _style_axis(ax, "Image resolution (pixels/side)",
                "Graph-construction time (ms)",
                title="Fig. 4 — CPU graph builder: before vs after vectorization",
                logx=True, logy=True)
    ax.text(0.02, 0.06,
            f"up to {speedup.max():.0f}× faster",
            transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="gray"))
    ax.legend(fontsize=10)
    _save(fig, "fig04_graph_build", png_dpi)


# ── Fig 5: Fiedler correlation ─────────────────────────────────────────────

def fig05_fiedler_correlation(png_dpi):
    rows = _read_csv("corr.csv")
    size = _f(rows, "size")
    corr = _f(rows, "fiedler_pearson")

    fig, ax = plt.subplots(figsize=(PAPER_WIDTH_IN, PAPER_WIDTH_IN / GOLDEN))
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0)
    ax.plot(size, corr, marker="o", color=FAST_COLOR, linewidth=1.8,
            label="|Pearson correlation|")
    ax.fill_between(size, corr, 1.0, color=FAST_COLOR, alpha=0.15)
    ax.set_ylim(0.995, 1.0005)
    _style_axis(ax, "Image resolution (pixels/side)",
                "|Pearson r| (CPU vs GPU Fiedler)",
                title="Fig. 5 — CPU–GPU numerical agreement")
    ax.legend(fontsize=10)
    _save(fig, "fig05_fiedler_correlation", png_dpi)


# ── Fig 6: NCut stability ──────────────────────────────────────────────────

def fig06_ncut_stability(png_dpi):
    rows = _read_csv("ncut.csv")
    size = _f(rows, "size")
    cpu = _f(rows, "cpu_ncut")
    gpu = _f(rows, "gpu_ncut")

    fig, ax = plt.subplots(figsize=(PAPER_WIDTH_IN, PAPER_WIDTH_IN / GOLDEN))
    ax.plot(size, cpu, marker=MARKERS["cpu"], color=CPU_COLOR, linewidth=1.6,
            label="CPU")
    ax.plot(size, gpu, marker=MARKERS["gpu"], color=GPU_COLOR, linewidth=1.6,
            label="GPU")
    _style_axis(ax, "Image resolution (pixels/side)",
                "NCut score (bipartition)",
                title="Fig. 6 — Segmentation quality across backends",
                logy=True)
    ax.legend(fontsize=10)
    _save(fig, "fig06_ncut_stability", png_dpi)


# ── Fig 7: graph complexity ────────────────────────────────────────────────

def fig07_graph_complexity(png_dpi):
    rows = _read_csv("graph_complexity.csv")
    nodes = _f(rows, "nodes")
    edges = _f(rows, "edges")
    sparsity = _f(rows, "sparsity")

    fig, ax1 = plt.subplots(figsize=(PAPER_WIDTH_IN, PAPER_WIDTH_IN / GOLDEN))
    ax1.plot(nodes, edges, marker="o", color=CPU_COLOR, linewidth=1.6,
             label="Stored edges (nnz)")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Graph nodes (# pixels)", fontsize=11)
    ax1.set_ylabel("Stored edges (nnz)", fontsize=11, color=CPU_COLOR)
    ax1.tick_params(axis="y", labelcolor=CPU_COLOR, labelsize=9)
    ax1.tick_params(axis="x", labelsize=9)
    ax1.grid(True, which="both", alpha=GRID_ALPHA, linewidth=0.5)

    ax2 = ax1.twinx()
    ax2.plot(nodes, sparsity, marker="s", color=GPU_COLOR, linewidth=1.4,
             linestyle="--", label="Sparsity")
    ax2.set_ylabel("Sparsity (1 − nnz / n²)", fontsize=11, color=GPU_COLOR)
    ax2.set_ylim(0.8, 1.0)
    ax2.tick_params(axis="y", labelcolor=GPU_COLOR, labelsize=9)

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], fontsize=10, loc="center right")
    ax1.set_title("Fig. 7 — Graph complexity scaling", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, "fig07_graph_complexity", png_dpi)


# ── Fig 8: qualitative segmentation panel ──────────────────────────────────

def fig08_segmentation_panel(png_dpi):
    from ncut.utils import load_image
    from ncut.graph import build_weight_matrix
    from ncut.ncut import find_fiedler_vector, find_k_eigenvectors
    from ncut.partition import partition_graph, partition_kway

    img = load_image(str(REPO_ROOT / "data" / "input" / "pipeline_sample.png"),
                     mode="grayscale", max_size=128)
    height, width = img.shape

    w_cpu, d_cpu = build_weight_matrix(
        img, sigma_i=0.1, sigma_x=4.0, radius=5, backend="cpu")
    w_gpu, d_gpu = build_weight_matrix(
        img, sigma_i=0.1, sigma_x=4.0, radius=5, backend="gpu")

    # Binary partition from each backend (matches Fig 5/6).
    fv_cpu = find_fiedler_vector(w_cpu.data, w_cpu.indices, w_cpu.indptr,
                                 d_cpu, backend="cpu")
    fv_gpu = find_fiedler_vector(w_gpu.data, w_gpu.indices, w_gpu.indptr,
                                 d_gpu, backend="gpu")
    part_cpu = partition_graph(w_cpu, fv_cpu, threshold_method="ncut",
                               backend="cpu")
    part_gpu = partition_graph(w_gpu, fv_gpu, threshold_method="ncut",
                               backend="gpu")

    # K-way labels (GPU).
    k = 4
    _, eigvecs = find_k_eigenvectors(w_gpu.data, w_gpu.indices, w_gpu.indptr,
                                     d_gpu, k=k, backend="gpu")
    kway = partition_kway(w_gpu, eigvecs, k=k, backend="gpu")

    def grid(v):
        return np.asarray(v, dtype=np.float64).reshape(height, width)

    panels = [
        ("Original", img, "gray", None),
        ("CPU partition", grid(part_cpu["mask"]), "gray", None),
        ("GPU partition", grid(part_gpu["mask"]), "gray", None),
        ("K-way (k=4, GPU)", grid(kway["labels"]), "viridis", None),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(PAPER_WIDTH_IN, PAPER_WIDTH_IN))
    for ax, (title, data, cmap, _) in zip(axes.ravel(), panels):
        ax.imshow(data, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Fig. 8 — Qualitative segmentation comparison (128×128)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, "fig08_segmentation_panel", png_dpi)


# ── Fig 9: k-way NCut vs k ─────────────────────────────────────────────────

def fig09_kway(png_dpi):
    rows = _read_csv("kway.csv")
    cpu = np.array([float(r["ncut_k"]) for r in rows if r["backend"] == "cpu"])
    gpu = np.array([float(r["ncut_k"]) for r in rows if r["backend"] == "gpu"])
    k_cpu = np.array([float(r["k"]) for r in rows if r["backend"] == "cpu"])

    fig, ax = plt.subplots(figsize=(PAPER_WIDTH_IN, PAPER_WIDTH_IN / GOLDEN))
    ax.plot(k_cpu, cpu, marker=MARKERS["cpu"], color=CPU_COLOR, linewidth=1.6,
            label="CPU")
    ax.plot(k_cpu, gpu, marker=MARKERS["gpu"], color=GPU_COLOR, linewidth=1.6,
            label="GPU")
    _style_axis(ax, "Number of clusters k",
                "K-way NCut cost",
                title="Fig. 9 — K-way NCut vs number of clusters (100×100 image)")
    ax.legend(fontsize=10)
    _save(fig, "fig09_kway", png_dpi)


# ── Fig 10: memory ─────────────────────────────────────────────────────────

def fig10_memory(png_dpi):
    rows = _read_csv("memory.csv")
    size = _f(rows, "size")
    cpu = _f(rows, "cpu_peak_mb")
    gpu = _f(rows, "gpu_peak_mb")

    fig, ax = plt.subplots(figsize=(PAPER_WIDTH_IN, PAPER_WIDTH_IN / GOLDEN))
    ax.plot(size, cpu, marker=MARKERS["cpu"], color=CPU_COLOR, linewidth=1.6,
            label="CPU (process RSS)")
    ax.plot(size, gpu, marker=MARKERS["gpu"], color=GPU_COLOR, linewidth=1.6,
            label="GPU (CuPy pool)")
    _style_axis(ax, "Image resolution (pixels/side)",
                "Peak memory (MB)",
                title="Fig. 10 — Peak memory footprint", logy=True)
    ax.legend(fontsize=10)
    _save(fig, "fig10_memory", png_dpi)


# ── registry ───────────────────────────────────────────────────────────────

FIGURES = {
    1: fig01_end_to_end_runtime,
    2: fig02_stage_breakdown,
    3: fig03_speedup,
    4: fig04_graph_build,
    5: fig05_fiedler_correlation,
    6: fig06_ncut_stability,
    7: fig07_graph_complexity,
    8: fig08_segmentation_panel,
    9: fig09_kway,
    10: fig10_memory,
}


def main():
    parser = argparse.ArgumentParser(description="Render NCut paper figures.")
    parser.add_argument("--png-dpi", type=int, default=300)
    parser.add_argument("--figs", type=str, default="all",
                        help="comma-separated figure numbers, e.g. 1,2,3")
    args = parser.parse_args()

    if args.figs == "all":
        selected = list(FIGURES)
    else:
        selected = [int(x) for x in args.figs.split(",") if x.strip()]

    for num in selected:
        if num not in FIGURES:
            print(f"[skip] unknown figure {num}")
            continue
        FIGURES[num](args.png_dpi)

    print(f"\nFigures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
