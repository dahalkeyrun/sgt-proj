#!/usr/bin/env python3
"""
Collect benchmark + correctness data for the NCut comparison paper.

Sweeps image resolutions on both CPU and GPU backends and writes raw CSV
files into stats/csv/. The plotting script (plot_figures.py) reads these
CSVs and renders the paper figures.

Figures covered by this data:
    Fig 1  timing.csv           end-to-end runtime CPU vs GPU
    Fig 2  stages.csv           per-stage breakdown (graph/eigen/partition)
    Fig 3  speedup.csv          CPU/GPU speedup factor vs size
    Fig 4  graph_build.csv      old loop vs vectorized CPU graph builder
    Fig 5  corr.csv             CPU vs GPU Fiedler vector Pearson correlation
    Fig 6  ncut.csv             NCut score stability across backends
    Fig 7  graph_complexity.csv edges/sparsity scaling
    Fig 9  kway.csv             k-way NCut vs k
    Fig 10 memory.csv           peak memory CPU (RSS) vs GPU (CuPy pool)
"""

import csv
import math
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ncut.backend import Backend, is_gpu_available
from ncut.graph import build_weight_matrix
from ncut.ncut import find_fiedler_vector, find_k_eigenvectors
from ncut.partition import partition_graph, partition_kway

STATS_CSV = REPO_ROOT / "stats" / "csv"

# ── Benchmark configuration ────────────────────────────────────────────────
SIZES = [20, 30, 50, 80, 100, 128, 160]
SIGMA_I, SIGMA_X, RADIUS = 0.1, 4.0, 5
KWAY_SIZE = 100
KWAY_K_RANGE = list(range(2, 17))
RUNS = {
    20: 5, 30: 5, 50: 3, 80: 3, 100: 3, 128: 1, 160: 1,
}


def _synthetic_image(size: int) -> np.ndarray:
    y, x = np.mgrid[0:size, 0:size]
    return ((x + y) / (2 * size - 2)).astype(np.float64)


_REAL_IMAGE_CACHE: dict[int, np.ndarray] = {}


def _benchmark_image(size: int) -> np.ndarray:
    """Real photo (pipeline_sample.png) downsampled to `size` px/side.

    The real image has a well-separated Laplacian spectrum so the CPU (ARPACK)
    and GPU (CuPy) eigensolvers converge to the same eigenvectors, giving
    corr ~ 1.0. The synthetic diagonal-gradient has near-duplicate eigenvalues
    that break ARPACK convergence, so it is only used for the pure graph-build
    timing figure.
    """
    if size not in _REAL_IMAGE_CACHE:
        from ncut.utils import load_image

        _REAL_IMAGE_CACHE[size] = load_image(
            str(REPO_ROOT / "data" / "input" / "pipeline_sample.png"),
            mode="grayscale",
            max_size=size,
        )
    return _REAL_IMAGE_CACHE[size]


def _sync(backend: Backend) -> None:
    if backend == Backend.GPU:
        import cupy as cp

        cp.cuda.Stream.null.synchronize()


def reference_build_graph_structures(
    image, sigma_i, sigma_x, radius,
    color_metric="grayscale", doog_features=None,
):
    """Faithful copy of the OLD nested-loop CPU graph builder."""
    from ncut.color import ColorMetric, build_feature_map

    feature_map = build_feature_map(
        image,
        ColorMetric.GRAYSCALE,
        doog_features=doog_features,
        backend="cpu",
    )

    height, width = feature_map.shape[:2]
    num_pixels = height * width
    sigma_i_sq = sigma_i ** 2
    sigma_x_sq = sigma_x ** 2
    radius_sq = radius ** 2

    values = []
    col_indices = []
    row_pointers = [0]
    degree_vector = np.zeros(num_pixels, dtype=np.float64)

    for y1 in range(height):
        for x1 in range(width):
            i = y1 * width + x1
            y_min = max(0, y1 - radius)
            y_max = min(height, y1 + radius + 1)
            x_min = max(0, x1 - radius)
            x_max = min(width, x1 + radius + 1)
            running_degree_sum = 0.0

            for y2 in range(y_min, y_max):
                for x2 in range(x_min, x_max):
                    j = y2 * width + x2
                    if i == j:
                        continue

                    dist_x_sq = (x1 - x2) ** 2 + (y1 - y2) ** 2
                    if dist_x_sq >= radius_sq:
                        continue

                    feature_i = feature_map[y1, x1]
                    feature_j = feature_map[y2, x2]
                    diff = feature_i - feature_j
                    dist_i_sq = float(np.dot(diff, diff))

                    weight = (
                        math.exp(-dist_i_sq / sigma_i_sq)
                        * math.exp(-dist_x_sq / sigma_x_sq)
                    )
                    values.append(weight)
                    col_indices.append(j)
                    running_degree_sum += weight

            degree_vector[i] = running_degree_sum
            row_pointers.append(len(values))

    return (
        np.array(values, dtype=np.float64),
        np.array(col_indices, dtype=np.int32),
        np.array(row_pointers, dtype=np.int32),
        degree_vector,
    )


def _time_stages(image, backend: Backend) -> dict:
    """Run the pipeline once; return per-stage best times across RUNS."""
    best = {"graph_ms": math.inf, "eigensolver_ms": math.inf,
            "partition_ms": math.inf}
    fiedler = None
    ncut = None
    weight_matrix = None
    degree_vector = None

    size = image.shape[0]
    runs = RUNS.get(size, 1)

    for _ in range(runs):
        t0 = time.perf_counter()
        weight_matrix, degree_vector = build_weight_matrix(
            image, sigma_i=SIGMA_I, sigma_x=SIGMA_X, radius=RADIUS,
            backend=backend,
        )
        _sync(backend)
        graph_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        fiedler = find_fiedler_vector(
            weight_matrix.data,
            weight_matrix.indices,
            weight_matrix.indptr,
            degree_vector,
            backend=backend,
        )
        _sync(backend)
        eigen_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        part = partition_graph(
            weight_matrix, fiedler, threshold_method="ncut", backend=backend,
        )
        _sync(backend)
        part_ms = (time.perf_counter() - t0) * 1000.0
        ncut = part["ncut"]

        best["graph_ms"] = min(best["graph_ms"], graph_ms)
        best["eigensolver_ms"] = min(best["eigensolver_ms"], eigen_ms)
        best["partition_ms"] = min(best["partition_ms"], part_ms)

    best["total_ms"] = best["graph_ms"] + best["eigensolver_ms"] + best["partition_ms"]
    best["ncut"] = ncut
    best["weight_matrix"] = weight_matrix
    best["degree_vector"] = degree_vector
    best["fiedler"] = fiedler
    return best


def _gpu_pool_total_mb() -> float:
    """GPU memory handed out by the CuPy pool (high-water allocation)."""
    import cupy as cp

    pool = cp.get_default_memory_pool()
    return pool.total_bytes() / (1024 * 1024)


def _cpu_rss_mb() -> float:
    """Current resident set size of this process (MB), read from /proc."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except (OSError, IndexError, ValueError):
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _write_csv(name: str, header: list, rows: list) -> None:
    STATS_CSV.mkdir(parents=True, exist_ok=True)
    path = STATS_CSV / name
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"[csv] {path.name} ({len(rows)} rows)")


def main() -> None:
    gpu = is_gpu_available()
    print(f"GPU available: {gpu}")
    if gpu:
        import cupy as cp

        cp.get_default_memory_pool().free_all_blocks()

    # ── Fig 1 / 2 / 3 / 5 / 6 / 7 / 10 sweep ────────────────────────────────
    timing_rows, stage_rows, speedup_rows = [], [], []
    corr_rows, ncut_rows, complexity_rows, memory_rows = [], [], [], []

    for size in SIZES:
        image = _benchmark_image(size)
        nodes = image.size

        # CPU pipeline
        cpu = _time_stages(image, Backend.CPU)
        cpu_peak = _cpu_rss_mb()

        # GPU pipeline
        if gpu:
            gpu_res = _time_stages(image, Backend.GPU)
            gpu_peak = _gpu_pool_total_mb()
        else:
            gpu_res = dict(total_ms=float("nan"), graph_ms=float("nan"),
                           eigensolver_ms=float("nan"), partition_ms=float("nan"))
            gpu_peak = float("nan")

        # Fiedler correlation (needs both backends)
        if gpu:
            c = float(np.corrcoef(cpu["fiedler"], gpu_res["fiedler"])[0, 1])
            corr_rows.append([size, round(abs(c), 6)])
            ncut_rows.append([size, round(float(cpu["ncut"]), 6),
                              round(float(gpu_res["ncut"]), 6)])
        else:
            corr_rows.append([size, float("nan")])
            ncut_rows.append([size, round(float(cpu["ncut"]), 6), float("nan")])

        # Edges / sparsity
        nnz = cpu["weight_matrix"].nnz
        sparsity = 1.0 - nnz / (nodes * nodes)
        complexity_rows.append([size, nodes, nnz, round(sparsity, 6)])

        # Timing / stages / speedup
        timing_rows.append([size, nodes,
                            round(cpu["total_ms"], 3),
                            round(gpu_res["total_ms"], 3)])
        stage_rows.append([size, "cpu", round(cpu["graph_ms"], 3),
                           round(cpu["eigensolver_ms"], 3),
                           round(cpu["partition_ms"], 3)])
        stage_rows.append([size, "gpu", round(gpu_res["graph_ms"], 3),
                           round(gpu_res["eigensolver_ms"], 3),
                           round(gpu_res["partition_ms"], 3)])
        if gpu:
            speedup_rows.append([size,
                                 round(cpu["total_ms"] / gpu_res["total_ms"], 3),
                                 round(cpu["eigensolver_ms"] / gpu_res["eigensolver_ms"], 3),
                                 round(cpu["graph_ms"] / gpu_res["graph_ms"], 3)])
        else:
            speedup_rows.append([size, float("nan"), float("nan"), float("nan")])

        memory_rows.append([size, round(cpu_peak, 3), round(gpu_peak, 3)])

        print(f"size={size:3d} cpu={cpu['total_ms']:9.2f}ms "
              f"gpu={gpu_res['total_ms']:9.2f}ms "
              f"corr={corr_rows[-1][1]}")

    _write_csv("timing.csv", ["size", "nodes", "cpu_total_ms", "gpu_total_ms"], timing_rows)
    _write_csv("stages.csv", ["size", "backend", "graph_ms", "eigensolver_ms", "partition_ms"], stage_rows)
    _write_csv("speedup.csv", ["size", "speedup_total", "speedup_eigen", "speedup_graph"], speedup_rows)
    _write_csv("corr.csv", ["size", "fiedler_pearson"], corr_rows)
    _write_csv("ncut.csv", ["size", "cpu_ncut", "gpu_ncut"], ncut_rows)
    _write_csv("graph_complexity.csv", ["size", "nodes", "edges", "sparsity"], complexity_rows)
    _write_csv("memory.csv", ["size", "cpu_peak_mb", "gpu_peak_mb"], memory_rows)

    # ── Fig 4: graph build old vs vectorized ────────────────────────────────
    graph_rows = []
    for size in SIZES:
        image = _synthetic_image(size)
        ref_best, fast_best = math.inf, math.inf
        runs = RUNS.get(size, 1)
        for _ in range(runs):
            t0 = time.perf_counter()
            reference_build_graph_structures(image, SIGMA_I, SIGMA_X, RADIUS)
            ref_best = min(ref_best, (time.perf_counter() - t0) * 1000.0)

            t0 = time.perf_counter()
            build_weight_matrix(image, sigma_i=SIGMA_I, sigma_x=SIGMA_X,
                                radius=RADIUS, backend=Backend.CPU)
            fast_best = min(fast_best, (time.perf_counter() - t0) * 1000.0)
        graph_rows.append([size, round(ref_best, 3), round(fast_best, 3)])
        print(f"graph_build size={size:3d} ref={ref_best:9.2f}ms "
              f"fast={fast_best:8.2f}ms speedup={ref_best / fast_best:.1f}x")

    _write_csv("graph_build.csv", ["size", "ref_loops_ms", "vectorized_ms"], graph_rows)

    # ── Fig 9: k-way NCut vs k at a fixed size ──────────────────────────────
    image = _synthetic_image(KWAY_SIZE)
    weight_matrix, degree_vector = build_weight_matrix(
        image, sigma_i=SIGMA_I, sigma_x=SIGMA_X, radius=RADIUS,
        backend=Backend.CPU,
    )
    kway_rows = []
    for backend in (Backend.CPU, Backend.GPU):
        if backend == Backend.GPU and not gpu:
            continue
        _, eigenvectors = find_k_eigenvectors(
            weight_matrix.data, weight_matrix.indices, weight_matrix.indptr,
            degree_vector, k=max(KWAY_K_RANGE), backend=backend,
        )
        for k in KWAY_K_RANGE:
            result = partition_kway(
                weight_matrix, eigenvectors[:, :k], k=k, backend=backend,
            )
            kway_rows.append([k, backend.value, round(float(result["ncut"]), 6)])
        print(f"kway {backend.value}: done k=2..{max(KWAY_K_RANGE)}")

    _write_csv("kway.csv", ["k", "backend", "ncut_k"], kway_rows)

    print("\nAll CSVs written to", STATS_CSV)


if __name__ == "__main__":
    main()
