#!/usr/bin/env python3
"""Benchmark the accelerated (vectorized) CPU graph construction.

Measures build_graph_structures() against a faithful reference copy of the
old 4-nested-loop implementation (same math.exp / list.append body) and
reports the speedup factor.
"""

import math
import os
import sys
import time
from statistics import mean

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from ncut.cpu.graph import build_graph_structures as fast_build_graph_structures
from ncut.color import ColorMetric, build_feature_map


def reference_build_graph_structures(
    image, sigma_i, sigma_x, radius, color_metric=ColorMetric.GRAYSCALE, doog_features=None
):
    """Reference (old) implementation: 4 nested Python loops over pixels."""
    feature_map = build_feature_map(
        image, color_metric, doog_features=doog_features, backend="cpu"
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


def _synthetic_image(size):
    y, x = np.mgrid[0:size, 0:size]
    return ((x + y) / (2 * size - 2)).astype(np.float64)


def _check_equivalent(image, sigma_i, sigma_x, radius):
    reference = reference_build_graph_structures(image, sigma_i, sigma_x, radius)
    fast = fast_build_graph_structures(image, sigma_i, sigma_x, radius)

    ref_values, ref_cols, ref_rows, ref_degree = reference
    fast_values, fast_cols, fast_rows, fast_degree = fast

    assert ref_values.shape == fast_values.shape, "edge count mismatch"
    assert ref_degree.shape == fast_degree.shape, "degree vector shape mismatch"

    ref_sorted = np.lexsort((ref_cols, np.repeat(np.arange(len(ref_rows) - 1), np.diff(ref_rows))))
    fast_sorted = np.lexsort((fast_cols, np.repeat(np.arange(len(fast_rows) - 1), np.diff(fast_rows))))

    np.testing.assert_allclose(ref_values[ref_sorted], fast_values[fast_sorted], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(ref_degree, fast_degree, rtol=1e-12, atol=1e-12)
    print(f"  equivalence check passed (edges={len(ref_values)})")


def _time(fn, *args, runs=5):
    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        fn(*args)
        timings.append((time.perf_counter() - start) * 1000.0)
    return min(timings), mean(timings)


def _check_end_to_end(image, sigma_i, sigma_x, radius):
    """Run the full pipeline on the vectorized graph to confirm it still works."""
    from ncut.graph import build_weight_matrix
    from ncut.ncut import find_fiedler_vector
    from ncut.partition import partition_graph

    weight_matrix, degree_vector = build_weight_matrix(
        image, sigma_i=sigma_i, sigma_x=sigma_x, radius=radius
    )
    fiedler = find_fiedler_vector(
        weight_matrix.data,
        weight_matrix.indices,
        weight_matrix.indptr,
        degree_vector,
    )
    result = partition_graph(weight_matrix, fiedler, threshold_method="ncut")
    assert result["mask"].shape == (image.size,)
    assert np.isfinite(result["ncut"])
    return result


def main():
    sizes = [20, 30, 50, 100]
    sigma_i, sigma_x, radius = 0.1, 4.0, 5
    runs = 5

    print("Acceleration benchmark: reference (loops) vs vectorized CPU graph builder")
    print("=" * 72)

    for size in sizes:
        image = _synthetic_image(size)
        _check_equivalent(image, sigma_i, sigma_x, radius)

        ref_best, ref_mean = _time(reference_build_graph_structures, image, sigma_i, sigma_x, radius, runs=runs)
        fast_best, fast_mean = _time(fast_build_graph_structures, image, sigma_i, sigma_x, radius, runs=runs)

        speedup_best = ref_best / fast_best if fast_best > 0 else float("inf")
        speedup_mean = ref_mean / fast_mean if fast_mean > 0 else float("inf")

        print(f"image={size}x{size} nodes={size * size}")
        print(f"  reference (best/mean): {ref_best:9.2f}ms / {ref_mean:9.2f}ms")
        print(f"  vectorized (best/mean): {fast_best:7.2f}ms / {fast_mean:7.2f}ms")
        print(f"  speedup (best/mean): {speedup_best:8.1f}x / {speedup_mean:8.1f}x")

        print("-" * 72)

    print()
    print("End-to-end sanity check on the vectorized graph builder:")
    for size in (20, 50):
        image = _synthetic_image(size)
        result = _check_end_to_end(image, sigma_i, sigma_x, radius)
        print(f"  {size}x{size}: NCut={result['ncut']:.4f} threshold={result['threshold']:.4f} OK")


if __name__ == "__main__":
    main()
