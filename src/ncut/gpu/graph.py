from __future__ import annotations

from typing import Any

import numpy as np

from ncut.backend import Backend, get_array_module, to_numpy
from ncut.color import ColorMetric, build_feature_map


def validate_graph_parameters(image, sigma_i, sigma_x, radius):
    if image.ndim not in (2, 3):
        raise ValueError("Image must be either 2D grayscale or 3D RGB.")
    if sigma_i <= 0:
        raise ValueError("sigma_i must be positive.")
    if sigma_x <= 0:
        raise ValueError("sigma_x must be positive.")
    if radius <= 0:
        raise ValueError("radius must be positive.")


def _build_offsets(radius: int):
    offsets = []
    radius_sq = radius * radius
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            if dx * dx + dy * dy >= radius_sq:
                continue
            offsets.append((dy, dx))
    return offsets


def build_graph_structures(
    image: Any,
    sigma_i: float,
    sigma_x: float,
    radius: int,
    color_metric: ColorMetric = ColorMetric.GRAYSCALE,
    doog_features: Any | None = None,
) -> tuple:
    validate_graph_parameters(image, sigma_i, sigma_x, radius)

    backend = Backend.GPU
    xp = get_array_module(backend)

    feature_map = build_feature_map(
        image,
        color_metric,
        doog_features=doog_features,
        backend=backend,
    )

    height, width = feature_map.shape[:2]
    num_pixels = height * width
    sigma_i_sq = sigma_i ** 2
    sigma_x_sq = sigma_x ** 2

    offsets = _build_offsets(radius)
    row_indices = []
    col_indices = []
    values = []

    for dy, dx in offsets:
        y_start = max(0, -dy)
        y_end = min(height, height - dy)
        x_start = max(0, -dx)
        x_end = min(width, width - dx)

        if y_start >= y_end or x_start >= x_end:
            continue

        y_src = xp.arange(y_start, y_end, dtype=xp.int32)
        x_src = xp.arange(x_start, x_end, dtype=xp.int32)
        y_grid, x_grid = xp.meshgrid(y_src, x_src, indexing="ij")

        y_tgt = y_grid + dy
        x_tgt = x_grid + dx

        i_flat = y_grid * width + x_grid
        j_flat = y_tgt * width + x_tgt

        features_i = feature_map[y_grid, x_grid]
        features_j = feature_map[y_tgt, x_tgt]

        diff = features_i - features_j
        dist_i_sq = xp.sum(diff * diff, axis=-1)

        weight = xp.exp(-dist_i_sq / sigma_i_sq) * xp.exp(-(dx * dx + dy * dy) / sigma_x_sq)

        row_indices.append(i_flat.ravel())
        col_indices.append(j_flat.ravel())
        values.append(weight.ravel())

    if len(values) == 0:
        return (
            np.array([], dtype=np.float64),
            np.array([], dtype=np.int32),
            np.array([0] * (num_pixels + 1), dtype=np.int32),
            np.zeros(num_pixels, dtype=np.float64),
        )

    row_indices = xp.concatenate(row_indices)
    col_indices = xp.concatenate(col_indices)
    values = xp.concatenate(values)

    values = values.astype(xp.float64, copy=False)
    row_indices = row_indices.astype(xp.int32, copy=False)
    col_indices = col_indices.astype(xp.int32, copy=False)

    degree_vector = xp.bincount(
        row_indices,
        weights=values,
        minlength=num_pixels,
    )

    # Build CSR arrays directly. Going through cupyx COO -> CSR performs
    # another sparse conversion even though the solver ultimately needs the
    # raw CSR buffers back on the CPU.
    sort_key = row_indices.astype(xp.int64) * num_pixels + col_indices.astype(xp.int64)
    order = xp.argsort(sort_key)
    row_indices = row_indices[order]
    col_indices = col_indices[order]
    values = values[order]

    row_counts = xp.bincount(row_indices, minlength=num_pixels)
    row_pointers = xp.empty(num_pixels + 1, dtype=xp.int32)
    row_pointers[0] = 0
    row_pointers[1:] = xp.cumsum(row_counts, dtype=xp.int32)

    return (
        to_numpy(values),
        to_numpy(col_indices),
        to_numpy(row_pointers),
        to_numpy(degree_vector.astype(xp.float64)),
    )
