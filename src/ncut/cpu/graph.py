import numpy as np
from scipy.sparse import csr_matrix

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
    image: np.ndarray,
    sigma_i: float,
    sigma_x: float,
    radius: int,
    color_metric: ColorMetric = ColorMetric.GRAYSCALE,
    doog_features: np.ndarray | None = None,
) -> tuple:
    validate_graph_parameters(image, sigma_i, sigma_x, radius)

    feature_map = build_feature_map(
        image,
        color_metric,
        doog_features=doog_features,
        backend="cpu",
    )

    height, width = feature_map.shape[:2]
    num_pixels = height * width
    sigma_i_sq = sigma_i ** 2
    sigma_x_sq = sigma_x ** 2

    offsets = _build_offsets(radius)
    row_blocks = []
    col_blocks = []
    value_blocks = []

    for dy, dx in offsets:
        y_start = max(0, -dy)
        y_end = min(height, height - dy)
        x_start = max(0, -dx)
        x_end = min(width, width - dx)

        if y_start >= y_end or x_start >= x_end:
            continue

        rows = np.arange(y_start, y_end, dtype=np.int32) * width
        cols = np.arange(x_start, x_end, dtype=np.int32)
        i_flat = (rows[:, None] + cols[None, :]).ravel()
        j_flat = i_flat + dy * width + dx

        diff = (
            feature_map[y_start:y_end, x_start:x_end]
            - feature_map[y_start + dy:y_end + dy, x_start + dx:x_end + dx]
        )
        dist_i_sq = np.einsum("ijk,ijk->ij", diff, diff).ravel()

        weight = np.exp(
            -(dist_i_sq / sigma_i_sq
              + (dx * dx + dy * dy) / sigma_x_sq)
        )
        row_blocks.append(i_flat)
        col_blocks.append(j_flat)
        value_blocks.append(weight)

    if len(row_blocks) == 0:
        return (
            np.array([], dtype=np.float64),
            np.array([], dtype=np.int32),
            np.array([0] * (num_pixels + 1), dtype=np.int32),
            np.zeros(num_pixels, dtype=np.float64),
        )

    row_indices = np.concatenate(row_blocks)
    col_indices = np.concatenate(col_blocks)
    values = np.concatenate(value_blocks)

    degree_vector = np.bincount(
        row_indices,
        weights=values,
        minlength=num_pixels,
    )

    # Sort into canonical row-major CSR order.
    order = np.lexsort((col_indices, row_indices))
    row_indices = row_indices[order]
    col_indices = col_indices[order]
    values = values[order]

    row_counts = np.bincount(row_indices, minlength=num_pixels)
    row_pointers = np.empty(num_pixels + 1, dtype=np.int32)
    row_pointers[0] = 0
    row_pointers[1:] = np.cumsum(row_counts, dtype=np.int32)

    return (
        values,
        col_indices,
        row_pointers,
        degree_vector.astype(np.float64),
    )


def build_weight_matrix(
    image: np.ndarray,
    sigma_i: float,
    sigma_x: float,
    radius: int,
    color_metric: ColorMetric = ColorMetric.GRAYSCALE,
    doog_features: np.ndarray | None = None,
) -> tuple:
    values, col_indices, row_pointers, degree_vector = build_graph_structures(
        image,
        sigma_i,
        sigma_x,
        radius,
        color_metric=color_metric,
        doog_features=doog_features,
    )

    num_pixels = len(degree_vector)
    weight_matrix = csr_matrix((values, col_indices, row_pointers),
                               shape=(num_pixels, num_pixels),
                               dtype=np.float64)
    return weight_matrix, degree_vector
