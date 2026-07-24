import math
import numpy as np
from scipy.sparse import csr_matrix
from ncut.color import ColorMetric, build_feature_map, intensity_distance_sq


def validate_graph_parameters(image, sigma_i, sigma_x, radius):
    if image.ndim not in (2, 3):
        raise ValueError("Image must be either 2D grayscale or 3D RGB.")
    if sigma_i <= 0:
        raise ValueError("sigma_i must be positive.")
    if sigma_x <= 0:
        raise ValueError("sigma_x must be positive.")
    if radius <= 0:
        raise ValueError("radius must be positive.")


def calculate_spatial_distance_sq(x1, y1, x2, y2):
    return (x1 - x2) ** 2 + (y1 - y2) ** 2


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

    height, width = image.shape[:2]
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

                    dist_x_sq = calculate_spatial_distance_sq(x1, y1, x2, y2)
                    if dist_x_sq >= radius_sq:
                        continue

                    feature_i = feature_map[y1, x1]
                    feature_j = feature_map[y2, x2]
                    dist_i_sq = intensity_distance_sq(feature_i, feature_j, color_metric)

                    weight = math.exp(-dist_i_sq / sigma_i_sq) * math.exp(-dist_x_sq / sigma_x_sq)
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
