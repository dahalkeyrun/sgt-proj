import numpy as np
from scipy.sparse import csr_matrix

from ncut.backend import Backend, is_gpu_available
from ncut.cpu.graph import build_graph_structures as cpu_build_graph_structures
from ncut.color import ColorMetric
from ncut.gpu.graph import build_graph_structures as gpu_build_graph_structures


def build_graph_structures(
    image: np.ndarray,
    sigma_i: float,
    sigma_x: float,
    radius: int,
    color_metric: ColorMetric = ColorMetric.GRAYSCALE,
    doog_features: np.ndarray | None = None,
    backend: Backend | str = Backend.CPU,
) -> tuple:
    backend = Backend.from_string(backend)

    if backend == Backend.GPU:
        if not is_gpu_available():
            raise RuntimeError(
                "GPU backend requested but CuPy/CUDA is unavailable."
            )
        return gpu_build_graph_structures(
            image,
            sigma_i,
            sigma_x,
            radius,
            color_metric=color_metric,
            doog_features=doog_features,
        )

    return cpu_build_graph_structures(
        image,
        sigma_i,
        sigma_x,
        radius,
        color_metric=color_metric,
        doog_features=doog_features,
    )


def build_weight_matrix(
    image: np.ndarray,
    sigma_i: float,
    sigma_x: float,
    radius: int,
    color_metric: ColorMetric = ColorMetric.GRAYSCALE,
    doog_features: np.ndarray | None = None,
    backend: Backend | str = Backend.CPU,
) -> tuple:
    values, col_indices, row_pointers, degree_vector = build_graph_structures(
        image,
        sigma_i,
        sigma_x,
        radius,
        color_metric=color_metric,
        doog_features=doog_features,
        backend=backend,
    )

    num_pixels = len(degree_vector)
    weight_matrix = csr_matrix((values, col_indices, row_pointers),
                               shape=(num_pixels, num_pixels),
                               dtype=np.float64)

    return weight_matrix, degree_vector
