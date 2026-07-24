import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from ncut.backend import Backend, is_gpu_available
from ncut.graph import build_graph_structures, build_weight_matrix


def run_backend_parsing_checks():
    assert Backend.from_string("cpu") == Backend.CPU
    assert Backend.from_string("CPU") == Backend.CPU
    assert Backend.from_string("gpu") == Backend.GPU
    assert Backend.from_string(Backend.GPU) == Backend.GPU


def run_cpu_graph_backend_checks():
    image = np.zeros((3, 3), dtype=np.float64)
    values, col_indices, row_pointers, degree_vector = build_graph_structures(
        image,
        sigma_i=1.0,
        sigma_x=2.0,
        radius=2,
        backend="cpu",
    )

    assert len(values) > 0
    assert values.shape == col_indices.shape
    assert len(row_pointers) == image.size + 1
    assert degree_vector.shape == (image.size,)

    weight_matrix, degree_vector2 = build_weight_matrix(
        image,
        sigma_i=1.0,
        sigma_x=2.0,
        radius=2,
        backend=Backend.CPU,
    )

    assert weight_matrix.shape == (image.size, image.size)
    assert np.allclose(degree_vector, degree_vector2)


def run_gpu_availability_check():
    available = is_gpu_available()
    assert isinstance(available, bool)


if __name__ == "__main__":
    run_backend_parsing_checks()
    print("- Backend parsing checks passed")
    run_cpu_graph_backend_checks()
    print("- CPU backend graph checks passed")
    run_gpu_availability_check()
    print("- GPU availability check passed")
    print("STATUS: BACKEND MODULE VERIFIED")
