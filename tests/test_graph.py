import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from ncut.graph import build_graph_structures, build_weight_matrix


def assert_raises_value_error(function, *args):
    try:
        function(*args)
    except ValueError:
        return

    raise AssertionError("Expected ValueError was not raised.")


def run_isolated_graph_check():
    mock_image = np.array(
        [
            [0.0, 0.5, 1.0],
            [0.0, 0.5, 1.0],
            [0.0, 0.5, 1.0],
        ],
        dtype=np.float64,
    )

    sigma_i = 1.0
    sigma_x = 2.0
    radius = 2

    values, col_indices, row_pointers, degree_vector = build_graph_structures(
        mock_image,
        sigma_i,
        sigma_x,
        radius,
    )

    num_nodes = len(degree_vector)
    num_edges = len(values)

    print("\n--- Diagnostic Pipeline Verification ---")
    print(f"- Total Graph Vertices (Pixels Mapped)   : {num_nodes} (Expected: 9)")
    print(f"- Total Compressed Sparse Edges Tracked  : {num_edges}")
    print(f"- Row Pointers Offset Array Structure    : {row_pointers}")
    print(f"- Calculated Degree Vector elements      : {degree_vector.round(2)}")

    assert num_nodes == 9, "Error: graph vertex mapping size mismatch."
    assert len(row_pointers) == 10, "Error: CSR row pointers must equal num_pixels + 1."
    assert num_edges == 40, "Error: self-edge removal should leave 40 edges for this 3x3 case."


def run_validation_checks():
    valid_image = np.zeros((3, 3), dtype=np.float64)
    invalid_1d_image = np.zeros((3,), dtype=np.float64)

    print("\n--- Graph Input Validation Verification ---")

    assert_raises_value_error(build_graph_structures, invalid_1d_image, 1.0, 2.0, 2)
    print("- Rejected invalid 1D image shape")

    assert_raises_value_error(build_graph_structures, valid_image, 0.0, 2.0, 2)
    print("- Rejected non-positive sigma_i")

    assert_raises_value_error(build_graph_structures, valid_image, 1.0, 0.0, 2)
    print("- Rejected non-positive sigma_x")

    assert_raises_value_error(build_graph_structures, valid_image, 1.0, 2.0, 0)
    print("- Rejected non-positive radius")


def run_self_edge_check():
    image = np.zeros((3, 3), dtype=np.float64)
    values, col_indices, row_pointers, degree_vector = build_graph_structures(
        image,
        sigma_i=1.0,
        sigma_x=2.0,
        radius=2,
    )

    has_self_edges = [
        i in col_indices[row_pointers[i]:row_pointers[i + 1]]
        for i in range(len(degree_vector))
    ]

    print("\n--- Self-Edge Removal Verification ---")
    print(f"- Self-edge presence by row              : {has_self_edges}")

    assert len(values) == 40, "Error: expected 40 edges after self-edge removal."
    assert not any(has_self_edges), "Error: graph should not contain self-edges."


def run_csr_matrix_check():
    image = np.zeros((3, 3), dtype=np.float64)
    weight_matrix, degree_vector = build_weight_matrix(
        image,
        sigma_i=1.0,
        sigma_x=2.0,
        radius=2,
    )

    print("\n--- SciPy CSR Matrix Verification ---")
    print(f"- Weight matrix shape                    : {weight_matrix.shape}")
    print(f"- Stored non-zero weights                : {weight_matrix.nnz}")
    print(f"- Weight matrix diagonal                 : {weight_matrix.diagonal()}")
    print(f"- Degree vector                          : {degree_vector.round(2)}")

    assert weight_matrix.shape == (9, 9), "Error: CSR matrix shape mismatch."
    assert weight_matrix.nnz == 40, "Error: CSR matrix should store 40 edges."
    assert np.allclose(weight_matrix.diagonal(), 0.0), "Error: CSR diagonal should be zero."
    assert np.allclose(weight_matrix.sum(axis=1).A1, degree_vector), "Error: degree vector must equal row sums."


if __name__ == "__main__":
    run_isolated_graph_check()
    run_validation_checks()
    run_self_edge_check()
    run_csr_matrix_check()
    print("\nSTATUS: GRAPH MODULE VERIFIED")
