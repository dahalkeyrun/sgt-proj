import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from ncut.graph import build_graph_structures
from ncut.ncut import (
    d_inverse_w_product,
    d_weighted_norm,
    find_fiedler_vector,
)


def assert_raises_value_error(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except ValueError:
        return

    raise AssertionError("Expected ValueError was not raised.")


def build_mock_split_graph():
    mock_image = np.array(
        [
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )

    return build_graph_structures(
        mock_image,
        sigma_i=0.5,
        sigma_x=2.0,
        radius=2,
    )


def run_isolated_ncut_check():
    values, col_indices, row_pointers, d = build_mock_split_graph()

    fiedler = find_fiedler_vector(
        values,
        col_indices,
        row_pointers,
        d,
        max_iter=500,
        tol=1e-6,
    )

    print("\n--- Solver Diagnostic Verification ---")
    print(f"- Total Partition Elements Calculated : {len(fiedler)} (Expected: 16)")

    reshaped_fiedler = fiedler.reshape((4, 4))
    print("\nReshaped Fiedler Vector Grid Layout (Rounded to 3 decimals):")
    print(np.round(reshaped_fiedler, 3))

    left_sign = np.sign(reshaped_fiedler[0, 0])
    right_sign = np.sign(reshaped_fiedler[0, 2])

    print(f"\nSign Alignment: Left Side = {left_sign} | Right Side = {right_sign}")

    assert len(fiedler) == 16, "Error: Fiedler vector length mismatch."
    assert np.all(np.isfinite(fiedler)), "Error: Fiedler vector contains NaN or infinity."
    assert left_sign != right_sign, "Error: Fiedler vector failed to find the image split boundary."


def run_d_weighted_normalization_check():
    values, col_indices, row_pointers, d = build_mock_split_graph()
    fiedler = find_fiedler_vector(values, col_indices, row_pointers, d)

    norm = d_weighted_norm(fiedler, d)

    print("\n--- D-Weighted Normalization Verification ---")
    print(f"- D-weighted Fiedler norm              : {norm:.6f}")

    assert np.isclose(norm, 1.0), "Error: Fiedler vector should have D-weighted norm 1."


def run_zero_degree_check():
    values = np.array([], dtype=np.float64)
    col_indices = np.array([], dtype=np.int32)
    row_pointers = np.array([0, 0], dtype=np.int32)
    d = np.array([0.0], dtype=np.float64)
    x = np.array([5.0], dtype=np.float64)

    result = d_inverse_w_product(values, col_indices, row_pointers, d, x)

    print("\n--- Zero-Degree Solver Verification ---")
    print(f"- D^-1 W x result for zero-degree node : {result}")

    assert result.shape == (1,), "Error: zero-degree result shape mismatch."
    assert result[0] == 0.0, "Error: zero-degree node should produce 0."


def run_solver_validation_checks():
    values = np.array([1.0], dtype=np.float64)
    col_indices = np.array([0], dtype=np.int32)
    row_pointers = np.array([0, 1], dtype=np.int32)
    d = np.array([1.0], dtype=np.float64)
    x = np.array([1.0], dtype=np.float64)

    print("\n--- Solver Input Validation Verification ---")

    assert_raises_value_error(
        d_inverse_w_product,
        values,
        col_indices,
        row_pointers,
        np.array([1.0, 2.0], dtype=np.float64),
        x,
    )
    print("- Rejected invalid degree vector length")

    assert_raises_value_error(
        d_inverse_w_product,
        values,
        col_indices,
        row_pointers,
        d,
        np.array([1.0, 2.0], dtype=np.float64),
    )
    print("- Rejected invalid x vector length")

    assert_raises_value_error(
        d_inverse_w_product,
        np.array([1.0, 2.0], dtype=np.float64),
        col_indices,
        row_pointers,
        d,
        x,
    )
    print("- Rejected mismatched values and column indices")

    assert_raises_value_error(
        find_fiedler_vector,
        values,
        col_indices,
        row_pointers,
        d,
        max_iter=0,
    )
    print("- Rejected non-positive max_iter")

    assert_raises_value_error(
        find_fiedler_vector,
        values,
        col_indices,
        row_pointers,
        d,
        tol=0.0,
    )
    print("- Rejected non-positive tolerance")


if __name__ == "__main__":
    run_isolated_ncut_check()
    run_d_weighted_normalization_check()
    run_zero_degree_check()
    run_solver_validation_checks()
    print("\nSTATUS: NCUT SOLVER VERIFIED")
