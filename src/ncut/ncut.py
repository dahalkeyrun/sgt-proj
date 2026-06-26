import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse import diags
from scipy.sparse.linalg import ArpackNoConvergence, eigsh


def validate_solver_inputs(values, col_indices, row_pointers, d, x):

    values = np.asarray(values)
    col_indices = np.asarray(col_indices)
    row_pointers = np.asarray(row_pointers)
    d = np.asarray(d)

    if row_pointers.ndim != 1:
        raise ValueError("row_pointers must be a 1D array")

    if len(row_pointers) < 2:
        raise ValueError("row_pointers must have at least 2 elements")

    num_pixels = len(row_pointers) - 1

    if values.ndim != 1:
        raise ValueError("values must be a 1D array")

    if col_indices.ndim != 1:
        raise ValueError("col_indices must be a 1D array")

    if d.ndim != 1:
        raise ValueError("d must be a 1D array")

    if len(values) != len(col_indices):
        raise ValueError(
            "values and col_indices must have the same length"
        )

    if len(d) != num_pixels:
        raise ValueError(
            "degree vector length must match number of graph nodes"
        )

    if row_pointers[0] != 0:
        raise ValueError("row_pointers must start with 0")

    if row_pointers[-1] != len(values):
        raise ValueError(
            "last row pointer must equal number of values"
        )

    if np.any(row_pointers[1:] < row_pointers[:-1]):
        raise ValueError(
            "row_pointers must be non-decreasing"
        )

    if len(col_indices) > 0:
        if (
            np.min(col_indices) < 0
            or np.max(col_indices) >= num_pixels
        ):
            raise ValueError(
                "col_indices contains an invalid node index"
            )

    if np.any(d < 0):
        raise ValueError(
            "degree vector d must not contain negative values"
        )

    if x is not None:

        x = np.asarray(x)

        if x.ndim != 1:
            raise ValueError("x must be a 1D array")

        if len(x) != num_pixels:
            raise ValueError(
                "x length must match number of graph nodes"
            )


def d_inverse_w_product(values, col_indices, row_pointers, d, x):
    validate_solver_inputs(values, col_indices, row_pointers, d, x)

    num_nodes = len(d)
    result = np.zeros(num_nodes, dtype=np.float64)

    for i in range(num_nodes):
        if d[i] == 0.0:
            continue
        start = row_pointers[i]
        end = row_pointers[i + 1]
        wx = 0.0
        for idx in range(start, end):
            j = col_indices[idx]
            wx += values[idx] * x[j]
        result[i] = wx / d[i]

    return result


def d_weighted_norm(x, d):
    x = np.asarray(x, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    return float(np.sqrt(np.sum(d * x ** 2)))


def find_fiedler_vector(
    values,
    col_indices,
    row_pointers,
    d,
    max_iter=5000,
    tol=1e-5,
):
    """
    Solve the Shi-Malik generalized eigenproblem:

        (D - W)y = λDy

    and return the Fiedler vector
    (second smallest eigenvector).
    """

    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")

    if tol <= 0:
        raise ValueError("tol must be positive.")

    validate_solver_inputs(
        values,
        col_indices,
        row_pointers,
        d,
        None
    )

    num_pixels = len(d)

    W = csr_matrix(
        (
            values,
            col_indices,
            row_pointers
        ),
        shape=(num_pixels, num_pixels),
        dtype=np.float64
    )

    D = diags(
        d,
        offsets=0,
        dtype=np.float64
    )

    L = D - W

    try:
        eigenvalues, eigenvectors = eigsh(
            A=L,
            M=D,
            k=2,
            which="SM",
            maxiter=max_iter,
            tol=tol,
        )
    except ArpackNoConvergence:
        eigenvalues, eigenvectors = eigsh(
            A=L,
            M=D,
            k=2,
            which="SM",
            maxiter=max_iter * 2,
            tol=tol,
        )

    fiedler_value = eigenvalues[1]
    fiedler_vector = eigenvectors[:, 1]

    # Paper: thresholding is on y₁ directly. Normalize sign so the positive
    # lobe is consistently foreground (eigsh may return either orientation).
    if fiedler_vector.mean() < 0:
        fiedler_vector = -fiedler_vector

    print(
        f"Fiedler eigenvalue: "
        f"{fiedler_value:.8f}"
    )

    return fiedler_vector