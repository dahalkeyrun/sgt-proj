import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import ArpackNoConvergence, eigsh

from ncut.backend import Backend, is_gpu_available, to_numpy


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
    W = csr_matrix(
        (values, col_indices, row_pointers),
        shape=(num_nodes, num_nodes),
        dtype=np.float64,
    )
    result = W.dot(x)
    nonzero = d != 0.0
    result[nonzero] = result[nonzero] / d[nonzero]
    result[~nonzero] = 0.0
    return result


def d_weighted_norm(x, d):
    x = np.asarray(x, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    return float(np.sqrt(np.sum(d * x ** 2)))


def find_k_eigenvectors(
    values,
    col_indices,
    row_pointers,
    d,
    k: int = 4,
    max_iter: int = 5000,
    tol: float = 1e-5,
    backend: Backend | str = Backend.CPU,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve the Shi-Malik generalized eigenproblem: (D - W) y = lambda D y
    for the k smallest eigenvalues and corresponding eigenvectors.

    Returns:
        eigenvalues: 1D array of shape (k,) sorted in ascending order.
        eigenvectors: 2D array of shape (num_pixels, k) where columns are eigenvectors.
    """
    if k <= 0:
        raise ValueError("k must be positive.")

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

    backend = Backend.from_string(backend)
    num_pixels = len(d)
    k_actual = min(k, max(1, num_pixels - 1))

    if backend == Backend.GPU and is_gpu_available():
        try:
            import cupy as cp
            import cupyx.scipy.sparse as cpsparse
            import cupyx.scipy.sparse.linalg as cpsl

            d_inv_sqrt = cp.maximum(cp.asarray(d, dtype=cp.float64), 1e-10) ** (-0.5)
            D_inv_sqrt = cpsparse.diags(d_inv_sqrt, offsets=0, dtype=cp.float64)
            val_gpu = cp.asarray(values, dtype=cp.float64)
            col_gpu = cp.asarray(col_indices, dtype=cp.int32)
            row_gpu = cp.asarray(row_pointers, dtype=cp.int32)

            W_gpu = cpsparse.csr_matrix(
                (val_gpu, col_gpu, row_gpu),
                shape=(num_pixels, num_pixels),
                dtype=cp.float64
            )
            d_gpu = cp.maximum(cp.asarray(d, dtype=cp.float64), 1e-10)
            D_gpu = cpsparse.diags(d_gpu, offsets=0, dtype=cp.float64)
            L_gpu = D_gpu - W_gpu
            L_sym = D_inv_sqrt.dot(L_gpu).dot(D_inv_sqrt)

            gpu_k = max(2, k_actual)
            eigenvalues_gpu, z_gpu = cpsl.eigsh(
                L_sym,
                k=gpu_k,
                which="SA",
                maxiter=max_iter,
                tol=tol,
            )
            y_gpu = D_inv_sqrt.dot(z_gpu)

            eigenvalues = to_numpy(eigenvalues_gpu[:k_actual])
            eigenvectors = to_numpy(y_gpu[:, :k_actual])

            sort_idx = np.argsort(eigenvalues)
            return eigenvalues[sort_idx], eigenvectors[:, sort_idx]
        except Exception as err:
            print(f"[Warning] GPU eigensolver failed ({err}); falling back to CPU.")

    # CPU solver via SciPy sparse ARPACK.
    # Solve the generalized problem (D - W)y = λDy through the symmetric
    # normalized Laplacian: D^-1/2 L D^-1/2 z = λz with y = D^-1/2 z.
    # which='SM' on the generalized problem can silently skip the zero
    # eigenvalue and return the 3rd-smallest vector as index 1; which='SA'
    # on the normalized Laplacian deterministically returns [0, Fiedler].
    W = csr_matrix(
        (values, col_indices, row_pointers),
        shape=(num_pixels, num_pixels),
        dtype=np.float64
    )

    d_safe = np.maximum(d, 1e-10)
    D = diags(d_safe, offsets=0, dtype=np.float64)
    L = D - W

    d_inv_sqrt = np.zeros(num_pixels, dtype=np.float64)
    nonzero = d > 0.0
    d_inv_sqrt[nonzero] = 1.0 / np.sqrt(d[nonzero])
    L_sym = diags(d_inv_sqrt) @ L @ diags(d_inv_sqrt)

    v0 = np.linspace(-1.0, 1.0, num_pixels, dtype=np.float64)

    cpu_k = max(2, k_actual)

    try:
        eigenvalues, z = eigsh(
            A=L_sym,
            k=cpu_k,
            which="SA",
            maxiter=max_iter,
            tol=tol,
            v0=v0,
        )
    except ArpackNoConvergence:
        eigenvalues, z = eigsh(
            A=L_sym,
            k=cpu_k,
            which="SA",
            maxiter=max_iter * 2,
            tol=tol,
            v0=v0,
        )

    eigenvectors = d_inv_sqrt[:, np.newaxis] * z

    sort_idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sort_idx[:k_actual]]
    eigenvectors = eigenvectors[:, sort_idx[:k_actual]]

    return eigenvalues, eigenvectors


def find_fiedler_vector(
    values,
    col_indices,
    row_pointers,
    d,
    max_iter=5000,
    tol=1e-5,
    backend="cpu",
):
    """
    Solve the Shi-Malik generalized eigenproblem:

        (D - W)y = λDy

    and return the Fiedler vector
    (second smallest eigenvector).
    """
    eigenvalues, eigenvectors = find_k_eigenvectors(
        values,
        col_indices,
        row_pointers,
        d,
        k=2,
        max_iter=max_iter,
        tol=tol,
        backend=backend,
    )

    if eigenvectors.shape[1] > 1:
        fiedler_value = eigenvalues[1]
        fiedler_vector = eigenvectors[:, 1]
    else:
        fiedler_value = eigenvalues[0]
        fiedler_vector = eigenvectors[:, 0]

    # Paper: thresholding is on y₁ directly. Normalize sign so the positive
    # lobe is consistently foreground (eigsh may return either orientation).
    # The Fiedler vector's mean is ~0 by construction, so use the lobe with the
    # larger peak magnitude instead of the numerically unstable mean sign.
    if abs(fiedler_vector.min()) > abs(fiedler_vector.max()):
        fiedler_vector = -fiedler_vector

    print(
        f"Fiedler eigenvalue: "
        f"{fiedler_value:.8f}"
    )

    return fiedler_vector

