import numpy as np

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
        raise ValueError("values and col_indices must have the same length")

    if len(d) != num_pixels:
        raise ValueError("degree vector length must match number of pixels graph nodes")
    
    if row_pointers[0] != 0:
        raise ValueError("row_pointers must start with 0")

    if row_pointers[-1] != len(values):
        raise ValueError("last row pointer must equal number of values")

    if np.any(row_pointers[1:] < row_pointers[:-1]):
        raise ValueError("row_pointers must be non-decreasing")

    if len(col_indices) > 0:
        if np.min(col_indices) < 0 or np.max(col_indices) >= num_pixels:
            raise ValueError("col_indices contains an invalid node index")

    if np.any(d < 0):
        raise ValueError("degree vector d must not contain negative values")

    if x is not None:
        x = np.asarray(x)

        if x.ndim != 1:
            raise ValueError("x must be a 1D array")

        if len(x) != num_pixels:
            raise ValueError("x length must match number of graph nodes")
        
def d_weighted_inner_product(a, b, d):
    return np.sum(a * b * d)

def d_weighted_norm(x, d):
    return np.sqrt(d_weighted_inner_product(x, x, d))

def normalize_d_weighted(x, d):
    norm = d_weighted_norm(x, d)

    if norm == 0:
        raise ValueError("Cannot normalize a zero vector")
    
    return x / norm

def d_inverse_w_product(values, col_indices, row_pointers, d, x):

    validate_solver_inputs(values, col_indices, row_pointers, d, x)

    num_pixels = len(row_pointers) - 1
    result = np.zeros(num_pixels, dtype=np.float64)

    for i in range(num_pixels):
        start = row_pointers[i]
        end = row_pointers[i+1]

        row_sum = 0.0
        for idx in range(start, end):
            j = col_indices[idx]
            row_sum += values[idx] * x[j]

        if d[i] > 0:
            result[i] = row_sum / d[i]
        else:
            result[i] = 0.0

    return result

def find_fiedler_vector(values, col_indices, row_pointers, d, max_iter=300, tol=1e-5):

    validate_solver_inputs(values, col_indices, row_pointers, d, None)

    if max_iter <= 0:
        raise ValueError("max_iter must be positive")

    if tol <= 0:
        raise ValueError("tol must be positive")

    num_pixels = len(row_pointers) - 1

    # Trivial first dominant eigenvector
    v0 = np.ones(num_pixels, dtype=np.float64)

    v0 = normalize_d_weighted(v0, d)

    np.random.seed(42)
    v1 = np.random.rand(num_pixels)

    v1 -= d_weighted_inner_product(v1, v0, d) * v0
    v1 /= d_weighted_norm(v1, d)

    for iteration in range(max_iter):
        v1_old = v1.copy()

        Aw = d_inverse_w_product(values, col_indices, row_pointers, d, v1)

        projection = d_weighted_inner_product(Aw, v0, d)
        v1 = Aw - (projection * v0)

        norm = d_weighted_norm(v1, d)
        if norm == 0:
            v1 = np.random.rand(num_pixels)
            v1 -= d_weighted_inner_product(v1, v0, d) * v0
            v1 = normalize_d_weighted(v1, d)
            continue

        v1 /= norm

        if np.linalg.norm(v1 - v1_old) < tol:
            break

    return v1
