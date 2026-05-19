import numpy as np

def d_inverse_w_product(values, col_indices, row_pointers, d, x):

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
            result = 0.0

    return result

def find_fiedler_vector(values, col_indices, row_pointers, d, max_iter=300, tol=1e-5):

    num_pixels = len(row_pointers) - 1

    # Trivial first dominant eigenvector
    v0 = np.ones(num_pixels, dtype=np.float64)

    v0 = v0 / np.sqrt(np.sum((v0 ** 2) * d))

    np.random.seed(42)
    v1 = np.random.rand(num_pixels)

    v1 -= np.sum(v1 * d * v0) * v0
    v1 /= np.linalg.norm(v1)

    for iteration in range(max_iter):
        v1_old = v1.copy()

        Aw = d_inverse_w_product(values, col_indices, row_pointers, d, v1)

        projection = np.sum(v0 * d * v1)
        v1 = Aw - (projection * v0)

        norm = np.linalg.norm(v1)
        if norm == 0:
            v1 = np.random.rand(num_pixels)
            v1 -= np.sum(v1 * d * v0) * v0
            continue

        v1 /= norm

        if np.linalg.norm(v1 - v1_old) < tol:
            break

    return v1
