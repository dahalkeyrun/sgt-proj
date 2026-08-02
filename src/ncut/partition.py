import numpy as np
from scipy.sparse import issparse

from ncut.backend import Backend, is_gpu_available, to_numpy


def validate_partition_inputs(W, fiedler_vector):

    if not issparse(W):
        raise TypeError(
            "W must be a scipy sparse matrix."
        )

    if W.shape[0] != W.shape[1]:
        raise ValueError(
            "W must be square."
        )

    if len(fiedler_vector.shape) != 1:
        raise ValueError(
            "Fiedler vector must be 1D."
        )

    if W.shape[0] != len(fiedler_vector):
        raise ValueError(
            "Weight matrix and Fiedler vector size mismatch."
        )

    if W.shape[0] == 0:
        raise ValueError(
            "Graph cannot be empty."
        )

    if not np.all(np.isfinite(fiedler_vector)):
        raise ValueError(
            "Fiedler vector contains invalid values."
        )


def compute_binary_partition(
    fiedler_vector,
    threshold
):
    """
    Convert Fiedler vector into
    a binary partition mask.
    """

    return fiedler_vector > threshold


def compute_assoc(
    W,
    mask
):
    """
    assoc(A,V)

    Sum of all edge weights
    incident to nodes in A.
    """

    assoc_value = W[mask].sum()

    return float(assoc_value)


def compute_cut(
    W,
    mask
):
    """
    cut(A,B)

    Sum of edge weights
    crossing between partitions.
    """

    A = mask
    B = ~mask

    cut_value = W[A][:, B].sum()

    return float(cut_value)


def compute_ncut(
    W,
    mask
):
    """
    NCut(A,B)

    NCut =
        cut(A,B)/assoc(A,V)
      + cut(A,B)/assoc(B,V)
    """

    cut_value = compute_cut(
        W,
        mask
    )

    assoc_a = compute_assoc(
        W,
        mask
    )

    assoc_b = compute_assoc(
        W,
        ~mask
    )

    if assoc_a == 0 or assoc_b == 0:
        return np.inf

    ncut_value = (
        cut_value / assoc_a
        +
        cut_value / assoc_b
    )

    return float(ncut_value)


def find_exact_ncut_threshold(W, fiedler_vector):
    """
    Sweep every non-degenerate split induced by the sorted Fiedler vector.

    This follows the Shi-Malik thresholding step more closely than sampling
    evenly spaced values, while avoiding repeated sparse submatrix slicing.
    """
    sorted_indices = np.argsort(fiedler_vector, kind="mergesort")
    sorted_values = fiedler_vector[sorted_indices]
    num_nodes = len(sorted_values)

    split_positions = np.flatnonzero(sorted_values[:-1] < sorted_values[1:]) + 1
    if len(split_positions) == 0:
        raise ValueError(
            "Failed to find a valid NCut threshold."
        )

    ranks = np.empty(num_nodes, dtype=np.int32)
    ranks[sorted_indices] = np.arange(num_nodes, dtype=np.int32)

    degree_vector = np.asarray(W.sum(axis=1)).ravel().astype(np.float64)
    prefix_assoc = np.concatenate(
        ([0.0], np.cumsum(degree_vector[sorted_indices]))
    )
    total_assoc = prefix_assoc[-1]

    coo = W.tocoo()
    row_ranks = ranks[coo.row]
    col_ranks = ranks[coo.col]
    forward = row_ranks > col_ranks

    cut_delta = np.zeros(num_nodes + 1, dtype=np.float64)
    np.add.at(cut_delta, col_ranks[forward] + 1, coo.data[forward])
    np.add.at(cut_delta, row_ranks[forward] + 1, -coo.data[forward])
    cut_values = np.cumsum(cut_delta)

    assoc_b = prefix_assoc[split_positions]
    assoc_a = total_assoc - assoc_b
    cuts = cut_values[split_positions]

    valid = (assoc_a > 0.0) & (assoc_b > 0.0)
    if not np.any(valid):
        raise ValueError(
            "Failed to find a valid NCut threshold."
        )

    ncut_values = np.full(len(split_positions), np.inf, dtype=np.float64)
    ncut_values[valid] = cuts[valid] / assoc_a[valid] + cuts[valid] / assoc_b[valid]

    best_index = int(np.argmin(ncut_values))
    best_split = split_positions[best_index]

    return float(
        0.5 * (sorted_values[best_split - 1] + sorted_values[best_split])
    )


def find_threshold(
    W,
    fiedler_vector,
    method="ncut"
):
    """
    Find threshold for partitioning.

    Methods:
        median
        mean
        ncut
    """

    if method == "median":
        return float(
            np.median(fiedler_vector)
        )

    if method == "mean":
        return float(
            np.mean(fiedler_vector)
        )

    if method == "ncut":
        return find_exact_ncut_threshold(W, fiedler_vector)

    raise ValueError(
        f"Unknown threshold method: {method}"
    )


def partition_graph(
    W,
    fiedler_vector,
    threshold_method="ncut",
    backend="cpu",
):
    """
    Perform spectral partitioning
    using the Fiedler vector.
    """
    backend = Backend.from_string(backend)
    fiedler_vector = np.asarray(to_numpy(fiedler_vector), dtype=np.float64)

    validate_partition_inputs(
        W,
        fiedler_vector
    )

    threshold = find_threshold(
        W,
        fiedler_vector,
        method=threshold_method
    )

    mask = compute_binary_partition(
        fiedler_vector,
        threshold
    )

    ncut_value = compute_ncut(
        W,
        mask
    )

    return {
        "mask": mask,
        "threshold": threshold,
        "ncut": ncut_value,
    }


def kway_kmeans(
    eigenvectors: np.ndarray,
    k: int,
    backend: Backend | str = Backend.CPU,
    max_iter: int = 100,
    seed: int = 42,
) -> np.ndarray:
    """
    Perform row-normalized k-means clustering on top k eigenvectors.
    Supports GPU-accelerated k-means via CuPy.
    """
    backend = Backend.from_string(backend)
    num_pixels, num_cols = eigenvectors.shape
    k = min(k, num_pixels)

    if k <= 1:
        return np.zeros(num_pixels, dtype=np.int32)

    # Row normalization to unit sphere (Shi-Malik / Ng-Jordan-Weiss)
    norms = np.linalg.norm(eigenvectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    normalized_vecs = eigenvectors / norms

    if backend == Backend.GPU and is_gpu_available():
        try:
            import cupy as cp

            X_gpu = cp.asarray(normalized_vecs, dtype=cp.float32)
            cp.random.seed(seed)
            initial_idx = cp.random.choice(num_pixels, size=k, replace=False)
            centroids = X_gpu[initial_idx]

            labels_gpu = cp.zeros(num_pixels, dtype=cp.int32)
            for _ in range(max_iter):
                similarities = cp.dot(X_gpu, centroids.T)
                new_labels = cp.argmax(similarities, axis=1).astype(cp.int32)

                if cp.array_equal(new_labels, labels_gpu):
                    break
                labels_gpu = new_labels

                new_centroids = cp.zeros_like(centroids)
                for c in range(k):
                    mask = (labels_gpu == c)
                    if cp.any(mask):
                        cluster_mean = cp.mean(X_gpu[mask], axis=0)
                        mean_norm = cp.linalg.norm(cluster_mean)
                        new_centroids[c] = cluster_mean / max(float(mean_norm), 1e-10)
                    else:
                        new_centroids[c] = X_gpu[cp.random.randint(0, num_pixels)]
                centroids = new_centroids

            return to_numpy(labels_gpu).astype(np.int32)
        except Exception as err:
            print(f"[Warning] GPU k-means failed ({err}); falling back to CPU.")

    # CPU k-means using NumPy dot product cosine similarity
    rng = np.random.RandomState(seed)
    initial_idx = rng.choice(num_pixels, size=k, replace=False)
    centroids = normalized_vecs[initial_idx].astype(np.float32)
    X_cpu = normalized_vecs.astype(np.float32)
    labels = np.zeros(num_pixels, dtype=np.int32)

    for _ in range(max_iter):
        similarities = np.dot(X_cpu, centroids.T)
        new_labels = np.argmax(similarities, axis=1).astype(np.int32)

        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        for c in range(k):
            mask = (labels == c)
            if np.any(mask):
                c_mean = np.mean(X_cpu[mask], axis=0)
                c_norm = np.linalg.norm(c_mean)
                centroids[c] = c_mean / max(float(c_norm), 1e-10)
            else:
                centroids[c] = X_cpu[rng.randint(0, num_pixels)]

    return labels


def compute_kway_ncut(W, labels: np.ndarray, k: int) -> float:
    """
    Compute multi-class NCut cost across all k clusters:
    NCut_k = sum_{c=0}^{k-1} cut(A_c, V \ A_c) / assoc(A_c, V)
    """
    degree_vector = np.asarray(W.sum(axis=1)).ravel().astype(np.float64)
    total_ncut = 0.0

    for c in range(k):
        mask = (labels == c)
        assoc_c = degree_vector[mask].sum()
        if assoc_c == 0:
            continue
        cut_c = compute_cut(W, mask)
        total_ncut += cut_c / assoc_c

    return float(total_ncut)


def partition_kway(
    W,
    eigenvectors: np.ndarray,
    k: int = 4,
    backend: Backend | str = Backend.CPU,
) -> dict:
    """
    Perform k-way spectral partitioning using row-normalized k-means on top k eigenvectors.
    """
    backend = Backend.from_string(backend)
    labels = kway_kmeans(eigenvectors, k=k, backend=backend)
    ncut_value = compute_kway_ncut(W, labels, k)

    return {
        "labels": labels,
        "ncut": ncut_value,
        "k": k,
        "threshold": None,
    }

