import numpy as np
from scipy.sparse import issparse

from ncut.backend import Backend, to_numpy


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

        y_min = float(fiedler_vector.min())
        y_max = float(fiedler_vector.max())

        # Paper Section 3.1 Step 3: sweep l evenly spaced candidate thresholds
        # between min(y₁) and max(y₁), pick the one that minimises NCut.
        num_candidates = max(20, min(200, len(fiedler_vector) // 20))

        # Evenly spaced in VALUE space (not index space); exclude endpoints to
        # avoid degenerate partitions where one side has 0 nodes.
        candidate_thresholds = np.linspace(y_min, y_max, num_candidates + 2)[1:-1]

        best_threshold = None
        best_ncut = np.inf

        for threshold in candidate_thresholds:

            mask = compute_binary_partition(
                fiedler_vector,
                threshold
            )

            partition_size = mask.sum()

            if (
                partition_size == 0
                or
                partition_size == len(mask)
            ):
                continue

            ncut_value = compute_ncut(
                W,
                mask
            )

            if ncut_value < best_ncut:

                best_ncut = ncut_value
                best_threshold = threshold

        if best_threshold is None:
            raise ValueError(
                "Failed to find a valid NCut threshold."
            )

        return float(
            best_threshold
        )

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

    if backend == Backend.GPU:
        # The current implementation keeps partitioning on the CPU side.
        # This still accepts a GPU-selected backend and converts inputs to NumPy.
        pass

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
