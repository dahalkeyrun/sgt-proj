import numpy as np
from scipy.sparse import issparse


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
    threshold_method="ncut"
):
    """
    Perform spectral partitioning
    using the Fiedler vector.
    """

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