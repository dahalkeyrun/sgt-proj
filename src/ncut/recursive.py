# from src.ncut import partition
import numpy as np

from ncut.ncut import find_fiedler_vector
from ncut.partition import partition_graph


def extract_subgraph(weight_matrix, node_indices):
    return weight_matrix[node_indices][:, node_indices]


def compute_degree_vector_for_subgraph(subgraph_weight_matrix):
    return np.asarray(subgraph_weight_matrix.sum(axis=1)).ravel()


def compute_fiedler_vector_for_subgraph(
    subgraph_weight_matrix,
    max_iter=5000,
    tol=1e-5,
    backend="cpu",
):

    degree_vector = compute_degree_vector_for_subgraph(subgraph_weight_matrix)
    return find_fiedler_vector(
        subgraph_weight_matrix.data,
        subgraph_weight_matrix.indices,
        subgraph_weight_matrix.indptr,
        degree_vector,
        max_iter=max_iter,
        tol=tol,
        backend=backend,
    )


def is_stable_fiedler_vector(fiedler_vector, num_bins=20, flatness_threshold=0.50):
    """
    Paper Section 3.2 Step 4: Histogram stability criterion.
    Checks the ratio of minimum bin count (valley) to maximum bin count (peak).
    If min_count / max_count >= flatness_threshold (0.50), the eigenvector is
    flat/oscillatory — the proposed split is unreliable, skip it.
    Returns True if the vector is stable (bimodal/valley present), False if flat.
    """
    counts, _ = np.histogram(fiedler_vector, bins=num_bins)
    max_count = counts.max()
    min_count = counts.min()
    if max_count == 0:
        return False
    return (min_count / max_count) < flatness_threshold


def compute_partition_for_subgraph(
    subgraph_weight_matrix,
    threshold_method="ncut",
    max_iter=5000,
    tol=1e-5,
    backend="cpu",
):
    """
    Compute a single NCut partition for a subgraph.
    Returns None if the eigenvector fails the stability criterion.
    """

    fiedler_vector = compute_fiedler_vector_for_subgraph(
        subgraph_weight_matrix,
        max_iter=max_iter,
        tol=tol,
        backend=backend,
    )

    print(
        "Fiedler:",
        np.min(fiedler_vector),
        np.max(fiedler_vector),
        np.mean(fiedler_vector),
        np.median(fiedler_vector)
    )

    # Paper Section 3.2: stability criterion — discard if eigenvector is flat/oscillatory.
    if not is_stable_fiedler_vector(fiedler_vector):
        return None

    return partition_graph(
        subgraph_weight_matrix,
        fiedler_vector,
        threshold_method
    )

def build_label_map(total_nodes, segment_list):
    """
    convert final segments into a label array
    (segment ma break garne image lai, and the assign unique id's to the
    segments)
    """

    label_map = np.full(total_nodes, -1, dtype=np.int32)

    for label_id, segment_nodes in enumerate(segment_list):
        label_map[segment_nodes] = label_id

    return label_map

def recursive_split(
    weight_matrix,
    current_node_indices,
    segment_list,
    current_depth,
    max_depth,
    min_segment_size,
    ncut_threshold,
    threshold_method="ncut",
    max_iter=5000,
    tol=1e-5,
    backend="cpu"):
    
    if current_depth >= max_depth:
        segment_list.append(current_node_indices)
        return

    if len(current_node_indices) < min_segment_size:
        segment_list.append(current_node_indices)
        return
    
    if len(current_node_indices) == 0:
        return

    subgraph_weight_matrix = extract_subgraph(
        weight_matrix,
        current_node_indices
    )

    partition_result = compute_partition_for_subgraph(
        subgraph_weight_matrix,
        threshold_method,
        max_iter=max_iter,
        tol=tol,
        backend=backend,
    )

    # Stability criterion failed — treat whole segment as a leaf.
    if partition_result is None:
        print(
            f"Depth={current_depth} | "
            f"Nodes={len(current_node_indices)} | "
            f"Stability criterion failed — keeping as leaf."
        )
        segment_list.append(current_node_indices)
        return

    partition_mask = partition_result["mask"]
    ncut_value = partition_result["ncut"]
    print(
        f"Depth={current_depth} | "
        f"Nodes={len(current_node_indices)} | "
        f"NCut={ncut_value:.4f}"
    )

    if ncut_value > ncut_threshold:
        segment_list.append(current_node_indices)
        return

    if (partition_mask.sum() == 0 or partition_mask.sum() == len(partition_mask)):
        segment_list.append(current_node_indices)
        return

    left_node_indices = current_node_indices[partition_mask]
    right_node_indices = current_node_indices[~partition_mask]

    print(
        f"Split -> "
        f"Left={len(left_node_indices)} "
        f"Right={len(right_node_indices)}"
    )

    recursive_split(
        weight_matrix,
        left_node_indices,
        segment_list,
        current_depth + 1,
        max_depth,
        min_segment_size,
        ncut_threshold,
        threshold_method,
        max_iter=max_iter,
        tol=tol,
        backend=backend,
    )

    recursive_split(
        weight_matrix,
        right_node_indices,
        segment_list,
        current_depth + 1,
        max_depth,
        min_segment_size,
        ncut_threshold,
        threshold_method,
        max_iter=max_iter,
        tol=tol,
        backend=backend,
    )


def recursive_partition(
    weight_matrix,
    max_depth=4,
    min_segment_size=50,
    ncut_threshold=0.4,
    threshold_method="ncut",
    max_iter=5000,
    tol=1e-5,
    backend="cpu",
):
    total_nodes = weight_matrix.shape[0]

    root_node_indices = np.arange(total_nodes)
    segment_list = []

    recursive_split(
        weight_matrix=weight_matrix,
        current_node_indices= root_node_indices,
        segment_list = segment_list,
        current_depth = 0,
        max_depth=max_depth,
        min_segment_size = min_segment_size,
        ncut_threshold = ncut_threshold,
        threshold_method = threshold_method,
        max_iter=max_iter,
        tol=tol,
        backend=backend,
    )

    label_map = build_label_map(total_nodes, segment_list)

    return label_map

