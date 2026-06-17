import numpy as np

from ncut.ncut import find_fiedler_vector
from ncut.partition import partition_graph


def compute_partition_for_subgraph(weight_matrix, threshold_method="median"):
    degree_vector = np.asarray(weight_matrix.sum(axis=1)).ravel()

    fiedler_vector = find_fiedler_vector(
        weight_matrix.data,
        weight_matrix.indices,
        weight_matrix.indptr,
        degree_vector
    )

    partition_result = partition_graph(
        weight_matrix,
        fiedler_vector,
        threshold_method
    )

    return partition_result


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

