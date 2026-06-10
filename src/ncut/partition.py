from multiprocessing import Value

import numpy as np
from scipy.sparse import issparse

def validate_partition_imputs(W, fiedler_vector):
    if not issparse(W):
        raise TypeError("W must be a scipy sparse matrix.")
    
    if W.shape[0] != W.shape[1]:
        raise TypeError("W must be a scipy sparse matrix.")
    
    if len(fiedler_vector.shape) != 1:
        raise ValueError("Fiedler vector must be 1D.")
    
    if W.shape[0] != len(fiedler_vector):
        raise ValueError("Weight matrix and Fiedler vector size mismatch.")
    
    if W.shape[0] == 0:
        raise ValueError("Graph cannot be empty.")
    
    if not np.all(np.isfinite(fiedler_vector)):
        raise ValueError("Fiedler Vector contains invalid values.")



    
def find_threshold(fiedler_vector, method="median"):
    
    if method == "median":
        return float(np.median(fiedler_vector))
    
    if method == "mean":
        return float(np.mean(fiedler_vector))
    
    else:
        raise ValueError(f"Unknown threshold method: {method}")


def compute_binary_partition(fiedler_vector, threshold):
    """
    Convert Fiedler Vector into binary partition mask.
    """

    mask = fiedler_vector > threshold

    return mask

def compute_assoc(w, mask):

    assoc_value = w[mask].sum()

    return float(assoc_value)


def compute_cut(W, mask):

    A = mask
    B = ~mask

    cut_value = W[A][:, B].sum()
    
    return float(cut_value)

def compute_ncut(w, mask):

    cut_value = compute_cut(w, mask)

    assoc_a = compute_assoc(w, mask)
    assoc_b = compute_assoc(w, ~mask)

    if assoc_a == 0 or assoc_b == 0:
        return np.inf
    
    ncut_value = (cut_value / assoc_a + cut_value / assoc_b)

    return float(ncut_value)




def partition_graph(W, fiedler_vector, threshold_method="median"):
    """

    Perform spectral graph partitioning 
    using the Fiedler vector

    """

    validate_partition_imputs(W, fiedler_vector)

    threshold = find_threshold(fiedler_vector, method=threshold_method)

    mask = compute_binary_partition(fiedler_vector, threshold)

    ncut_value = compute_ncut(W, mask)


    return {
        "mask": mask,
        "threshold": threshold,
        "ncut": ncut_value,
    }

