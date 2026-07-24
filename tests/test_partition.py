import os
import sys

import numpy as np
from scipy.sparse import csr_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from ncut.partition import partition_graph


def test_partition_graph_accepts_backend_argument():
    weight_matrix = csr_matrix(np.array([
        [0.0, 1.0],
        [1.0, 0.0],
    ], dtype=np.float64))
    fiedler_vector = np.array([0.2, -0.2], dtype=np.float64)

    result = partition_graph(weight_matrix, fiedler_vector, backend="cpu")

    assert set(result.keys()) == {"mask", "threshold", "ncut"}
    assert result["mask"].shape == (2,)
    assert np.isfinite(result["ncut"])


if __name__ == "__main__":
    test_partition_graph_accepts_backend_argument()
    print("STATUS: PARTITION MODULE VERIFIED")
