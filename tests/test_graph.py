import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from ncut.graph import build_graph_structures

def run_isolated_graph_check():
    
    mock_image = np.array([
        [0.0, 0.5, 1.0],
        [0.0, 0.5, 1.0],
        [0.0, 0.5, 1.0]
        ], dtype=np.float64)

    sigma_i = 1.0
    sigma_x = 2.0
    radius = 2
    
    values, col_indices, row_pointers, degree_vector = build_graph_structures( mock_image, sigma_i, sigma_x, radius)

    num_nodes = len(degree_vector)
    num_edges = len(values)

    print("\n--- Diagnostic Pipeline Verification ---")
    print(f"• Total Graph Vertices (Pixels Mapped)   : {num_nodes} (Expected: 9)")
    print(f"• Total Compressed Sparse Edges Tracked  : {num_edges}")
    print(f"• Row Pointers Offset Array Structure   : {row_pointers}")
    print(f"• Calculated Degree Vector elements     : {degree_vector.round(2)}")
    
    # Mathematical sanity checks
    assert num_nodes == 9, "Error: Graph vertex mapping size mismatch."
    assert len(row_pointers) == 10, "Error: CSR Row Pointers array must equal num_pixels + 1."

if __name__ == '__main__':
    run_isolated_graph_check()
