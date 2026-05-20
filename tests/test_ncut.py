import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from ncut.graph import build_graph_structures
from ncut.ncut import find_fiedler_vector

def run_isolated_ncut_check():

    mock_image = np.array([
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0]
        ], dtype=np.float64)

    sigma_i = 0.5
    sigma_x = 2.0
    radius = 2

    values, col_indices, row_pointers, d = build_graph_structures(mock_image, sigma_i, sigma_x, radius)

    fiedler = find_fiedler_vector(values, col_indices, row_pointers, d, max_iter=500, tol=1e-6)

    print("\n--- Solver Diagnostic Verification ---")
    print(f"Total Partition Elements Calculated : {len(fiedler)} (Expected: 16)")
    
    reshaped_fiedler = fiedler.reshape((4, 4))
    print("\nReshaped Fiedler Vector Grid Layout (Rounded to 3 decimals):")
    print(np.round(reshaped_fiedler, 3))
    
    left_sign = np.sign(reshaped_fiedler[0, 0])
    right_sign = np.sign(reshaped_fiedler[0, 2])
    
    print(f"\nSign Alignment: Left Side = {left_sign} | Right Side = {right_sign}")
    
    assert left_sign != right_sign, "Error: The Fiedler Vector failed to find the image split boundary."
    print("\nSTATUS: NCUT SOLVER MATHEMATICALLY VERIFIED")

if(__name__) == "__main__":
    run_isolated_ncut_check()
