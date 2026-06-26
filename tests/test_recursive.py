import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from ncut.utils import load_image
from ncut.graph import build_weight_matrix
from ncut.recursive import recursive_partition
from ncut.ncut import find_fiedler_vector


def test_recursive_partition():

    print("\n=== Loading Image ===")

    image = load_image(
        "data/input/pipeline_sample.png"
    )

    height, width = image.shape[:2]

    print(f"Image shape: {image.shape}")

    print("\n=== Building Graph ===")

    weight_matrix, degree_vector = build_weight_matrix(
        image,
        sigma_i=0.1,
        sigma_x=4.0,
        radius=5
    )

    print(f"Graph nodes : {weight_matrix.shape[0]}")
    print(f"Graph edges : {weight_matrix.nnz}")

    # ==========================================================
    # FIEDLER VECTOR DIAGNOSTIC
    # ==========================================================

    print("\n=== Computing Root Fiedler Vector ===")

    csr = weight_matrix.tocsr()

    values = csr.data
    col_indices = csr.indices
    row_pointers = csr.indptr

    fiedler_vector = find_fiedler_vector(
        values,
        col_indices,
        row_pointers,
        degree_vector
    )

    print("\n=== Fiedler Statistics ===")

    print(f"min    : {np.min(fiedler_vector)}")
    print(f"max    : {np.max(fiedler_vector)}")
    print(f"mean   : {np.mean(fiedler_vector)}")
    print(f"median : {np.median(fiedler_vector)}")
    print(f"std    : {np.std(fiedler_vector)}")

    print("\n=== Visualizing Fiedler Vector ===")

    fiedler_image = fiedler_vector.reshape(
        height,
        width
    )

    plt.figure(figsize=(8, 6))
    plt.imshow(fiedler_image, cmap="coolwarm")
    plt.colorbar()
    plt.title("Root Fiedler Vector")
    plt.show()

    # ==========================================================
    # RECURSIVE NCUT
    # ==========================================================

    print("\n=== Running Recursive NCut ===")

    label_map = recursive_partition(
        weight_matrix,
        max_depth=3,
        min_segment_size=50,
        ncut_threshold=0.4,
        threshold_method="ncut"
    )

    print("\n=== Validation ===")

    assert len(label_map) == weight_matrix.shape[0]

    assert np.all(label_map >= 0)

    unique_labels = np.unique(label_map)

    print(f"Number of segments: {len(unique_labels)}")

    print("\n=== Segment Statistics ===")

    for label in unique_labels:

        segment_size = np.sum(
            label_map == label
        )

        print(
            f"Segment {label}: "
            f"{segment_size} nodes"
        )

    segmentation_image = label_map.reshape(
        height,
        width
    )

    print("\n=== Top Left 20x20 Region ===")

    print(segmentation_image[:20, :20])

    print("\n=== Segmentation Image ===")

    print(
        f"Segmentation shape: "
        f"{segmentation_image.shape}"
    )

    print(
        f"Min label: "
        f"{segmentation_image.min()}"
    )

    print(
        f"Max label: "
        f"{segmentation_image.max()}"
    )

    plt.figure(figsize=(8, 6))
    plt.imshow(segmentation_image)
    plt.title("Final Segmentation Labels")
    plt.colorbar()
    plt.show()

    print("\n=== Test Passed ===")


if __name__ == "__main__":
    test_recursive_partition()