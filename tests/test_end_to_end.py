#!/usr/bin/env python3
"""
End-to-end test with synthetic image.
Run this to see the full pipeline in action without needing real images.
"""

import sys
import os
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from ncut.utils import load_image, reshape_vector_to_image
from ncut.graph import build_weight_matrix
from ncut.ncut import find_fiedler_vector
from ncut.partition import partition_graph
from ncut.visualize import (
    print_segmentation_summary,
    print_fiedler_heatmap,
    print_partition_mask,
    save_mask_as_image,
    save_fiedler_as_image,
)


def create_synthetic_image(size=64):
    """
    Create a simple synthetic binary image for testing.
    Left side: black (0), Right side: white (1)
    """
    image = np.zeros((size, size), dtype=np.float64)
    image[:, size//2:] = 1.0
    return image


def test_with_real_image(image_path):
    """Run segmentation on a real image."""
    print(f"\n{'='*70}")
    print(f"RUNNING END-TO-END TEST WITH REAL IMAGE")
    print(f"{'='*70}")
    
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"✗ Image not found: {image_path}")
        return
    
    print(f"\n[STEP 1] Loading image: {image_path}")
    image = load_image(image_path, mode="grayscale", max_size=128)
    image_height, image_width = image.shape
    print(f"  ✓ Loaded {image_height}x{image_width} image")
    print(f"  Value range: [{image.min():.3f}, {image.max():.3f}]")
    
    print(f"\n[STEP 2] Building weight matrix...")
    weight_matrix, degree_vector = build_weight_matrix(
        image,
        sigma_i=0.5,
        sigma_x=2.0,
        radius=2,
    )
    print(f"  ✓ Graph: {weight_matrix.shape[0]} nodes, {weight_matrix.nnz} edges")
    
    print(f"\n[STEP 3] Computing Fiedler vector...")
    fiedler_vector = find_fiedler_vector(
        weight_matrix.data,
        weight_matrix.indices,
        weight_matrix.indptr,
        degree_vector,
        max_iter=500,
        tol=1e-6,
    )
    print(f"  ✓ Fiedler vector computed")
    
    print(f"\n[STEP 4] Partitioning graph...")
    partition_result = partition_graph(weight_matrix, fiedler_vector, threshold_method="median")
    print(f"  ✓ Partition complete (NCut: {partition_result['ncut']:.6f})")
    
    print(f"\n[STEP 5] Results:")
    graph_info = {"num_nodes": weight_matrix.shape[0], "num_edges": weight_matrix.nnz}
    print_segmentation_summary(image.shape, graph_info, fiedler_vector, partition_result)
    
    # Show heatmaps for smaller images
    if image_height <= 32:
        print_fiedler_heatmap(fiedler_vector, image_height, image_width, sample_every=1)
        print_partition_mask(partition_result["mask"], image_height, image_width, sample_every=1)


def test_with_synthetic_image():
    """Run segmentation on a synthetic binary image."""
    print(f"\n{'='*70}")
    print(f"RUNNING END-TO-END TEST WITH SYNTHETIC IMAGE")
    print(f"{'='*70}")
    
    # Create synthetic image
    print(f"\n[STEP 1] Creating synthetic 32x32 binary image (left=black, right=white)")
    image = create_synthetic_image(size=32)
    image_height, image_width = image.shape
    print(f"  ✓ Created {image_height}x{image_width} image")
    print(f"  Value range: [{image.min():.3f}, {image.max():.3f}]")
    
    # Build weight matrix
    print(f"\n[STEP 2] Building weight matrix (sigma_i=0.5, sigma_x=2.0, radius=2)")
    weight_matrix, degree_vector = build_weight_matrix(
        image,
        sigma_i=0.5,
        sigma_x=2.0,
        radius=2,
    )
    num_nodes = weight_matrix.shape[0]
    num_edges = weight_matrix.nnz
    print(f"  ✓ Graph: {num_nodes} nodes, {num_edges} edges")
    
    # Compute Fiedler vector
    print(f"\n[STEP 3] Computing Fiedler vector (eigensolver iteration)")
    fiedler_vector = find_fiedler_vector(
        weight_matrix.data,
        weight_matrix.indices,
        weight_matrix.indptr,
        degree_vector,
        max_iter=500,
        tol=1e-6,
    )
    print(f"  ✓ Fiedler vector: shape={fiedler_vector.shape}, range=[{fiedler_vector.min():.4f}, {fiedler_vector.max():.4f}]")
    
    # Partition graph
    print(f"\n[STEP 4] Partitioning graph using Fiedler vector (threshold=median)")
    partition_result = partition_graph(weight_matrix, fiedler_vector, threshold_method="median")
    mask = partition_result["mask"]
    print(f"  ✓ Partitioning complete")
    print(f"  ✓ Region A: {np.sum(mask)} pixels ({100*np.sum(mask)/len(mask):.2f}%)")
    print(f"  ✓ Region B: {np.sum(~mask)} pixels ({100*np.sum(~mask)/len(mask):.2f}%)")
    print(f"  ✓ NCut Score: {partition_result['ncut']:.6f}")
    
    # Print detailed results
    print(f"\n[STEP 5] Detailed Results:")
    graph_info = {"num_nodes": num_nodes, "num_edges": num_edges}
    print_segmentation_summary(image.shape, graph_info, fiedler_vector, partition_result)
    
    # Print ASCII heatmaps
    print_fiedler_heatmap(fiedler_vector, image_height, image_width, sample_every=1)
    print_partition_mask(mask, image_height, image_width, sample_every=1)
    
    # Save results
    print(f"\n[STEP 6] Saving results...")
    output_dir = Path("outputs") / "synthetic_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_mask_as_image(mask, image_height, image_width, output_dir / "mask.png")
    save_fiedler_as_image(fiedler_vector, image_height, image_width, output_dir / "fiedler_heatmap.png")
    print(f"  ✓ Saved to {output_dir}")
    
    print(f"\n✓ END-TO-END TEST COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    # Test 1: Synthetic image (always works)
    test_with_synthetic_image()
    
    # Test 2: Real image (if available)
    real_image_path = Path("data/input/pipeline_sample.png")
    if real_image_path.exists():
        test_with_real_image(real_image_path)
    else:
        print(f"\n(Real image not available at {real_image_path})")
    
    print(f"\n{'='*70}")
    print("All tests completed!")
    print(f"{'='*70}\n")
