#!/usr/bin/env python3
"""
Main entry point for image segmentation using Normalized Cuts.

Usage:
    python run_segmentation.py path/to/image.png [--sigma-i 1.0] [--sigma-x 2.0] [--radius 2]
"""

import sys
import os
import argparse
from pathlib import Path

import numpy as np

# Add src to path for imports
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
    save_comparison,
)


def run_segmentation_pipeline(
    image_path,
    sigma_i=1.0,
    sigma_x=2.0,
    radius=2,
    threshold_method="median",
    output_dir=None,
):
    """
    Run the complete Normalized Cuts segmentation pipeline.
    
    Args:
        image_path: Path to input image
        sigma_i: Intensity distance parameter
        sigma_x: Spatial distance parameter
        radius: Neighborhood radius in pixels
        threshold_method: "median" or "mean"
        output_dir: Directory to save results (if None, skip saving)
    
    Returns:
        Dictionary with results
    """
    
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Step 1: Load Image
    print(f"\n[STEP 1] Loading image: {image_path}")
    image = load_image(image_path, mode="grayscale", max_size=256)
    image_height, image_width = image.shape
    print(f"  ✓ Loaded {image_height}x{image_width} grayscale image")
    
    # Step 2: Build Weight Matrix
    print(f"\n[STEP 2] Building weight matrix")
    print(f"  Parameters: sigma_i={sigma_i}, sigma_x={sigma_x}, radius={radius}")
    weight_matrix, degree_vector = build_weight_matrix(
        image,
        sigma_i=sigma_i,
        sigma_x=sigma_x,
        radius=radius,
    )
    num_nodes = weight_matrix.shape[0]
    num_edges = weight_matrix.nnz
    print(f"  ✓ Graph: {num_nodes} nodes, {num_edges} edges")
    
    # Step 3: Find Fiedler Vector
    print(f"\n[STEP 3] Computing Fiedler vector (eigensolver)")
    fiedler_vector = find_fiedler_vector(
        weight_matrix.data,
        weight_matrix.indices,
        weight_matrix.indptr,
        degree_vector,
        max_iter=500,
        tol=1e-6,
    )
    print(f"  ✓ Fiedler vector computed")
    
    # Step 4: Partition Graph
    print(f"\n[STEP 4] Partitioning graph using Fiedler vector")
    partition_result = partition_graph(
        weight_matrix,
        fiedler_vector,
        threshold_method=threshold_method,
    )
    mask = partition_result["mask"]
    threshold = partition_result["threshold"]
    ncut_score = partition_result["ncut"]
    print(f"  ✓ Partitioning complete (NCut score: {ncut_score:.6f})")
    
    # Step 5: Print Raw Results
    print(f"\n[STEP 5] Displaying results")
    graph_info = {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
    }
    print_segmentation_summary(
        image.shape,
        graph_info,
        fiedler_vector,
        partition_result,
    )
    
    # Print ASCII heatmaps for small images
    if image_height <= 32 and image_width <= 32:
        print_fiedler_heatmap(fiedler_vector, image_height, image_width, sample_every=1)
        print_partition_mask(mask, image_height, image_width, sample_every=1)
    else:
        print_fiedler_heatmap(fiedler_vector, image_height, image_width, sample_every=max(1, image_height//16))
        print_partition_mask(mask, image_height, image_width, sample_every=max(1, image_height//16))
    
    # Step 6: Save Results (optional)
    if output_dir:
        print(f"\n[STEP 6] Saving results to {output_dir}")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save binary mask
        save_mask_as_image(mask, image_height, image_width, output_dir / "mask.png")
        
        # Save Fiedler heatmap
        save_fiedler_as_image(fiedler_vector, image_height, image_width, output_dir / "fiedler_heatmap.png")
        
        # Save original and comparison
        save_comparison(image, mask, image_height, image_width, output_dir)
        
        print(f"  ✓ Results saved to {output_dir}")
    
    return {
        "image": image,
        "image_shape": image.shape,
        "weight_matrix": weight_matrix,
        "degree_vector": degree_vector,
        "fiedler_vector": fiedler_vector,
        "partition_result": partition_result,
        "mask": mask,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Segment image using Normalized Cuts"
    )
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("--sigma-i", type=float, default=1.0, help="Intensity distance parameter (default: 1.0)")
    parser.add_argument("--sigma-x", type=float, default=2.0, help="Spatial distance parameter (default: 2.0)")
    parser.add_argument("--radius", type=float, default=2, help="Neighborhood radius (default: 2)")
    parser.add_argument("--threshold", choices=["median", "mean"], default="median", help="Threshold method (default: median)")
    parser.add_argument("--output", type=str, default="outputs", help="Output directory (default: outputs/)")
    
    args = parser.parse_args()
    
    try:
        results = run_segmentation_pipeline(
            image_path=args.image_path,
            sigma_i=args.sigma_i,
            sigma_x=args.sigma_x,
            radius=args.radius,
            threshold_method=args.threshold,
            output_dir=args.output,
        )
        print("\n✓ Segmentation completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
