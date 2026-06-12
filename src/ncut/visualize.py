import numpy as np
from pathlib import Path
from PIL import Image as pil


def print_segmentation_summary(image_shape, graph_info, fiedler_vector, partition_result):
    """
    Print raw segmentation data to console.
    """
    print("\n" + "=" * 70)
    print("SEGMENTATION RESULTS - RAW DATA OUTPUT")
    print("=" * 70)
    
    print(f"\n[IMAGE INFO]")
    print(f"  Shape (HxW): {image_shape}")
    print(f"  Total pixels: {image_shape[0] * image_shape[1]}")
    
    print(f"\n[GRAPH INFO]")
    print(f"  Nodes: {graph_info['num_nodes']}")
    print(f"  Edges: {graph_info['num_edges']}")
    print(f"  Sparsity: {1.0 - (2 * graph_info['num_edges'] / (graph_info['num_nodes'] ** 2)):.4f}")
    
    print(f"\n[FIEDLER VECTOR STATS]")
    print(f"  Min value: {np.min(fiedler_vector):.6f}")
    print(f"  Max value: {np.max(fiedler_vector):.6f}")
    print(f"  Mean value: {np.mean(fiedler_vector):.6f}")
    print(f"  Median value: {np.median(fiedler_vector):.6f}")
    print(f"  Std dev: {np.std(fiedler_vector):.6f}")
    
    print(f"\n[PARTITION RESULTS]")
    mask = partition_result["mask"]
    threshold = partition_result["threshold"]
    ncut_score = partition_result["ncut"]
    
    print(f"  Threshold: {threshold:.6f}")
    print(f"  Region A size: {np.sum(mask)} pixels ({100*np.sum(mask)/len(mask):.2f}%)")
    print(f"  Region B size: {np.sum(~mask)} pixels ({100*np.sum(~mask)/len(mask):.2f}%)")
    print(f"  NCut Score: {ncut_score:.6f}")
    print(f"  Balance ratio: {min(np.sum(mask), np.sum(~mask)) / max(np.sum(mask), np.sum(~mask)):.4f}")
    
    print("\n" + "=" * 70 + "\n")


def print_fiedler_heatmap(fiedler_vector, image_height, image_width, sample_every=1):
    """
    Print Fiedler vector as ASCII heatmap (downsampled for readability).
    """
    fiedler_grid = fiedler_vector.reshape((image_height, image_width))
    
    # Normalize to 0-9 for ASCII display
    fiedler_min = np.min(fiedler_grid)
    fiedler_max = np.max(fiedler_grid)
    fiedler_normalized = ((fiedler_grid - fiedler_min) / (fiedler_max - fiedler_min + 1e-8)) * 9
    fiedler_int = fiedler_normalized.astype(int)
    
    print("[FIEDLER VECTOR HEATMAP]")
    print("(0=Low, 9=High, sampled every {} pixels)".format(sample_every))
    print()
    
    for i in range(0, image_height, sample_every):
        for j in range(0, image_width, sample_every):
            print(fiedler_int[i, j], end=" ")
        print()
    print()


def print_partition_mask(mask, image_height, image_width, sample_every=1):
    """
    Print binary partition mask as ASCII (0 or 1).
    """
    mask_grid = mask.reshape((image_height, image_width))
    
    print("[BINARY PARTITION MASK]")
    print("(0=Region B, 1=Region A, sampled every {} pixels)".format(sample_every))
    print()
    
    for i in range(0, image_height, sample_every):
        for j in range(0, image_width, sample_every):
            print(int(mask_grid[i, j]), end=" ")
        print()
    print()


def save_mask_as_image(mask, image_height, image_width, output_path, invert=False):
    """
    Save binary mask as a grayscale PNG image.
    
    Args:
        mask: Boolean array (True/False or 1/0)
        image_height: Height of original image
        image_width: Width of original image
        output_path: Path to save PNG
        invert: If True, swap black/white (Region A=white, Region B=black)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    mask_grid = mask.reshape((image_height, image_width))
    mask_uint8 = (mask_grid.astype(np.uint8)) * 255
    
    if invert:
        mask_uint8 = 255 - mask_uint8
    
    mask_image = pil.fromarray(mask_uint8, mode="L")
    mask_image.save(output_path)
    
    print(f"✓ Saved binary mask to: {output_path}")


def save_fiedler_as_image(fiedler_vector, image_height, image_width, output_path):
    """
    Save Fiedler vector as a grayscale PNG image (normalized to 0-255).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fiedler_grid = fiedler_vector.reshape((image_height, image_width))
    
    # Normalize to 0-255
    fiedler_min = np.min(fiedler_grid)
    fiedler_max = np.max(fiedler_grid)
    fiedler_normalized = ((fiedler_grid - fiedler_min) / (fiedler_max - fiedler_min + 1e-8)) * 255
    fiedler_uint8 = fiedler_normalized.astype(np.uint8)
    
    fiedler_image = pil.fromarray(fiedler_uint8, mode="L")
    fiedler_image.save(output_path)
    
    print(f"✓ Saved Fiedler heatmap to: {output_path}")


def save_comparison(original_image, mask, image_height, image_width, output_dir):
    """
    Save original image, mask, and overlay for comparison.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save original (normalized to 0-255)
    original_uint8 = (original_image * 255).astype(np.uint8)
    if original_image.ndim == 2:
        original_pil = pil.fromarray(original_uint8, mode="L")
    else:
        original_uint8 = (original_image * 255).astype(np.uint8)
        original_pil = pil.fromarray(original_uint8, mode="RGB")
    original_pil.save(output_dir / "00_original.png")
    print(f"✓ Saved original image")
    
    # Save mask
    save_mask_as_image(mask, image_height, image_width, output_dir / "01_mask.png")
    
    # Save Fiedler heatmap
    fiedler_path = output_dir / "02_fiedler_heatmap.png"
    # Note: We'd need fiedler vector for this, so skip if not available
    print(f"✓ Comparison images saved to: {output_dir}")
