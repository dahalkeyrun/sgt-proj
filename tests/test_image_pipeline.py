import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from ncut.utils import flatten_image, load_image, reshape_vector_to_image


def run_image_pipeline_check():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    image_path = os.path.join(project_root, "data", "input", "pipeline_sample.png")

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            "Missing test image. Add a sample image at: "
            f"{image_path}"
        )

    grayscale_image = load_image(image_path, mode="grayscale", max_size=128)
    rgb_image = load_image(image_path, mode="rgb", max_size=128)

    print("\n--- Image Loading Pipeline Verification ---")
    print(f"Image path                 : {image_path}")
    print(f"Grayscale image shape      : {grayscale_image.shape}")
    print(f"RGB image shape            : {rgb_image.shape}")
    print(f"Grayscale value range      : {grayscale_image.min():.3f} to {grayscale_image.max():.3f}")
    print(f"RGB value range            : {rgb_image.min():.3f} to {rgb_image.max():.3f}")

    assert grayscale_image.ndim == 2, "Error: grayscale image should be 2D."
    assert rgb_image.ndim == 3, "Error: RGB image should be 3D."
    assert rgb_image.shape[2] == 3, "Error: RGB image should have 3 channels."

    assert max(grayscale_image.shape) <= 128, "Error: grayscale resize exceeded max_size."
    assert max(rgb_image.shape[:2]) <= 128, "Error: RGB resize exceeded max_size."

    assert grayscale_image.min() >= 0.0, "Error: grayscale values should be >= 0."
    assert grayscale_image.max() <= 1.0, "Error: grayscale values should be <= 1."
    assert rgb_image.min() >= 0.0, "Error: RGB values should be >= 0."
    assert rgb_image.max() <= 1.0, "Error: RGB values should be <= 1."

    flat_gray = flatten_image(grayscale_image)
    flat_rgb = flatten_image(rgb_image)

    expected_pixels = grayscale_image.shape[0] * grayscale_image.shape[1]

    print(f"Flattened grayscale shape  : {flat_gray.shape}")
    print(f"Flattened RGB shape        : {flat_rgb.shape}")

    assert flat_gray.shape == (expected_pixels,), "Error: grayscale flatten shape mismatch."
    assert flat_rgb.shape == (expected_pixels, 3), "Error: RGB flatten shape mismatch."

    restored_grid = reshape_vector_to_image(
        flat_gray,
        grayscale_image.shape[0],
        grayscale_image.shape[1],
    )

    print(f"Restored vector grid shape : {restored_grid.shape}")

    assert restored_grid.shape == grayscale_image.shape, "Error: restored grid shape mismatch."
    assert np.allclose(restored_grid, grayscale_image), "Error: restored grid values changed."

    print("\nSTATUS: IMAGE PIPELINE VERIFIED")


if __name__ == "__main__":
    run_image_pipeline_check()
