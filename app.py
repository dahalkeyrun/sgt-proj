import sys
import os
import base64
import io
import time

import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image as pil

# Make the ncut package importable from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ncut.utils import resize_image
from ncut.graph import build_weight_matrix
from ncut.ncut import find_fiedler_vector
from ncut.partition import partition_graph

app = Flask(__name__)


def _decode_base64_image(data_url: str) -> pil.Image:
    """Strip the data-URL prefix and decode the base64 image bytes."""
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    image_bytes = base64.b64decode(data_url)
    return pil.open(io.BytesIO(image_bytes))


def _normalize_image(pil_image: pil.Image, max_size: int) -> np.ndarray:
    """Convert PIL image to a float64 grayscale numpy array in [0, 1]."""
    pil_image = pil_image.convert("L")
    pil_image = resize_image(pil_image, max_size=max_size)
    arr = np.asarray(pil_image, dtype=np.float64) / 255.0
    return arr


def _fiedler_to_heatmap_rgb(fiedler_vector: np.ndarray, height: int, width: int) -> list:
    """
    Convert the flat Fiedler vector to a flat RGBA list (r,g,b,a per pixel)
    using a blue-to-red colormap so the frontend can draw it on a canvas.
    """
    grid = fiedler_vector.reshape((height, width))
    v_min, v_max = grid.min(), grid.max()
    normalized = (grid - v_min) / (v_max - v_min + 1e-8)  # [0, 1]

    # Blue (low) → White (mid) → Red (high)
    rgba = []
    for val in normalized.ravel():
        if val < 0.5:
            t = val * 2.0          # 0 → 1
            r = int(t * 255)
            g = int(t * 255)
            b = 255
        else:
            t = (val - 0.5) * 2.0  # 0 → 1
            r = 255
            g = int((1.0 - t) * 255)
            b = int((1.0 - t) * 255)
        rgba.extend([r, g, b, 255])
    return rgba


def _mask_to_rgba(mask: np.ndarray, height: int, width: int) -> list:
    """
    Convert a boolean mask to a flat RGBA list.
    Region A (True)  → white (255, 255, 255, 255)
    Region B (False) → black (0, 0, 0, 255)
    """
    grid = mask.reshape((height, width))
    rgba = []
    for val in grid.ravel():
        if val:
            rgba.extend([255, 255, 255, 255])
        else:
            rgba.extend([0, 0, 0, 255])
    return rgba


def _image_to_rgba(image: np.ndarray) -> list:
    """Convert a float64 grayscale [0,1] array to a flat RGBA list."""
    rgba = []
    for val in image.ravel():
        v = int(val * 255)
        rgba.extend([v, v, v, 255])
    return rgba


def _foreground_rgba(image: np.ndarray, mask: np.ndarray, height: int, width: int) -> list:
    """
    Apply the mask to the original image.
    Region A pixels are shown at full opacity; Region B pixels are transparent.
    """
    img_grid = image.reshape((height, width))
    mask_grid = mask.reshape((height, width))
    rgba = []
    for i in range(height):
        for j in range(width):
            v = int(img_grid[i, j] * 255)
            a = 255 if mask_grid[i, j] else 0
            rgba.extend([v, v, v, a])
    return rgba


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/segment", methods=["POST"])
def segment():
    """
    POST body (JSON):
        image      : str   — base64 data-URL of the image
        sigma_i    : float — intensity kernel parameter (default 1.0)
        sigma_x    : float — spatial kernel parameter   (default 2.0)
        resolution : int   — max dimension of the resized image (default 30)

    Returns (JSON):
        width, height  : int
        nodes, edges   : int
        ncut_score     : float
        threshold      : float
        elapsed_ms     : float
        heatmap_rgba   : flat [r,g,b,a, ...] for the Fiedler heatmap
        bisection_rgba : flat [r,g,b,a, ...] for the binary mask
        nodes_rgba     : flat [r,g,b,a, ...] for the original (downscaled) image
        fg_rgba        : flat [r,g,b,a, ...] for the foreground extraction
    """
    t0 = time.perf_counter()

    data = request.get_json(force=True)
    if not data or "image" not in data:
        return jsonify({"error": "No image data received."}), 400

    # --- Parameters ---
    sigma_i    = float(data.get("sigma_i", 1.0))
    sigma_x    = float(data.get("sigma_x", 2.0))
    resolution = int(data.get("resolution", 30))
    resolution = max(10, min(resolution, 100))  # clamp to safe range

    try:
        # --- 1. Decode image ---
        pil_image = _decode_base64_image(data["image"])
        image = _normalize_image(pil_image, max_size=resolution)
        height, width = image.shape

        # --- 2. Build weight matrix ---
        weight_matrix, degree_vector = build_weight_matrix(
            image,
            sigma_i=sigma_i,
            sigma_x=sigma_x,
            radius=2,
        )
        num_nodes = weight_matrix.shape[0]
        num_edges = weight_matrix.nnz

        # --- 3. Find Fiedler vector ---
        fiedler_vector = find_fiedler_vector(
            weight_matrix.data,
            weight_matrix.indices,
            weight_matrix.indptr,
            degree_vector,
            max_iter=300,
            tol=1e-5,
        )

        # --- 4. Partition ---
        partition_result = partition_graph(
            weight_matrix,
            fiedler_vector,
            threshold_method="median",
        )
        mask      = partition_result["mask"]
        threshold = partition_result["threshold"]
        ncut      = partition_result["ncut"]

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return jsonify({
            "width":          width,
            "height":         height,
            "nodes":          num_nodes,
            "edges":          num_edges,
            "ncut_score":     round(float(ncut), 6),
            "threshold":      round(float(threshold), 6),
            "elapsed_ms":     round(elapsed_ms, 1),
            "heatmap_rgba":   _fiedler_to_heatmap_rgb(fiedler_vector, height, width),
            "bisection_rgba": _mask_to_rgba(mask, height, width),
            "nodes_rgba":     _image_to_rgba(image),
            "fg_rgba":        _foreground_rgba(image, mask, height, width),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
