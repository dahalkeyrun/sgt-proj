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
from ncut.ncut import find_fiedler_vector, find_k_eigenvectors
from ncut.partition import partition_graph, partition_kway
from ncut.recursive import recursive_partition
from ncut.color import ColorMetric, RECOMMENDED_SIGMA_I
from ncut.backend import Backend, warm_gpu_backend
from ncut.timing import StageTimer


app = Flask(__name__)
_warmed_segmentation_backends: set[Backend] = set()


def _warm_segmentation_path(backend: Backend) -> bool:
    if backend in _warmed_segmentation_backends:
        return True

    if backend == Backend.GPU and not warm_gpu_backend():
        return False

    image = np.linspace(0.0, 1.0, 64, dtype=np.float64).reshape((8, 8))
    weight_matrix, degree_vector = build_weight_matrix(
        image,
        sigma_i=0.1,
        sigma_x=4.0,
        radius=3,
        backend=backend,
    )
    fiedler_vector = find_fiedler_vector(
        weight_matrix.data,
        weight_matrix.indices,
        weight_matrix.indptr,
        degree_vector,
        max_iter=200,
        tol=1e-3,
        backend=backend,
    )
    partition_graph(
        weight_matrix,
        fiedler_vector,
        threshold_method="ncut",
        backend=backend,
    )
    _warmed_segmentation_backends.add(backend)
    return True


@app.route("/api/warmup", methods=["POST"])
def warmup():
    data = request.get_json(silent=True) or {}
    backend = Backend.from_string(data.get("backend", "cpu"))

    start = time.perf_counter()
    warmed = _warm_segmentation_path(backend)
    warmup_ms = (time.perf_counter() - start) * 1000.0

    return jsonify({
        "backend": backend.value,
        "warmed": bool(warmed),
        "warmup_ms": round(warmup_ms, 1),
    })


def _decode_base64_image(data_url: str) -> pil.Image:
    """Strip the data-URL prefix and decode the base64 image bytes."""
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    image_bytes = base64.b64decode(data_url)
    return pil.open(io.BytesIO(image_bytes))


def _prepare_image(
    image_array: np.ndarray,
    resolution: int,
    color_metric: ColorMetric,
) -> np.ndarray:
    """
    Resize and normalize the image for graph construction.
    Preserves RGB channels for HSV, RGB, and TEXTURE metrics.
    Converts to grayscale only for GRAYSCALE metric.
    """
    from PIL import Image as PILImage

    pil_img = PILImage.fromarray(image_array)

    # Resize to target resolution (preserve aspect ratio)
    width, height = pil_img.size
    largest_side = max(width, height)
    if largest_side > resolution:
        scale = resolution / largest_side
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        pil_img = pil_img.resize((new_width, new_height), PILImage.LANCZOS)

    img = np.array(pil_img, dtype=np.float64)

    if color_metric == ColorMetric.GRAYSCALE:
        # Convert to single-channel grayscale, normalized to [0,1]
        if img.ndim == 3:
            img = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
        img = img / 255.0
    else:
        # Keep RGB channels in [0,255] — feature extraction normalizes internally
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        img = img[..., :3]   # drop alpha if present

    return img


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
    """Convert a float64 image array (grayscale [0,1] or RGB [0,255]) to a flat RGBA list."""
    rgba = []
    if image.ndim == 2:
        for val in image.ravel():
            v = int(val * 255)
            rgba.extend([v, v, v, 255])
    else:
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                px = image[y, x]
                r = int(np.clip(px[0], 0, 255))
                g = int(np.clip(px[1], 0, 255))
                b = int(np.clip(px[2], 0, 255))
                rgba.extend([r, g, b, 255])
    return rgba


def _foreground_rgba(image: np.ndarray, mask: np.ndarray, height: int, width: int) -> list:
    """
    Apply the mask to the original image.
    Region A pixels are shown at full opacity; Region B pixels are transparent.
    """
    rgba = []
    if image.ndim == 2:
        img_grid = image.reshape((height, width))
        mask_grid = mask.reshape((height, width))
        for i in range(height):
            for j in range(width):
                v = int(img_grid[i, j] * 255)
                a = 255 if mask_grid[i, j] else 0
                rgba.extend([v, v, v, a])
    else:
        img_grid = image.reshape((height, width, 3))
        mask_grid = mask.reshape((height, width))
        for i in range(height):
            for j in range(width):
                px = img_grid[i, j]
                r = int(np.clip(px[0], 0, 255))
                g = int(np.clip(px[1], 0, 255))
                b = int(np.clip(px[2], 0, 255))
                a = 255 if mask_grid[i, j] else 0
                rgba.extend([r, g, b, a])
    return rgba
def _labels_to_rgba(labels, height, width):
    colors = np.array([
        [230,25,75],
        [60,180,75],
        [255,225,25],
        [0,130,200],
        [245,130,48],
        [145,30,180],
        [70,240,240],
        [240,50,230],
        [210,245,60],
        [250,190,190],
        [0,128,128],
        [230,190,255],
        [170,110,40],
        [255,250,200],
        [128,0,0],
        [170,255,195]
    ], dtype=np.uint8)

    rgba = []

    for label in labels:
        r, g, b = colors[label % len(colors)]
        rgba.extend([int(r), int(g), int(b), 255])

    return rgba


def _fiedler_to_heatmap_rgb_fast(fiedler_vector: np.ndarray, height: int, width: int) -> list:
    grid = fiedler_vector.reshape((height, width))
    v_min, v_max = grid.min(), grid.max()
    flat = ((grid - v_min) / (v_max - v_min + 1e-8)).ravel()
    low = flat < 0.5
    rgba = np.empty((height * width, 4), dtype=np.uint8)

    t_low = flat[low] * 2.0
    rgba[low, 0] = np.clip(t_low * 255, 0, 255).astype(np.uint8)
    rgba[low, 1] = np.clip(t_low * 255, 0, 255).astype(np.uint8)
    rgba[low, 2] = 255

    t_high = (flat[~low] - 0.5) * 2.0
    rgba[~low, 0] = 255
    rgba[~low, 1] = np.clip((1.0 - t_high) * 255, 0, 255).astype(np.uint8)
    rgba[~low, 2] = np.clip((1.0 - t_high) * 255, 0, 255).astype(np.uint8)
    rgba[:, 3] = 255
    return rgba.ravel().tolist()


def _image_to_rgba_fast(image: np.ndarray) -> list:
    if image.ndim == 2:
        gray = np.clip(image.reshape((-1,)) * 255, 0, 255).astype(np.uint8)
        rgba = np.empty((gray.size, 4), dtype=np.uint8)
        rgba[:, 0] = gray
        rgba[:, 1] = gray
        rgba[:, 2] = gray
    else:
        rgb = np.clip(image.reshape((-1, 3))[:, :3], 0, 255).astype(np.uint8)
        rgba = np.empty((rgb.shape[0], 4), dtype=np.uint8)
        rgba[:, :3] = rgb
    rgba[:, 3] = 255
    return rgba.ravel().tolist()


def _foreground_rgba_fast(image: np.ndarray, mask: np.ndarray, height: int, width: int) -> list:
    if image.ndim == 2:
        gray = np.clip(image.reshape((-1,)) * 255, 0, 255).astype(np.uint8)
        rgba = np.empty((height * width, 4), dtype=np.uint8)
        rgba[:, 0] = gray
        rgba[:, 1] = gray
        rgba[:, 2] = gray
    else:
        rgb = np.clip(image.reshape((-1, 3))[:, :3], 0, 255).astype(np.uint8)
        rgba = np.empty((height * width, 4), dtype=np.uint8)
        rgba[:, :3] = rgb
    rgba[:, 3] = mask.reshape((-1,)).astype(np.uint8) * 255
    return rgba.ravel().tolist()


def _border_mask(height: int, width: int) -> np.ndarray:
    border = np.zeros((height, width), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return border.reshape((-1,))


def _select_binary_foreground_mask(mask: np.ndarray, height: int, width: int) -> tuple[np.ndarray, str]:
    mask = np.asarray(mask, dtype=bool).reshape((-1,))
    border = _border_mask(height, width)

    true_border = int(np.count_nonzero(mask & border))
    false_border = int(np.count_nonzero((~mask) & border))

    if true_border != false_border:
        if true_border < false_border:
            return mask, "binary_low_border_true"
        return ~mask, "binary_low_border_false"

    true_size = int(np.count_nonzero(mask))
    false_size = mask.size - true_size
    if true_size <= false_size:
        return mask, "binary_smaller_true"
    return ~mask, "binary_smaller_false"


def _select_recursive_foreground_mask(label_map: np.ndarray, height: int, width: int) -> tuple[np.ndarray, int | None, str]:
    labels = np.asarray(label_map, dtype=np.int32).reshape((-1,))
    unique_labels = np.unique(labels)

    if len(unique_labels) <= 1:
        return np.ones(labels.shape, dtype=bool), None, "recursive_single_segment"

    border = _border_mask(height, width)
    border_counts = {
        int(label): int(np.count_nonzero((labels == label) & border))
        for label in unique_labels
    }
    sizes = {
        int(label): int(np.count_nonzero(labels == label))
        for label in unique_labels
    }

    background_label = max(
        unique_labels,
        key=lambda label: (border_counts[int(label)], sizes[int(label)]),
    )
    background_label = int(background_label)
    return labels != background_label, background_label, "recursive_largest_border_segment"


def _select_foreground_mask(
    label_map: np.ndarray,
    height: int,
    width: int,
    segmentation_mode: str,
    foreground_mode: str,
    top_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, int | None, str]:
    labels = np.asarray(label_map, dtype=np.int32).reshape((-1,))

    if foreground_mode == "legacy":
        return labels == 1, 0, "legacy_label_1"

    if segmentation_mode == "binary":
        if top_mask is None:
            top_mask = labels.astype(bool)
        mask, strategy = _select_binary_foreground_mask(top_mask, height, width)
        return mask, None, strategy

    return _select_recursive_foreground_mask(labels, height, width)


def _labels_to_rgba_fast(labels, height, width):
    colors = np.array([
        [230,25,75],
        [60,180,75],
        [255,225,25],
        [0,130,200],
        [245,130,48],
        [145,30,180],
        [70,240,240],
        [240,50,230],
        [210,245,60],
        [250,190,190],
        [0,128,128],
        [230,190,255],
        [170,110,40],
        [255,250,200],
        [128,0,0],
        [170,255,195],
    ], dtype=np.uint8)
    rgb = colors[np.asarray(labels, dtype=np.int32) % len(colors)]
    rgba = np.empty((height * width, 4), dtype=np.uint8)
    rgba[:, :3] = rgb.reshape((-1, 3))
    rgba[:, 3] = 255
    return rgba.ravel().tolist()


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
        width, height       : int
        nodes, edges        : int
        num_segments        : int
        ncut_score          : float
        threshold           : float
        elapsed_ms          : float
        heatmap_rgba        : flat [r,g,b,a, ...] for the Fiedler heatmap
        segmentation_rgba   : flat [r,g,b,a, ...] for the segment labels
        nodes_rgba          : flat [r,g,b,a, ...] for the original (downscaled) image
        fg_rgba             : flat [r,g,b,a, ...] for the foreground extraction
    """
    data = request.get_json(force=True)
    if not data or "image" not in data:
        return jsonify({"error": "No image data received."}), 400

    # Paper-calibrated defaults (Fig. 4: σ_I=0.1, σ_X=4.0, r=5 for [0,1] images).
    sigma_i    = float(data.get("sigma_i", 0.1))
    sigma_x    = float(data.get("sigma_x", 4.0))
    radius     = int(data.get("radius", 5))          # independent parameter r (paper Eq. 11)
    resolution = int(data.get("resolution", 30))
    resolution = max(10, min(resolution, 100))  # clamp to safe range

    color_metric_str = data.get("color_metric", "grayscale")
    try:
        color_metric = ColorMetric.from_string(color_metric_str)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        backend = Backend.from_string(data.get("backend", "cpu"))
        segmentation_mode = str(data.get("segmentation_mode", "binary")).strip().lower()
        if segmentation_mode not in ("binary", "recursive", "kway"):
            return jsonify({"error": "segmentation_mode must be 'binary', 'recursive', or 'kway'."}), 400

        foreground_mode = str(data.get("foreground_mode", "auto")).strip().lower()
        if foreground_mode not in ("auto", "legacy"):
            return jsonify({"error": "foreground_mode must be 'auto' or 'legacy'."}), 400

        if backend == Backend.GPU:
            warm_gpu_backend()

        t0 = time.perf_counter()
        timer = StageTimer(enabled=True)

        # --- 1. Decode image ---
        pil_image = _decode_base64_image(data["image"])
        image_array = np.array(pil_image)
        image = _prepare_image(image_array, resolution, color_metric)
        height, width = image.shape[:2]
        timer.record("prepare_image_ms")

        # --- 2. Build weight matrix ---
        weight_matrix, degree_vector = build_weight_matrix(
            image,
            sigma_i=sigma_i,
            sigma_x=sigma_x,
            radius=radius,
            color_metric=color_metric,
            backend=backend,
        )
        num_nodes = weight_matrix.shape[0]
        num_edges = weight_matrix.nnz
        timer.record("graph_ms")

        # --- 3. K-Way Partitioning (GPU Accelerated or CPU) ---
        if segmentation_mode == "kway":
            k_clusters = int(data.get("k", 4))
            k_clusters = max(2, min(k_clusters, 16))

            eigenvalues, eigenvectors = find_k_eigenvectors(
                weight_matrix.data,
                weight_matrix.indices,
                weight_matrix.indptr,
                degree_vector,
                k=k_clusters,
                backend=backend,
            )
            timer.record("eigensolver_ms")

            kway_result = partition_kway(
                weight_matrix,
                eigenvectors,
                k=k_clusters,
                backend=backend,
            )
            timer.record("partition_ms")

            label_map = kway_result["labels"]
            ncut_score = kway_result["ncut"]
            fiedler_vec = eigenvectors[:, 1] if eigenvectors.shape[1] > 1 else eigenvectors[:, 0]

            num_segments = int(np.max(label_map)) + 1
            foreground_mask, background_label, foreground_strategy = _select_foreground_mask(
                label_map,
                height,
                width,
                segmentation_mode,
                foreground_mode,
            )
            segmentation_rgba = _labels_to_rgba_fast(label_map, height, width)

            elapsed_ms = (time.perf_counter() - t0) * 1000
            timings = timer.finish("total_ms")

            return jsonify({
                "width":             width,
                "height":            height,
                "nodes":             num_nodes,
                "edges":             num_edges,
                "backend":           backend.value,
                "segmentation_mode": segmentation_mode,
                "color_metric":      color_metric.value,
                "foreground_mode":   foreground_mode,
                "foreground_strategy": foreground_strategy,
                "background_label":   background_label,
                "num_segments":      num_segments,
                "ncut_score":        round(ncut_score, 6),
                "threshold":         None,
                "elapsed_ms":        round(elapsed_ms, 1),
                "timings":           timings,
                "heatmap_rgba":      _fiedler_to_heatmap_rgb_fast(fiedler_vec, height, width),
                "segmentation_rgba": segmentation_rgba,
                "nodes_rgba":        _image_to_rgba_fast(image),
                "fg_rgba":           _foreground_rgba_fast(image, foreground_mask, height, width),
                "label_map":         label_map.tolist(),
            })

        # --- 3. Find Fiedler vector (Binary / Recursive) ---
        fiedler_vector = find_fiedler_vector(
            weight_matrix.data,
            weight_matrix.indices,
            weight_matrix.indptr,
            degree_vector,
            backend=backend,
        )
        timer.record("eigensolver_ms")

        # --- 4a. Top-level partition ---
        top_partition = partition_graph(
            weight_matrix,
            fiedler_vector,
            threshold_method="ncut",
            backend=backend,
        )
        timer.record("partition_ms")
        ncut_score = top_partition["ncut"]
        threshold  = top_partition["threshold"]

        if segmentation_mode == "binary":
            label_map = top_partition["mask"].astype(np.int32)
            timer.record("segmentation_ms")

            num_segments = int(np.max(label_map)) + 1
            foreground_mask, background_label, foreground_strategy = _select_foreground_mask(
                label_map,
                height,
                width,
                segmentation_mode,
                foreground_mode,
                top_mask=top_partition["mask"],
            )
            segmentation_rgba = _labels_to_rgba_fast(label_map, height, width)

            elapsed_ms = (time.perf_counter() - t0) * 1000
            timings = timer.finish("total_ms")

            return jsonify({
                "width":             width,
                "height":            height,
                "nodes":             num_nodes,
                "edges":             num_edges,
                "backend":           backend.value,
                "segmentation_mode": segmentation_mode,
                "color_metric":      color_metric.value,
                "foreground_mode":   foreground_mode,
                "foreground_strategy": foreground_strategy,
                "background_label":   background_label,
                "num_segments":      num_segments,
                "ncut_score":        round(ncut_score, 6),
                "threshold":         round(threshold,  6),
                "elapsed_ms":        round(elapsed_ms, 1),
                "timings":           timings,
                "heatmap_rgba":      _fiedler_to_heatmap_rgb_fast(fiedler_vector, height, width),
                "segmentation_rgba": segmentation_rgba,
                "nodes_rgba":        _image_to_rgba_fast(image),
                "fg_rgba":           _foreground_rgba_fast(image, foreground_mask, height, width),
                "label_map":         label_map.tolist(),
            })

        # --- 4b. Recursive Partition ---
        label_map = recursive_partition(
            weight_matrix,
            max_depth=3,
            min_segment_size=30,
            ncut_threshold=0.15,
            threshold_method="ncut",
            backend=backend,
        )
        timer.record("recursive_ms")

        num_segments = int(np.max(label_map)) + 1
        foreground_mask, background_label, foreground_strategy = _select_foreground_mask(
            label_map,
            height,
            width,
            segmentation_mode,
            foreground_mode,
        )
        segmentation_rgba = _labels_to_rgba_fast(label_map, height, width)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        timings = timer.finish("total_ms")

        return jsonify({
            "width":             width,
            "height":            height,
            "nodes":             num_nodes,
            "edges":             num_edges,
            "backend":           backend.value,
            "segmentation_mode": segmentation_mode,
            "color_metric":      color_metric.value,
            "foreground_mode":   foreground_mode,
            "foreground_strategy": foreground_strategy,
            "background_label":   background_label,
            "num_segments":      num_segments,
            "ncut_score":        round(ncut_score, 6),
            "threshold":         round(threshold,  6),
            "elapsed_ms":        round(elapsed_ms, 1),
            "timings":           timings,
            "heatmap_rgba":      _fiedler_to_heatmap_rgb_fast(fiedler_vector, height, width),
            "segmentation_rgba": segmentation_rgba,
            "nodes_rgba":        _image_to_rgba_fast(image),
            "fg_rgba":           _foreground_rgba_fast(image, foreground_mask, height, width),
            "label_map":         label_map.tolist(),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
