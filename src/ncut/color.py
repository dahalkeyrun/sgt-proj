"""
Color metric feature vectors for Normalized Cuts affinity computation.

Reference: Shi & Malik (2000), Section 4 — "Experiments"
  The weight function w(i,j) = exp(-||F(i)-F(j)||² / σ_I²)
                               × exp(-||X(i)-X(j)||² / σ_X²)
  where F(i) is a feature vector defined per color metric:

    GRAYSCALE : F(i) = I(i)                     — scalar luma, range [0,1]
    HSV       : F(i) = [v, v·s·sin(h), v·s·cos(h)]  — paper's color mode
    RGB       : F(i) = [R/255, G/255, B/255]     — engineering approx, not in paper
    TEXTURE   : F(i) = [|I*f₁|, ..., |I*fₙ|]   — DOOG filter responses

Paper ground-truth σ_I values for [0,1]-normalized features:
    σ_I = 0.1   brightness images (general)
    σ_I = 0.007 high-contrast / radar images
    σ_I = 0.01  color images
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Tuple

import numpy as np

from ncut.backend import Backend, get_array_module


# ── Public API ────────────────────────────────────────────────────────────────

class ColorMetric(str, Enum):
    """
    Supported color feature metrics for graph affinity computation.
    Inherits from str so values are directly JSON-serialisable and
    comparable to plain strings without explicit .value access.
    """
    GRAYSCALE = "grayscale"
    HSV       = "hsv"
    RGB       = "rgb"
    TEXTURE   = "texture"

    @classmethod
    def from_string(cls, value: str) -> "ColorMetric":
        """
        Parse a color metric from a string, raising a clear ValueError
        for unrecognised values rather than returning None silently.
        """
        try:
            return cls(value.lower().strip())
        except ValueError:
            valid = [m.value for m in cls]
            raise ValueError(
                f"Unknown color metric '{value}'. "
                f"Valid options are: {valid}"
            )

    @property
    def label(self) -> str:
        """Human-readable label for UI display."""
        return {
            ColorMetric.GRAYSCALE : "Grayscale — F(i) = I(i)",
            ColorMetric.HSV       : "HSV Color — F(i) = [v, vs·sin(h), vs·cos(h)]  ✦ Paper",
            ColorMetric.RGB       : "RGB (approx) — F(i) = [R,G,B]/255",
            ColorMetric.TEXTURE   : "Texture (DOOG) — F(i) = [|I*f₁|...|I*fₙ|]",
        }[self]

    @property
    def is_paper_faithful(self) -> bool:
        """True if this metric matches a mode described in Shi & Malik (2000)."""
        return self in (ColorMetric.GRAYSCALE, ColorMetric.HSV, ColorMetric.TEXTURE)


# ── HSV conversion ────────────────────────────────────────────────────────────

def rgb_to_hsv_feature(r: float, g: float, b: float) -> Tuple[float, float, float]:
    """
    Convert an RGB pixel to the HSV feature vector used in Shi & Malik (2000).

    Paper (Section 4):
        F(i) = [v, v·s·sin(h), v·s·cos(h)]
    where h, s, v are the HSV components at pixel i.

    Args:
        r, g, b: Raw pixel values in [0, 255].

    Returns:
        (f0, f1, f2) feature vector where:
            f0 = v              ∈ [0, 1]
            f1 = v·s·sin(h)    ∈ [-1, 1]
            f2 = v·s·cos(h)    ∈ [-1, 1]

    Example:
        >>> rgb_to_hsv_feature(255, 0, 0)   # pure red
        (1.0, 0.0, 1.0)
        >>> rgb_to_hsv_feature(0, 0, 255)   # pure blue
        (1.0, -0.866..., -0.5)
    """
    r_n = r / 255.0
    g_n = g / 255.0
    b_n = b / 255.0

    c_max = max(r_n, g_n, b_n)
    c_min = min(r_n, g_n, b_n)
    delta = c_max - c_min

    # Value
    v = c_max

    # Saturation
    s = 0.0 if c_max == 0.0 else delta / c_max

    # Hue in radians [0, 2π]
    h = 0.0
    if delta != 0.0:
        if c_max == r_n:
            h = ((g_n - b_n) / delta) % 6.0
        elif c_max == g_n:
            h = (b_n - r_n) / delta + 2.0
        else:
            h = (r_n - g_n) / delta + 4.0
        h = h * math.pi / 3.0          # → radians
        if h < 0.0:
            h += 2.0 * math.pi

    return (v, v * s * math.sin(h), v * s * math.cos(h))


def hsv_feature_array(image_rgb: np.ndarray) -> np.ndarray:
    """
    Vectorised HSV feature computation for a full image.

    Args:
        image_rgb: uint8 or float array of shape (H, W, 3), channels = RGB.
                   Values may be in [0,255] (uint8) or [0,1] (float).

    Returns:
        np.ndarray of shape (H, W, 3) — the HSV feature map F,
        where F[y, x] = [v, v·s·sin(h), v·s·cos(h)].
    """
    img = image_rgb.astype(np.float64)
    if img.max() > 1.0:
        img = img / 255.0

    r, g, b = img[..., 0], img[..., 1], img[..., 2]

    c_max = np.maximum(np.maximum(r, g), b)
    c_min = np.minimum(np.minimum(r, g), b)
    delta = c_max - c_min

    v = c_max
    s = np.where(c_max == 0.0, 0.0, delta / np.where(c_max == 0.0, 1.0, c_max))

    # Hue per channel
    safe_delta = np.where(delta == 0.0, 1.0, delta)
    h_r = ((g - b) / safe_delta) % 6.0
    h_g = (b - r) / safe_delta + 2.0
    h_b = (r - g) / safe_delta + 4.0

    h = np.where(delta == 0.0, 0.0,
        np.where(c_max == r, h_r,
        np.where(c_max == g, h_g, h_b)))

    h = h * math.pi / 3.0
    h = np.where(h < 0.0, h + 2.0 * math.pi, h)

    f0 = v
    f1 = v * s * np.sin(h)
    f2 = v * s * np.cos(h)

    return np.stack([f0, f1, f2], axis=-1)


# ── Intensity distance functions ──────────────────────────────────────────────

def intensity_distance_sq(
    pixel_i: np.ndarray | float,
    pixel_j: np.ndarray | float,
    metric: ColorMetric,
) -> float:
    """
    Compute the squared feature distance ||F(i) - F(j)||² between two pixels.

    Args:
        pixel_i: Feature vector for pixel i. Shape and meaning depend on metric:
                 - GRAYSCALE : scalar float in [0, 1]
                 - HSV       : (f0, f1, f2) tuple/array from rgb_to_hsv_feature
                 - RGB       : (r/255, g/255, b/255) array
                 - TEXTURE   : DOOG feature vector (arbitrary length)
        pixel_j: Same structure as pixel_i.
        metric:  ColorMetric enum value.

    Returns:
        Scalar float — the squared Euclidean distance between F(i) and F(j).

    Note:
        For TEXTURE mode, this function expects pre-computed feature vectors.
        The caller (graph.py) is responsible for computing DOOG responses
        before calling this function.
    """
    if metric == ColorMetric.GRAYSCALE:
        diff = float(pixel_i) - float(pixel_j)
        return diff * diff

    diff = np.asarray(pixel_i, dtype=np.float64) - np.asarray(pixel_j, dtype=np.float64)
    return float(np.dot(diff, diff))


def intensity_distance_sq_rgb_normalized(r1: float, g1: float, b1: float, r2: float, g2: float, b2: float) -> float:
    """
    Isolate standard normalized RGB squared distance helper.
    """
    dr = (r1 - r2) / 255.0
    dg = (g1 - g2) / 255.0
    db = (b1 - b2) / 255.0
    return (dr * dr + dg * dg + db * db) / 3.0


# ── Feature extraction helpers ────────────────────────────────────────────────

def extract_pixel_feature(
    image: np.ndarray,
    y: int,
    x: int,
    metric: ColorMetric,
    doog_features: np.ndarray | None = None,
) -> np.ndarray | float:
    """
    Extract the feature vector F(i) for pixel (y, x) under the given metric.

    Args:
        image:         Grayscale (H,W) or RGB (H,W,3) numpy array.
                       Grayscale values must be in [0,1].
                       RGB values may be in [0,255] (uint8 or float).
        y, x:          Pixel coordinates.
        metric:        ColorMetric enum value.
        doog_features: Pre-computed DOOG feature map of shape (H, W, K),
                       required only when metric == ColorMetric.TEXTURE.

    Returns:
        Scalar float for GRAYSCALE, or 1D numpy array for HSV / RGB / TEXTURE.

    Raises:
        ValueError: if metric == TEXTURE and doog_features is None.
        ValueError: if metric == HSV or RGB and image is not 3-channel.
    """
    if metric == ColorMetric.GRAYSCALE:
        if image.ndim == 2:
            return float(image[y, x])
        # RGB image — convert to grayscale on the fly
        px = image[y, x].astype(np.float64)
        if px.max() > 1.0:
            px = px / 255.0
        return 0.299 * px[0] + 0.587 * px[1] + 0.114 * px[2]

    if metric == ColorMetric.HSV:
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError(
                "HSV metric requires a 3-channel RGB image. "
                f"Got image.shape = {image.shape}"
            )
        px = image[y, x]
        r, g, b = float(px[0]), float(px[1]), float(px[2])
        return np.array(rgb_to_hsv_feature(r, g, b), dtype=np.float64)

    if metric == ColorMetric.RGB:
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError(
                "RGB metric requires a 3-channel RGB image. "
                f"Got image.shape = {image.shape}"
            )
        px = image[y, x].astype(np.float64)
        if px.max() > 1.0:
            px = px / 255.0
        return px[:3]

    if metric == ColorMetric.TEXTURE:
        if doog_features is None:
            raise ValueError(
                "TEXTURE metric requires pre-computed doog_features array. "
                "Pass the DOOG feature map as doog_features=..."
            )
        return doog_features[y, x].astype(np.float64)

    raise ValueError(f"Unhandled metric: {metric}")


def build_feature_map(
    image: np.ndarray,
    color_metric: ColorMetric,
    doog_features: np.ndarray | None = None,
    backend: Backend | str = Backend.CPU,
) -> np.ndarray:
    """
    Build a full-image feature map for the given color metric.

    Returns:
        Feature map of shape (H, W, C) where C is 1 for GRAYSCALE,
        3 for RGB/HSV, or K for TEXTURE.
    """
    backend = Backend.from_string(backend)
    xp = get_array_module(backend)

    image_arr = xp.asarray(image)

    if color_metric == ColorMetric.GRAYSCALE:
        if image_arr.ndim == 2:
            luma = image_arr.astype(xp.float64)
        else:
            img = image_arr.astype(xp.float64)
            if img.max() > 1.0:
                img = img / 255.0
            luma = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
        return luma[..., xp.newaxis]

    if color_metric == ColorMetric.RGB:
        if image_arr.ndim != 3 or image_arr.shape[2] < 3:
            raise ValueError(
                "RGB metric requires a 3-channel RGB image. "
                f"Got image.shape = {image_arr.shape}"
            )
        img = image_arr.astype(xp.float64)
        if img.max() > 1.0:
            img = img / 255.0
        return img[..., :3]

    if color_metric == ColorMetric.HSV:
        if image_arr.ndim != 3 or image_arr.shape[2] < 3:
            raise ValueError(
                "HSV metric requires a 3-channel RGB image. "
                f"Got image.shape = {image_arr.shape}"
            )

        img = image_arr.astype(xp.float64)
        if img.max() > 1.0:
            img = img / 255.0

        r = img[..., 0]
        g = img[..., 1]
        b = img[..., 2]

        c_max = xp.maximum(xp.maximum(r, g), b)
        c_min = xp.minimum(xp.minimum(r, g), b)
        delta = c_max - c_min

        v = c_max
        s = xp.where(c_max == 0.0, 0.0, delta / xp.where(c_max == 0.0, 1.0, c_max))

        safe_delta = xp.where(delta == 0.0, 1.0, delta)
        h_r = ((g - b) / safe_delta) % 6.0
        h_g = (b - r) / safe_delta + 2.0
        h_b = (r - g) / safe_delta + 4.0

        h = xp.where(delta == 0.0, 0.0,
            xp.where(c_max == r, h_r,
            xp.where(c_max == g, h_g, h_b)))

        h = h * math.pi / 3.0
        h = xp.where(h < 0.0, h + 2.0 * math.pi, h)

        f0 = v
        f1 = v * s * xp.sin(h)
        f2 = v * s * xp.cos(h)
        return xp.stack([f0, f1, f2], axis=-1)

    if color_metric == ColorMetric.TEXTURE:
        if doog_features is None:
            raise ValueError(
                "TEXTURE metric requires pre-computed doog_features array. "
                "Pass the DOOG feature map as doog_features=..."
            )
        feature_map = xp.asarray(doog_features)
        if feature_map.ndim != 3:
            raise ValueError(
                "doog_features must be a 3D array of shape (H, W, K)."
            )
        return feature_map

    raise ValueError(f"Unhandled metric: {color_metric}")


# ── Sigma guidance ────────────────────────────────────────────────────────────

RECOMMENDED_SIGMA_I: dict[ColorMetric, float] = {
    ColorMetric.GRAYSCALE : 0.1,
    ColorMetric.HSV       : 0.01,
    ColorMetric.RGB       : 0.1,
    ColorMetric.TEXTURE   : 0.1,
}
