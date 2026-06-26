import os
import sys
import math
import json
import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from ncut.color import (
    ColorMetric,
    rgb_to_hsv_feature,
    hsv_feature_array,
    intensity_distance_sq,
    extract_pixel_feature,
)


def test_from_string_round_trips():
    assert ColorMetric.from_string("grayscale") == ColorMetric.GRAYSCALE
    assert ColorMetric.from_string("HSV") == ColorMetric.HSV
    assert ColorMetric.from_string("  RGB  ") == ColorMetric.RGB
    with pytest.raises(ValueError, match="Unknown color metric"):
        ColorMetric.from_string("lab")


def test_color_metric_json_serializable():
    payload = {"metric": ColorMetric.HSV}
    result = json.dumps(payload)
    assert '"hsv"' in result


def test_rgb_to_hsv_feature_pure_red():
    v, f1, f2 = rgb_to_hsv_feature(255, 0, 0)
    assert abs(v - 1.0) < 1e-9
    assert abs(f1 - 0.0) < 1e-9    # sin(0) = 0
    assert abs(f2 - 1.0) < 1e-9    # cos(0) = 1


def test_rgb_to_hsv_feature_pure_blue():
    v, f1, f2 = rgb_to_hsv_feature(0, 0, 255)
    assert abs(v - 1.0) < 1e-9
    # h = 4π/3, sin(4π/3) = -√3/2, cos(4π/3) = -0.5
    assert abs(f1 - (-math.sqrt(3)/2)) < 1e-6
    assert abs(f2 - (-0.5)) < 1e-6


def test_rgb_to_hsv_feature_black():
    v, f1, f2 = rgb_to_hsv_feature(0, 0, 0)
    assert v == 0.0
    assert f1 == 0.0
    assert f2 == 0.0


def test_rgb_to_hsv_feature_white():
    v, f1, f2 = rgb_to_hsv_feature(255, 255, 255)
    assert abs(v - 1.0) < 1e-9
    assert f1 == 0.0   # s=0, so v·s·sin(h) = 0
    assert f2 == 0.0   # s=0, so v·s·cos(h) = 0


def test_intensity_distance_sq_grayscale_same():
    d = intensity_distance_sq(0.5, 0.5, ColorMetric.GRAYSCALE)
    assert d == 0.0


def test_intensity_distance_sq_grayscale_white_black():
    d = intensity_distance_sq(1.0, 0.0, ColorMetric.GRAYSCALE)
    assert abs(d - 1.0) < 1e-9


def test_intensity_distance_sq_hsv_same():
    red_feature = np.array(rgb_to_hsv_feature(255, 0, 0))
    d = intensity_distance_sq(red_feature, red_feature, ColorMetric.HSV)
    assert d == 0.0


def test_intensity_distance_sq_hsv_red_blue():
    red = np.array(rgb_to_hsv_feature(255, 0, 0))
    blue = np.array(rgb_to_hsv_feature(0, 0, 255))
    d = intensity_distance_sq(red, blue, ColorMetric.HSV)
    assert abs(d - 3.0) < 1e-4


def test_extract_pixel_feature_grayscale_on_grayscale():
    img = np.array([[0.0, 0.5], [1.0, 0.25]])
    f = extract_pixel_feature(img, 0, 1, ColorMetric.GRAYSCALE)
    assert abs(f - 0.5) < 1e-9


def test_extract_pixel_feature_hsv_raises_on_grayscale():
    img = np.array([[0.5, 0.5], [0.5, 0.5]])
    with pytest.raises(ValueError, match="3-channel"):
        extract_pixel_feature(img, 0, 0, ColorMetric.HSV)


def test_extract_pixel_feature_texture_raises_without_doog():
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="doog_features"):
        extract_pixel_feature(img, 0, 0, ColorMetric.TEXTURE)


def test_weight_suppression_calibrated():
    # Cross-region (white→black grayscale): should be near zero
    d_cross = intensity_distance_sq(1.0, 0.0, ColorMetric.GRAYSCALE)
    w_cross = math.exp(-d_cross / (0.1 ** 2))
    assert w_cross < 1e-40, f"Cross-region weight should be ~0, got {w_cross}"

    # Same-region (identical): should be 1.0
    d_same = intensity_distance_sq(0.5, 0.5, ColorMetric.GRAYSCALE)
    w_same = math.exp(-d_same / (0.1 ** 2))
    assert w_same == 1.0


def test_hsv_feature_array_shape_type():
    img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
    features = hsv_feature_array(img)
    assert features.shape == (10, 10, 3)
    assert features.dtype == np.float64


def test_color_metric_paper_faithful():
    assert ColorMetric.GRAYSCALE.is_paper_faithful is True
    assert ColorMetric.HSV.is_paper_faithful is True
    assert ColorMetric.TEXTURE.is_paper_faithful is True
    assert ColorMetric.RGB.is_paper_faithful is False
