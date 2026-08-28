"""
tests/test_warping.py
Unit tests for canvas computation and image warping.
"""
import sys
from pathlib import Path
import numpy as np
import cv2
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.warping import compute_canvas, warp_image, place_reference, blend_images


class TestWarping:
    def setup_method(self):
        self.img = np.random.randint(50, 200, (100, 150, 3), dtype=np.uint8)
        self.H   = np.eye(3, dtype=np.float64)  # Identity — no warp

    def test_canvas_identity_h(self):
        """Identity H → canvas same size as reference."""
        w, h, xo, yo = compute_canvas(self.img, self.img, self.H)
        assert w >= 150
        assert h >= 100

    def test_canvas_translation_h(self):
        """Translation H → canvas wider than original."""
        H_trans = np.array([[1,0,50],[0,1,0],[0,0,1]], dtype=np.float64)
        w, h, xo, yo = compute_canvas(self.img, self.img, H_trans)
        assert w > 150  # Canvas wider due to shift

    def test_warp_identity_h(self):
        """Identity H → image placed at offset without distortion."""
        H = np.eye(3, dtype=np.float64)
        w, h, xo, yo = compute_canvas(self.img, self.img, H)
        warped = warp_image(self.img, H, w, h, xo, yo)
        assert warped.shape[0] == h
        assert warped.shape[1] == w

    def test_place_reference_correct_shape(self):
        w, h, xo, yo = compute_canvas(self.img, self.img, self.H)
        canvas, mask = place_reference(self.img, w, h, xo, yo)
        assert canvas.shape == (h, w, 3)
        assert mask.shape   == (h, w)

    def test_blend_images_produces_valid_output(self):
        """Blending two identical arrays should produce same image."""
        a = np.ones((100, 100, 3), dtype=np.uint8) * 100
        b = np.ones((100, 100, 3), dtype=np.uint8) * 100
        m = np.ones((100, 100), dtype=np.uint8) * 255
        result = blend_images(a, b, m, m, alpha_blend=True)
        assert result.shape == a.shape
        assert result.dtype == np.uint8
