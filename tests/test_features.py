"""
tests/test_features.py
======================
Unit tests for feature detection and description.
Uses synthetic small images — no external dataset dependency.
"""
import sys
from pathlib import Path
import numpy as np
import cv2
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.features import create_detector, detect_and_describe


def make_test_image(w=200, h=200):
    """Synthetic checkerboard — provides detectable corners."""
    img = np.zeros((h, w), dtype=np.uint8)
    sq = 25
    for r in range(0, h, sq * 2):
        for c in range(0, w, sq * 2):
            img[r:r+sq, c:c+sq] = 255
            img[r+sq:r+sq*2, c+sq:c+sq*2] = 255
    return img


class TestDetectorFactory:
    def test_sift_creation(self):
        det = create_detector("SIFT")
        assert det is not None

    def test_orb_creation(self):
        det = create_detector("ORB")
        assert det is not None

    def test_case_insensitive(self):
        det = create_detector("sift")
        assert det is not None

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            create_detector("SURF")


class TestDetectAndDescribe:
    def setup_method(self):
        self.img = make_test_image()

    def test_sift_detects_keypoints(self):
        result = detect_and_describe(self.img, "SIFT")
        assert result["num_kp"] > 0
        assert result["descriptors"] is not None
        assert result["descriptors"].dtype == np.float32
        assert result["desc_shape"][1] == 128  # SIFT descriptor dim

    def test_orb_detects_keypoints(self):
        result = detect_and_describe(self.img, "ORB")
        assert result["num_kp"] > 0
        assert result["descriptors"] is not None
        assert result["descriptors"].dtype == np.uint8  # binary descriptor

    def test_timing_recorded(self):
        result = detect_and_describe(self.img, "SIFT")
        assert result["time_s"] > 0.0

    def test_empty_image_returns_zero(self):
        blank = np.zeros((200, 200), dtype=np.uint8)
        result = detect_and_describe(blank, "SIFT")
        assert result["num_kp"] == 0

    def test_returns_dict_keys(self):
        result = detect_and_describe(self.img, "ORB")
        for key in ["keypoints","descriptors","method","num_kp","time_s"]:
            assert key in result
