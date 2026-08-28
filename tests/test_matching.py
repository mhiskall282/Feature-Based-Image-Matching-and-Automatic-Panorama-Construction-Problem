"""
tests/test_matching.py
"""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.features import detect_and_describe
from src.matching import match_descriptors, apply_ratio_test, create_matcher


def make_checkerboard(w=200, h=200):
    img = np.zeros((h, w), dtype=np.uint8)
    for r in range(0, h, 50):
        for c in range(0, w, 50):
            if (r // 50 + c // 50) % 2 == 0:
                img[r:r+50, c:c+50] = 200
    return img


class TestMatching:
    def setup_method(self):
        self.img  = make_checkerboard()
        self.img2 = np.roll(self.img, 10, axis=1)

    def test_sift_match_returns_keys(self):
        f1 = detect_and_describe(self.img,  "SIFT")
        f2 = detect_and_describe(self.img2, "SIFT")
        if f1["num_kp"] < 4 or f2["num_kp"] < 4:
            pytest.skip("Insufficient keypoints in synthetic image")
        res = match_descriptors(f1, f2)
        assert "good_matches" in res
        assert "num_good_matches" in res
        assert res["time_s"] >= 0

    def test_orb_match_returns_keys(self):
        f1 = detect_and_describe(self.img,  "ORB")
        f2 = detect_and_describe(self.img2, "ORB")
        if f1["num_kp"] < 4 or f2["num_kp"] < 4:
            pytest.skip("Insufficient keypoints")
        res = match_descriptors(f1, f2)
        assert "good_matches" in res

    def test_empty_descriptor_handled(self):
        blank = np.zeros((200, 200), dtype=np.uint8)
        f1 = detect_and_describe(blank, "ORB")
        f2 = detect_and_describe(self.img, "ORB")
        res = match_descriptors(f1, f2)
        assert res["num_good_matches"] == 0

    def test_create_matcher_sift(self):
        m = create_matcher("SIFT")
        assert m is not None

    def test_create_matcher_orb(self):
        m = create_matcher("ORB")
        assert m is not None

    def test_ratio_test(self):
        class FakeM:
            def __init__(self, d):
                self.distance = d
                self.queryIdx = self.trainIdx = self.imgIdx = 0
        pairs = [(FakeM(10), FakeM(100)),  # passes  0.10 < 0.75
                 (FakeM(70), FakeM(80))]   # fails   0.875 > 0.75
        good = apply_ratio_test(pairs, threshold=0.75)
        assert len(good) == 1
