"""
tests/test_homography.py
Unit tests for RANSAC and homography estimation using synthetic correspondences.
"""
import sys
from pathlib import Path
import numpy as np
import cv2
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.homography import estimate_homography, diagnose_homography


def _synthetic_feature_pair(n_inliers=50, n_outliers=10, noise=1.0):
    """
    Build synthetic feature dicts with a known homography.
    True H = translation by (30, 20).
    """
    H_true = np.array([[1,0,30],[0,1,20],[0,0,1]], dtype=np.float64)

    # Random source points
    np.random.seed(42)
    src = np.random.uniform(50, 150, (n_inliers, 2)).astype(np.float32)
    # Map through H_true + noise
    dst_true = src + np.array([30, 20], dtype=np.float32)
    dst = dst_true + np.random.normal(0, noise, dst_true.shape).astype(np.float32)

    # Outliers
    outlier_src = np.random.uniform(10, 190, (n_outliers, 2)).astype(np.float32)
    outlier_dst = np.random.uniform(10, 190, (n_outliers, 2)).astype(np.float32)

    all_src = np.vstack([src, outlier_src])
    all_dst = np.vstack([dst, outlier_dst])

    # Build fake keypoints and matches
    kp1 = [cv2.KeyPoint(float(x), float(y), 1) for x, y in all_src]
    kp2 = [cv2.KeyPoint(float(x), float(y), 1) for x, y in all_dst]
    matches = [type("M", (), {"queryIdx": i, "trainIdx": i, "distance": 1.0})()
               for i in range(len(all_src))]

    feat1 = {"keypoints": kp1, "descriptors": None, "method": "SIFT", "num_kp": len(kp1)}
    feat2 = {"keypoints": kp2, "descriptors": None, "method": "SIFT", "num_kp": len(kp2)}
    return feat1, feat2, matches, H_true


class TestHomography:
    def test_succeeds_with_clean_correspondences(self):
        feat1, feat2, matches, _ = _synthetic_feature_pair(n_inliers=50, n_outliers=0, noise=0.5)
        res = estimate_homography(feat1, feat2, matches)
        assert res["H"] is not None
        assert res["success"] is True
        assert res["num_inliers"] >= 4

    def test_inlier_ratio_computed(self):
        feat1, feat2, matches, _ = _synthetic_feature_pair()
        res = estimate_homography(feat1, feat2, matches)
        if res["success"]:
            assert 0.0 <= res["inlier_ratio"] <= 1.0

    def test_fails_with_fewer_than_4_matches(self):
        feat1 = {"keypoints": [], "method": "SIFT", "num_kp": 0}
        feat2 = {"keypoints": [], "method": "SIFT", "num_kp": 0}
        res = estimate_homography(feat1, feat2, [])
        assert res["H"] is None
        assert res["success"] is False
        assert "failure_reason" in res

    def test_timing_recorded(self):
        feat1, feat2, matches, _ = _synthetic_feature_pair()
        res = estimate_homography(feat1, feat2, matches)
        assert res["time_s"] >= 0.0


class TestHomographyDiagnostics:
    def test_identity_not_degenerate(self):
        H = np.eye(3, dtype=np.float64)
        diag = diagnose_homography(H, (200, 200), (200, 200))
        assert diag["degenerate"] is False

    def test_none_homography_is_degenerate(self):
        diag = diagnose_homography(None, (200, 200), (200, 200))
        assert diag["degenerate"] is True

    def test_zero_matrix_is_degenerate(self):
        H = np.zeros((3, 3), dtype=np.float64)
        diag = diagnose_homography(H, (200, 200), (200, 200))
        assert diag["degenerate"] is True
