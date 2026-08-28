"""
tests/test_pipeline.py
Integration test for the full pipeline using synthetic images.
Tests that no crashes occur and key result keys are present.
"""
import sys
from pathlib import Path
import numpy as np
import cv2
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.pipeline import run_pair_pipeline


def make_synthetic_pair():
    """
    Synthetic image pair: checkerboard + small translation.
    Sufficient to test that the pipeline runs without error.
    """
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    sq = 40
    for r in range(0, 300, sq * 2):
        for c in range(0, 400, sq * 2):
            img[r:r+sq, c:c+sq] = (200, 150, 100)
            img[r+sq:r+sq*2, c+sq:c+sq*2] = (200, 150, 100)
    img2 = np.roll(img, 30, axis=1)  # Shift right by 30px
    return img, img2


@pytest.fixture
def synthetic_pair():
    return make_synthetic_pair()


@pytest.fixture
def tmp_out(tmp_path):
    return tmp_path / "test_output"


class TestPipelineIntegration:
    def test_pipeline_runs_sift(self, synthetic_pair, tmp_out):
        img1, img2 = synthetic_pair
        row = run_pair_pipeline(img1, img2, method="SIFT",
                                out_dir=tmp_out, save_viz=False,
                                experiment="test", pair_name="t1-t2")
        assert "num_kp_img1" in row
        assert "num_good_matches" in row
        assert "inlier_ratio" in row
        assert "homography_success" in row

    def test_pipeline_runs_orb(self, synthetic_pair, tmp_out):
        img1, img2 = synthetic_pair
        row = run_pair_pipeline(img1, img2, method="ORB",
                                out_dir=tmp_out, save_viz=False,
                                experiment="test", pair_name="t1-t2")
        assert "num_kp_img1" in row

    def test_pipeline_handles_failure_gracefully(self, tmp_out):
        """Very dark images → likely no matches — should not crash."""
        dark = np.zeros((300, 400, 3), dtype=np.uint8)
        row  = run_pair_pipeline(dark, dark, method="SIFT",
                                 out_dir=tmp_out, save_viz=False,
                                 experiment="test", pair_name="dark-dark")
        # Should return a row (possibly with homography_success=False), not raise
        assert isinstance(row, dict)

    def test_pipeline_inlier_ratio_in_range(self, synthetic_pair, tmp_out):
        img1, img2 = synthetic_pair
        row = run_pair_pipeline(img1, img2, method="SIFT",
                                out_dir=tmp_out, save_viz=False,
                                experiment="test", pair_name="ratio_test")
        ratio = row.get("inlier_ratio", 0.0)
        assert 0.0 <= ratio <= 1.0
