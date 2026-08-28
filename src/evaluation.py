"""
src/evaluation.py
=================
Quantitative metrics computation and results recording.

Computes all 8 metrics required by the evaluation framework (M1–M8):
  M1 Detected keypoints
  M2 Initial matches
  M3 RANSAC inliers
  M4 Inlier ratio
  M5 Processing time (per stage)
  M6 Panorama quality
  M7 Reprojection error
  M8 Homography success

Also handles CSV and JSON logging of experiment results.
"""

import json
import logging
import time
import csv
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Metric computation
# ─────────────────────────────────────────────────────────────

def compute_metrics(feat1: dict, feat2: dict,
                    match_result: dict,
                    ransac_result: dict,
                    stitch_info: dict | None = None,
                    panorama: np.ndarray | None = None) -> dict:
    """
    Assemble all quantitative metrics into a single flat dict.

    Args:
        feat1, feat2:   Feature dicts from detect_and_describe()
        match_result:   Dict from match_descriptors()
        ransac_result:  Dict from estimate_homography()
        stitch_info:    Dict from stitch_images() (optional)
        panorama:       Final panorama array (optional, for quality metrics)

    Returns:
        Flat dict of all metrics (suitable for CSV row).
    """
    metrics = {
        # M1 – Detected keypoints
        "num_kp_img1":        feat1.get("num_kp", 0),
        "num_kp_img2":        feat2.get("num_kp", 0),
        "desc_dim_img1":      feat1.get("desc_shape", [None, None])[1] if feat1.get("desc_shape") else None,
        "desc_dtype":         feat1.get("desc_dtype", ""),

        # M2 – Initial / good matches
        "num_raw_matches":    match_result.get("num_raw_matches", 0),
        "num_good_matches":   match_result.get("num_good_matches", 0),

        # M3 – RANSAC inliers
        "num_ransac_inliers": ransac_result.get("num_inliers", 0),
        "num_ransac_outliers":ransac_result.get("num_outliers", 0),

        # M4 – Inlier ratio
        "inlier_ratio":       round(ransac_result.get("inlier_ratio", 0.0), 4),

        # M5 – Timing
        "time_feat_s":        round(feat1.get("time_s",0) + feat2.get("time_s",0), 4),
        "time_match_s":       round(match_result.get("time_s", 0.0), 4),
        "time_ransac_s":      round(ransac_result.get("time_s", 0.0), 4),

        # M7 – Reprojection error
        "reprojection_error_px": round(ransac_result.get("reprojection_error", float("inf")), 3),

        # M8 – Homography success
        "homography_success": ransac_result.get("success", False),
        "failure_reason":     ransac_result.get("failure_reason", ""),
    }

    # Stitch info
    if isinstance(stitch_info, dict):
        metrics["panorama_w"] = stitch_info.get("panorama_w", 0)
        metrics["panorama_h"] = stitch_info.get("panorama_h", 0)

    # M6 – Panorama quality
    if panorama is not None:
        metrics.update(assess_panorama_quality(panorama))

    return metrics


def assess_panorama_quality(panorama: np.ndarray) -> dict:
    """
    Compute quantitative panorama quality metrics.

    valid_area_ratio: fraction of canvas covered (non-black pixels)
    mean_gradient:   mean gradient magnitude (proxy for sharpness)
    """
    gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY) if panorama.ndim == 3 else panorama
    total_pixels = gray.size

    # Valid pixel coverage
    non_black = np.sum(gray > 1)
    valid_ratio = non_black / total_pixels if total_pixels > 0 else 0.0

    # Laplacian variance (sharpness proxy)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = float(lap.var())

    return {
        "pano_valid_ratio":  round(valid_ratio, 4),
        "pano_lap_variance": round(lap_var, 2),
    }


# ─────────────────────────────────────────────────────────────
# Results I/O
# ─────────────────────────────────────────────────────────────

CSV_FIELDS = [
    "timestamp", "experiment", "method", "image_pair", "condition",
    "num_kp_img1", "num_kp_img2", "desc_dim_img1", "desc_dtype",
    "num_raw_matches", "num_good_matches", "num_ransac_inliers", "num_ransac_outliers",
    "inlier_ratio", "time_feat_s", "time_match_s", "time_ransac_s",
    "reprojection_error_px", "homography_success", "failure_reason",
    "panorama_w", "panorama_h", "pano_valid_ratio", "pano_lap_variance",
    "time_total_s", "ratio_threshold", "ransac_threshold",
    "rotation_angle_deg", "scale_factor", "viewpoint_level", "viewpoint_offset_px",
    "illumination_tag", "brightness_delta", "contrast_factor", "illumination_type",
]


def build_result_row(experiment: str, method: str,
                     image_pair: str, condition: str,
                     metrics: dict, extra: dict | None = None) -> dict:
    """Build a complete CSV row dict from metrics + experiment metadata."""
    row = {f: None for f in CSV_FIELDS}
    row.update({
        "timestamp":   datetime.now(timezone.utc).isoformat(),
        "experiment":  experiment,
        "method":      method,
        "image_pair":  image_pair,
        "condition":   condition,
    })
    row.update(metrics)
    if extra:
        row.update(extra)
    return row


def append_result_csv(row: dict, csv_path: str | Path) -> None:
    """Append one result row to a CSV file (creates header if new file)."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    is_new = not csv_path.exists()
    row_aligned = {f: row.get(f, "") for f in CSV_FIELDS}
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if is_new:
            writer.writeheader()
        writer.writerow(row_aligned)


def log_failure(experiment: str, method: str, image_pair: str,
                stage: str, reason: str,
                failures_csv: str | Path) -> None:
    """Log a failure to results/logs/failures.csv."""
    row = {
        "timestamp":   datetime.now(timezone.utc).isoformat(),
        "experiment":  experiment,
        "algorithm":   method,
        "image_pair":  image_pair,
        "failure_stage": stage,
        "failure_reason": reason,
    }
    append_result_csv(row, failures_csv)
    logger.warning(f"[FAILURE] {experiment}/{method}/{image_pair}: {stage} — {reason}")


def save_experiment_config(config: dict, out_path: str | Path) -> None:
    """Save experiment config as JSON (RULE-REP03)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, default=str)


def get_software_versions() -> dict:
    """Return current library versions for reproducibility (RULE-REP02)."""
    import sys
    import matplotlib
    try:
        import skimage
        sk_ver = skimage.__version__
    except ImportError:
        sk_ver = "not installed"
    return {
        "python":     sys.version,
        "opencv":     cv2.__version__,
        "numpy":      np.__version__,
        "matplotlib": matplotlib.__version__,
        "skimage":    sk_ver,
    }
