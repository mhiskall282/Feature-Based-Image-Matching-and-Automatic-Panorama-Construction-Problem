"""
src/homography.py
=================
RANSAC-based homography estimation and diagnostics.

Pipeline stages:
  RANSAC (REQ-07)
  Homography Estimation (REQ-08)

The homography H maps points from image 1 into image 2's coordinate frame:
    p' ~ H * p   (in homogeneous coordinates)

Homogeneous coordinates are used because the projective transformation
involves division by the third coordinate w, which cannot be expressed
as a pure linear (affine) transformation in Euclidean coordinates.
"""

import time
import json
import logging
import numpy as np
import cv2
from pathlib import Path
from src.config import (RANSAC_REPROJ_THRESHOLD, RANSAC_CONFIDENCE,
                        RANSAC_MAX_ITERS, MIN_RANSAC_INLIERS)
from src.matching import extract_point_pairs

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# RANSAC + Homography
# REQ-07: Apply RANSAC to eliminate incorrect correspondences
# REQ-08: Estimate the homography matrix
# ─────────────────────────────────────────────────────────────

def estimate_homography(feat1: dict, feat2: dict, matches: list,
                        reproj_threshold: float = RANSAC_REPROJ_THRESHOLD,
                        confidence: float = RANSAC_CONFIDENCE,
                        max_iters: int = RANSAC_MAX_ITERS,
                        min_inliers: int = MIN_RANSAC_INLIERS) -> dict:
    """
    Estimate a 3×3 homography H using RANSAC to robustly reject outliers.

    RANSAC algorithm (Fischler & Bolles 1981):
      1. Randomly sample 4 point correspondences (minimum for homography: 8 DOF / 2 eq/pt = 4 pts)
      2. Compute candidate H from those 4 points
      3. Count inliers: pairs where reprojection_error < reproj_threshold
      4. Repeat for max_iters iterations
      5. Return H with the most inliers; refine using all inliers

    An inlier is a point pair whose reprojection error — the pixel distance
    between the actual dst point and H(src point) — is below the threshold.
    An outlier is any point pair exceeding that threshold.

    Args:
        feat1, feat2:       Feature dicts from detect_and_describe()
        matches:            Good matches from match_descriptors()
        reproj_threshold:   Maximum pixel reprojection error for inliers (default 5px)
        confidence:         Desired probability that result is correct (default 0.995)
        max_iters:          Maximum RANSAC iterations (default 2000)
        min_inliers:        Minimum inlier count to accept result (default 10)

    Returns:
        dict with keys:
            H                 – 3×3 float64 homography matrix (or None if failed)
            mask              – np.ndarray bool mask: inlier=1, outlier=0
            inlier_matches    – list[cv2.DMatch]: inlier subset
            outlier_matches   – list[cv2.DMatch]: outlier subset
            num_inliers       – int
            num_outliers      – int
            inlier_ratio      – float  (inliers / total good matches)
            reprojection_error – float: mean pixel error over inliers
            time_s            – float: RANSAC time
            success           – bool
            failure_reason    – str or None

    # REQ-07: Apply RANSAC to eliminate incorrect correspondences
    # REQ-08: Estimate the homography matrix
    """
    kp1 = feat1["keypoints"]
    kp2 = feat2["keypoints"]
    method = feat1["method"]

    result_base = {
        "H": None, "mask": None,
        "inlier_matches": [], "outlier_matches": matches,
        "num_inliers": 0, "num_outliers": len(matches),
        "inlier_ratio": 0.0, "reprojection_error": float("inf"),
        "time_s": 0.0, "success": False, "failure_reason": None,
    }

    if len(matches) < 4:
        result_base["failure_reason"] = f"insufficient_matches (n={len(matches)}, need ≥4)"
        logger.warning(f"[{method}] RANSAC skipped — only {len(matches)} matches, need ≥4.")
        return result_base

    src_pts, dst_pts = extract_point_pairs(kp1, kp2, matches)

    t0 = time.perf_counter()
    H, mask = cv2.findHomography(
        src_pts, dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=reproj_threshold,
        confidence=confidence,
        maxIters=max_iters,
    )
    elapsed = time.perf_counter() - t0

    result_base["time_s"] = elapsed

    if H is None or mask is None:
        result_base["failure_reason"] = "findHomography_returned_none"
        logger.warning(f"[{method}] cv2.findHomography returned None.")
        return result_base

    mask_flat = mask.ravel().astype(bool)
    inlier_m  = [m for m, f in zip(matches, mask_flat) if f]
    outlier_m = [m for m, f in zip(matches, mask_flat) if not f]
    n_in      = len(inlier_m)
    n_out     = len(outlier_m)
    ratio     = n_in / len(matches) if matches else 0.0

    # Compute mean reprojection error over inliers
    reproj_err = _reprojection_error(src_pts, dst_pts, mask_flat, H)

    success = n_in >= min_inliers
    if not success:
        failure_reason = f"insufficient_inliers (n={n_in}, need ≥{min_inliers})"
    else:
        failure_reason = None

    logger.info(
        f"[{method}] RANSAC: {n_in} inliers / {len(matches)} matches "
        f"({ratio:.1%}) | reproj_err={reproj_err:.2f}px | {elapsed:.3f}s | "
        f"{'OK' if success else 'FAILED'}"
    )

    return {
        "H":                  H if success else None,
        "mask":               mask,
        "inlier_matches":     inlier_m,
        "outlier_matches":    outlier_m,
        "num_inliers":        n_in,
        "num_outliers":       n_out,
        "inlier_ratio":       ratio,
        "reprojection_error": reproj_err,
        "time_s":             elapsed,
        "success":            success,
        "failure_reason":     failure_reason,
    }


def _reprojection_error(src_pts: np.ndarray, dst_pts: np.ndarray,
                        mask: np.ndarray, H: np.ndarray) -> float:
    """Compute mean reprojection error (in pixels) for inlier points."""
    in_src = src_pts[mask]
    in_dst = dst_pts[mask]
    if len(in_src) == 0:
        return float("inf")
    projected = cv2.perspectiveTransform(in_src, H)
    errors = np.linalg.norm(in_dst - projected, axis=2).ravel()
    return float(np.mean(errors))


# ─────────────────────────────────────────────────────────────
# Homography diagnostics
# ─────────────────────────────────────────────────────────────

def diagnose_homography(H: np.ndarray,
                        img1_shape: tuple,
                        img2_shape: tuple) -> dict:
    """
    Diagnostic checks on the estimated homography:
      - Determinant (degenerate if near 0 or very large)
      - Condition number (ill-conditioned if very large)
      - Corner mapping (do corners of img1 map inside reasonable bounds of img2?)

    Returns dict of diagnostic values and a 'degenerate' flag.
    """
    if H is None:
        return {"degenerate": True, "reason": "H is None"}

    det  = float(np.linalg.det(H))
    cond = float(np.linalg.cond(H))

    h1, w1 = img1_shape[:2]
    corners = np.float32([[0,0],[w1,0],[w1,h1],[0,h1]]).reshape(-1,1,2)
    mapped  = cv2.perspectiveTransform(corners, H).reshape(-1, 2)

    h2, w2 = img2_shape[:2]
    margin  = max(h2, w2) * 5.0
    in_bounds = all(
        -margin <= x <= w2 + margin and -margin <= y <= h2 + margin
        for x, y in mapped
    )

    degenerate = abs(det) < 1e-6 or abs(det) > 1e6 or cond > 1e8 or not in_bounds

    diag = {
        "determinant":         det,
        "condition_number":    cond,
        "mapped_corners":      mapped.tolist(),
        "corners_in_bounds":   in_bounds,
        "degenerate":          degenerate,
    }
    if degenerate:
        reasons = []
        if abs(det) < 1e-6:     reasons.append("near-zero determinant")
        if abs(det) > 1e6:      reasons.append("very large determinant")
        if cond > 1e8:          reasons.append("ill-conditioned matrix")
        if not in_bounds:       reasons.append("corners map outside reasonable bounds")
        diag["reason"] = "; ".join(reasons)

    return diag


# ─────────────────────────────────────────────────────────────
# Saving homography
# ─────────────────────────────────────────────────────────────

def save_homography(H: np.ndarray, out_dir: Path,
                    prefix: str = "H", metadata: dict | None = None) -> None:
    """
    Save the homography matrix in multiple formats:
      - {prefix}.txt   (human-readable)
      - {prefix}.npy   (NumPy binary)
      - {prefix}.json  (machine-readable)

    # REQ-08 / instruction §13: Save matrices as JSON, CSV, NumPy.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if H is None:
        logger.warning(f"save_homography: H is None, skipping save at {out_dir}/{prefix}")
        return

    # Text
    txt_path = out_dir / f"{prefix}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("# Homography Matrix H (3x3)\n")
        f.write("# Maps points from image 1 -> image 2 coordinate frame\n")
        f.write("# x' ~ H*x  (homogeneous coordinates)\n")
        if metadata:
            for k, v in metadata.items():
                f.write(f"# {k}: {v}\n")
        f.write("#\n")
        np.savetxt(f, H, fmt="%.10f")

    # NumPy
    np.save(out_dir / f"{prefix}.npy", H)

    # JSON
    json_data = {"H": H.tolist()}
    if metadata:
        json_data.update(metadata)
    with open(out_dir / f"{prefix}.json", "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)

    logger.debug(f"Saved homography to {out_dir}/{prefix}.*")
