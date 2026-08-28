"""
src/matching.py
===============
Descriptor matching between image pairs.

Pipeline stage: Descriptor Matching → Initial Correspondences
REQ-05: Match descriptors between overlapping image pairs.
REQ-06: Display initial feature correspondences.

SIFT  → BFMatcher(NORM_L2) + kNN(k=2) + Lowe's ratio test
ORB   → BFMatcher(NORM_HAMMING, crossCheck=True)

RULE-PY06: Distance metric is always matched to descriptor type.
"""

import time
import logging
import cv2
import numpy as np
from src.config import METHOD_NORMS, RATIO_THRESHOLD, CROSS_CHECK

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Matcher factory
# ─────────────────────────────────────────────────────────────

def create_matcher(method: str) -> cv2.DescriptorMatcher:
    """
    Return a BFMatcher appropriate for the given descriptor type.

    SIFT: NORM_L2, no cross-check (we apply ratio test via kNN)
    ORB:  NORM_HAMMING, cross-check enabled

    This enforces the rule that float and binary descriptors must never
    be compared with the wrong distance metric.

    # RULE-PY06: Correct distance metric per descriptor type.
    """
    norm = METHOD_NORMS.get(method.upper())
    if norm is None:
        raise ValueError(f"Unknown method '{method}'. Add to METHOD_NORMS in config.py.")

    if method.upper() == "SIFT":
        return cv2.BFMatcher(norm, crossCheck=False)
    else:  # ORB and binary descriptors
        return cv2.BFMatcher(norm, crossCheck=CROSS_CHECK)


# ─────────────────────────────────────────────────────────────
# Ratio test (Lowe 2004)
# ─────────────────────────────────────────────────────────────

def apply_ratio_test(knn_matches: list, threshold: float = RATIO_THRESHOLD) -> list:
    """
    Lowe's ratio test for float-descriptor matches (SIFT).

    Accept a match only when:
        best_match.distance < threshold * second_best.distance

    Rationale: Ambiguous matches — where two descriptors are nearly equally close —
    are likely incorrect correspondences. A threshold of 0.75 (Lowe 2004) rejects
    most ambiguous matches while retaining reliable ones.

    Args:
        knn_matches: Output of BFMatcher.knnMatch(desc1, desc2, k=2)
        threshold:   Ratio threshold (default 0.75)

    Returns:
        List of good cv2.DMatch objects.
    """
    good = []
    for pair in knn_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < threshold * n.distance:
                good.append(m)
    return good


# ─────────────────────────────────────────────────────────────
# Main matching function
# REQ-05: Match descriptors between overlapping image pairs
# ─────────────────────────────────────────────────────────────

def match_descriptors(feat1: dict, feat2: dict,
                      ratio_threshold: float = RATIO_THRESHOLD) -> dict:
    """
    Match descriptors from two images using the appropriate strategy.

    SIFT path:
        1. BFMatcher(NORM_L2) kNN(k=2) matching
        2. Lowe's ratio test to filter ambiguous matches

    ORB path:
        1. BFMatcher(NORM_HAMMING, crossCheck=True) single match
        2. Sort by distance

    Args:
        feat1, feat2:      Feature dicts from detect_and_describe()
        ratio_threshold:   Ratio test threshold (SIFT only)

    Returns:
        dict with keys:
            method            – str
            good_matches      – list[cv2.DMatch]: filtered matches
            num_raw_matches   – int: total kNN/cross-check matches before filtering
            num_good_matches  – int: after filtering
            time_s            – float: matching time

    # REQ-05: Match descriptors between overlapping image pairs
    """
    method = feat1["method"].upper()
    desc1  = feat1["descriptors"]
    desc2  = feat2["descriptors"]

    # Guard: no descriptors
    if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
        logger.warning(f"[{method}] Cannot match: one or both descriptor arrays are empty.")
        return {
            "method": method,
            "good_matches": [],
            "num_raw_matches": 0,
            "num_good_matches": 0,
            "time_s": 0.0,
            "error": "empty_descriptors",
        }

    matcher = create_matcher(method)
    t0 = time.perf_counter()

    if method == "SIFT":
        # kNN match then ratio test
        knn = matcher.knnMatch(desc1, desc2, k=2)
        raw_count   = len(knn)
        good        = apply_ratio_test(knn, ratio_threshold)

    else:  # ORB (binary, cross-check)
        raw         = matcher.match(desc1, desc2)
        raw_count   = len(raw)
        good        = sorted(raw, key=lambda m: m.distance)

    elapsed = time.perf_counter() - t0

    logger.info(
        f"[{method}] raw={raw_count} → good={len(good)} | "
        f"ratio_threshold={ratio_threshold:.2f} | {elapsed:.3f}s"
    )

    return {
        "method":           method,
        "good_matches":     good,
        "num_raw_matches":  raw_count,
        "num_good_matches": len(good),
        "time_s":           elapsed,
    }


# ─────────────────────────────────────────────────────────────
# Extract matched point arrays for RANSAC
# ─────────────────────────────────────────────────────────────

def extract_point_pairs(kp1: list, kp2: list,
                        matches: list) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract (x,y) coordinates from matched keypoints.

    Returns:
        (src_pts, dst_pts) — float32 arrays shaped (N,1,2)
        Required by cv2.findHomography().
    """
    src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    return src, dst
