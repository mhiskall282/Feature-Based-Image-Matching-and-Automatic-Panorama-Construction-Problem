"""
src/features.py
===============
Feature detection and descriptor computation.

Pipeline stages:
  Feature Detection  (REQ-03)
  Feature Description (REQ-04)

Implements SIFT and ORB via a common factory interface so additional
methods can be added without restructuring the codebase.
"""

import time
import logging
import cv2
import numpy as np
from src.config import SIFT_PARAMS, ORB_PARAMS, SUPPORTED_METHODS

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Detector factory
# REQ-03: Detect distinctive keypoints using an appropriate feature detector
# ─────────────────────────────────────────────────────────────

def create_detector(method: str, params: dict | None = None) -> cv2.Feature2D:
    """
    Factory that returns an OpenCV Feature2D detector/descriptor for the
    given method name.

    Args:
        method: "SIFT" or "ORB"  (case-insensitive)
        params: Override dict for detector parameters. If None, uses config defaults.

    Returns:
        cv2.Feature2D object that supports detectAndCompute().

    Raises:
        ValueError: if method is not supported.

    Design note: This factory makes it trivial to add a third method (e.g., AKAZE,
    BRISK) without touching any other module.
    """
    method = method.upper()
    if method not in SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported method: '{method}'. "
            f"Choose from: {SUPPORTED_METHODS}"
        )

    if method == "SIFT":
        p = {**SIFT_PARAMS, **(params or {})}
        detector = cv2.SIFT_create(
            nfeatures=p["nfeatures"],
            nOctaveLayers=p["nOctaveLayers"],
            contrastThreshold=p["contrastThreshold"],
            edgeThreshold=p["edgeThreshold"],
            sigma=p["sigma"],
        )
        logger.debug(f"Created SIFT detector: {p}")

    elif method == "ORB":
        p = {**ORB_PARAMS, **(params or {})}
        detector = cv2.ORB_create(
            nfeatures=p["nfeatures"],
            scaleFactor=p["scaleFactor"],
            nlevels=p["nlevels"],
            edgeThreshold=p["edgeThreshold"],
            patchSize=p["patchSize"],
            fastThreshold=p["fastThreshold"],
        )
        logger.debug(f"Created ORB detector: {p}")

    return detector


# ─────────────────────────────────────────────────────────────
# Detection + Description
# REQ-04: Compute descriptors
# ─────────────────────────────────────────────────────────────

def detect_and_describe(img_gray: np.ndarray,
                        method: str,
                        params: dict | None = None) -> dict:
    """
    Detect keypoints and compute descriptors in one call.

    Args:
        img_gray: Single-channel uint8 grayscale image.
        method:   "SIFT" or "ORB".
        params:   Optional parameter overrides for the detector.

    Returns:
        dict with keys:
            keypoints   – list of cv2.KeyPoint
            descriptors – np.ndarray (float32 for SIFT, uint8 for ORB)
            method      – method name
            num_kp      – int: number of keypoints detected
            desc_shape  – tuple: descriptor array shape
            desc_dtype  – str: descriptor dtype
            time_s      – float: detection + description time (seconds)

    Raises:
        ValueError: if image has no features (descriptors is None).

    # REQ-03: Detect distinctive keypoints
    # REQ-04: Compute descriptors
    """
    if img_gray is None or img_gray.size == 0:
        raise ValueError("Input image is empty or None.")

    detector = create_detector(method, params)

    t0 = time.perf_counter()
    keypoints, descriptors = detector.detectAndCompute(img_gray, None)
    elapsed = time.perf_counter() - t0

    if descriptors is None or len(keypoints) == 0:
        logger.warning(f"[{method}] No features detected. Image may be textureless or too uniform.")
        return {
            "keypoints":   [],
            "descriptors": None,
            "method":      method,
            "num_kp":      0,
            "desc_shape":  None,
            "desc_dtype":  None,
            "time_s":      elapsed,
        }

    logger.info(
        f"[{method}] {len(keypoints)} keypoints | "
        f"desc={descriptors.shape} dtype={descriptors.dtype} | "
        f"{elapsed:.3f}s"
    )

    return {
        "keypoints":   keypoints,
        "descriptors": descriptors,
        "method":      method,
        "num_kp":      len(keypoints),
        "desc_shape":  descriptors.shape,
        "desc_dtype":  str(descriptors.dtype),
        "time_s":      elapsed,
    }


def describe_image_pair(img1_gray: np.ndarray,
                        img2_gray: np.ndarray,
                        method: str,
                        params: dict | None = None) -> tuple[dict, dict]:
    """
    Convenience wrapper: detect and describe both images.

    Returns:
        (result1, result2) — each is the dict from detect_and_describe().
    """
    r1 = detect_and_describe(img1_gray, method, params)
    r2 = detect_and_describe(img2_gray, method, params)
    return r1, r2
