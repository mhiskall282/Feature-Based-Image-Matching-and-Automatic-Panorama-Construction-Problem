"""
src/preprocessing.py
====================
Image loading, validation, and preprocessing.

Pipeline stage: Input Images → Image Preparation
REQ-02: Perform necessary image preparation.

Every preprocessing step is explicitly documented with its justification.
"""

import logging
import cv2
import numpy as np
from pathlib import Path
from src.config import MAX_IMAGE_DIMENSION, RESIZE_INTERPOLATION

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────

def load_image(path: str | Path) -> np.ndarray:
    """
    Load an image from disk.

    Uses cv2.imread which returns BGR format.
    Raises FileNotFoundError if path is invalid or image cannot be decoded.

    # RULE-PY04: Every image load must be validated.
    """
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(
            f"Failed to load image: {path}\n"
            "Check that the file exists and is a valid image format (JPG/PNG/BMP)."
        )
    logger.info(f"Loaded image: {path}  shape={img.shape}")
    return img


def load_image_set(directory: str | Path,
                   extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp")) -> list[tuple]:
    """
    Load all images from a directory in sorted order.

    Returns:
        List of (path, img_BGR) tuples, sorted by filename.

    # REQ-01: Acquire at least three overlapping images.
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Image directory not found: {directory}")

    paths = sorted([p for p in directory.iterdir()
                    if p.suffix.lower() in extensions])

    if len(paths) < 2:
        raise ValueError(
            f"Need at least 2 images in {directory}, found {len(paths)}.\n"
            "Add overlapping images (JPG/PNG) to the directory."
        )

    images = []
    for p in paths:
        img = load_image(p)
        images.append((p, img))

    logger.info(f"Loaded {len(images)} images from {directory}")
    return images


# ─────────────────────────────────────────────────────────────
# Preprocessing pipeline
# ─────────────────────────────────────────────────────────────

def resize_if_needed(img: np.ndarray,
                     max_dim: int = MAX_IMAGE_DIMENSION) -> np.ndarray:
    """
    Resize image if its largest dimension exceeds max_dim, preserving aspect ratio.

    Justification: Very large images slow down all pipeline stages significantly.
    Resizing to ≤1280px on the largest side balances quality with performance.
    """
    h, w = img.shape[:2]
    largest = max(h, w)
    if largest <= max_dim:
        return img
    scale = max_dim / largest
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=RESIZE_INTERPOLATION)
    logger.debug(f"Resized: {w}×{h} → {new_w}×{new_h}")
    return resized


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """
    Convert BGR image to single-channel grayscale.

    Justification: SIFT and ORB operate on intensity (single-channel) images.
    Grayscale reduces data volume and focuses the detector on luminance structure.
    Formula: Y = 0.114·B + 0.587·G + 0.299·R  (OpenCV's default weighting).
    """
    if len(img.shape) == 2:
        return img  # Already grayscale
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def equalize_histogram(gray: np.ndarray, use_clahe: bool = True) -> np.ndarray:
    """
    Apply histogram equalization to improve contrast in low-contrast images.

    Justification: Used in the illumination experiment to normalize brightness
    differences before feature detection, as a preprocessing mitigation strategy.

    CLAHE (Contrast Limited Adaptive Histogram Equalization) is preferred over
    global equalization because it avoids over-amplification in uniform regions.
    """
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)
    return cv2.equalizeHist(gray)


def preprocess(img_bgr: np.ndarray,
               max_dim: int = MAX_IMAGE_DIMENSION,
               equalize: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply the full preprocessing pipeline.

    Steps (in order):
      1. Resize if necessary                  (speed)
      2. Convert to grayscale                 (feature detection requires single channel)
      3. Optional histogram equalization       (illumination robustness, explicitly labeled)

    Returns:
        (img_color_resized, img_gray) — colour copy for stitching, gray for detection.

    # REQ-02: Perform necessary image preparation.
    """
    color = resize_if_needed(img_bgr, max_dim)
    gray  = to_grayscale(color)
    if equalize:
        gray = equalize_histogram(gray)
    return color, gray


def validate_image_pair(img1: np.ndarray, img2: np.ndarray) -> None:
    """
    Basic validation that two images are suitable for matching.
    Raises ValueError with a descriptive message if images are unsuitable.
    """
    for i, img in enumerate([img1, img2], 1):
        if img is None:
            raise ValueError(f"Image {i} is None.")
        if img.size == 0:
            raise ValueError(f"Image {i} is empty (zero size).")
        h, w = img.shape[:2]
        if h < 100 or w < 100:
            raise ValueError(
                f"Image {i} is too small ({w}×{h}). "
                "Feature detectors need at least 100×100 pixels."
            )
