"""
src/warping.py
==============
Image warping using estimated homographies.

Pipeline stage: Image Warping (REQ-09)
  Transforms one image into another image's coordinate system using
  cv2.warpPerspective (inverse mapping with bilinear interpolation).

Key geometry:
  The output canvas must be large enough to contain both images.
  A translation offset T is prepended to H when the warped image
  extends into negative coordinates.
"""

import logging
import cv2
import numpy as np
from src.config import WARP_INTERPOLATION

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Canvas geometry
# REQ-09: Transform one image into another image's coordinate system
# ─────────────────────────────────────────────────────────────

def compute_canvas(img_src: np.ndarray,
                   img_ref: np.ndarray,
                   H: np.ndarray) -> tuple[int, int, int, int]:
    """
    Compute the canvas (width, height) and translation offset (x_off, y_off)
    needed to contain both the reference image and the warped source image.

    Algorithm:
      1. Map all four corners of img_src through H
      2. Include all four corners of img_ref (which stays fixed at origin)
      3. Take the bounding box of the union
      4. offset = -min(x, y) if negative, else 0

    Args:
        img_src: source image to be warped
        img_ref: reference image (stays fixed)
        H:       3×3 homography mapping src → ref frame

    Returns:
        (canvas_w, canvas_h, x_off, y_off)
    """
    h_s, w_s = img_src.shape[:2]
    h_r, w_r = img_ref.shape[:2]

    # Corners of source image
    corners_src = np.float32([[0,0],[w_s,0],[w_s,h_s],[0,h_s]]).reshape(-1,1,2)
    # Map through H
    corners_warped = cv2.perspectiveTransform(corners_src, H)

    # Corners of reference image (at origin)
    corners_ref = np.float32([[0,0],[w_r,0],[w_r,h_r],[0,h_r]]).reshape(-1,1,2)

    all_corners = np.concatenate([corners_ref, corners_warped], axis=0)

    x_min = int(np.floor(all_corners[:,:,0].min()))
    y_min = int(np.floor(all_corners[:,:,1].min()))
    x_max = int(np.ceil(all_corners[:,:,0].max()))
    y_max = int(np.ceil(all_corners[:,:,1].max()))

    x_off = -x_min if x_min < 0 else 0
    y_off = -y_min if y_min < 0 else 0

    canvas_w = x_max - x_min
    canvas_h = y_max - y_min

    logger.debug(f"Canvas: {canvas_w}×{canvas_h}  offset=({x_off},{y_off})")
    return canvas_w, canvas_h, x_off, y_off


def warp_image(img: np.ndarray,
               H: np.ndarray,
               canvas_w: int, canvas_h: int,
               x_off: int, y_off: int,
               interpolation: int = WARP_INTERPOLATION) -> np.ndarray:
    """
    Warp img onto the canvas using the adjusted homography H.

    The translation offset is prepended to H so that negative coordinates
    in the warped result are shifted into the positive canvas region.

    Uses inverse mapping (warpPerspective default):
      For each output pixel, compute the corresponding input pixel location
      and sample with bilinear interpolation.

    # REQ-09: Transform one image into another image's coordinate system
    """
    # Translation matrix to handle offset
    T = np.array([[1, 0, x_off],
                  [0, 1, y_off],
                  [0, 0, 1    ]], dtype=np.float64)
    H_adj = T @ H

    warped = cv2.warpPerspective(
        img, H_adj, (canvas_w, canvas_h),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return warped


def place_reference(img_ref: np.ndarray,
                    canvas_w: int, canvas_h: int,
                    x_off: int, y_off: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Place the reference image on a black canvas at (x_off, y_off).

    Returns:
        (canvas_bgr, canvas_mask)  — the placed image and a binary validity mask.
    """
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    h_r, w_r = img_ref.shape[:2]
    canvas[y_off:y_off+h_r, x_off:x_off+w_r] = img_ref

    mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    mask[y_off:y_off+h_r, x_off:x_off+w_r] = 255

    return canvas, mask


# ─────────────────────────────────────────────────────────────
# Blending
# ─────────────────────────────────────────────────────────────

def blend_images(base: np.ndarray, overlay: np.ndarray,
                 base_mask: np.ndarray, overlay_mask: np.ndarray,
                 alpha_blend: bool = True) -> np.ndarray:
    """
    Composite overlay onto base.

    Modes:
      alpha_blend=True  → distance-weighted alpha blend in overlap region
      alpha_blend=False → simple overwrite: base pixels have priority

    Both modes:
      - Regions only in base  → base pixel
      - Regions only in overlay → overlay pixel
      - Overlap region → blended or base-priority

    Returns merged BGR image.
    """
    result = base.copy()

    overlay_only = (base_mask == 0) & (overlay_mask > 0)
    result[overlay_only] = overlay[overlay_only]

    if alpha_blend:
        overlap = (base_mask > 0) & (overlay_mask > 0)
        if overlap.any():
            # Distance-transform weights
            d1 = cv2.distanceTransform(base_mask.astype(np.uint8),    cv2.DIST_L2, 5).astype(np.float64)
            d2 = cv2.distanceTransform(overlay_mask.astype(np.uint8), cv2.DIST_L2, 5).astype(np.float64)
            total = d1 + d2
            total[total == 0] = 1.0
            w1 = (d1 / total)[..., np.newaxis]  # shape (H,W,1)
            w2 = (d2 / total)[..., np.newaxis]

            blended = (w1 * base.astype(np.float32) +
                       w2 * overlay.astype(np.float32)).clip(0, 255).astype(np.uint8)
            result[overlap] = blended[overlap]

    return result


def crop_black_borders(img: np.ndarray) -> np.ndarray:
    """Remove black border rows/columns from a panorama image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return img[y:y+h, x:x+w]
