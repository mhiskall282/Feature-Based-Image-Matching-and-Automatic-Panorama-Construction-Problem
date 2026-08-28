"""
src/stitching.py
================
Multi-image panorama construction.

Pipeline stage: Image Alignment → Panorama Stitching (REQ-10, REQ-15)

Strategy: Reference-based stitching
  - Choose the centre image as the reference coordinate frame
  - Estimate H_i for every non-reference image → reference
  - Warp each image onto a shared canvas
  - Composite with alpha blending

This minimises maximum warp distance and cumulative error compared to
sequential (chain) stitching.
"""

import logging
import numpy as np
import cv2
from src.warping import (compute_canvas, warp_image, place_reference,
                         blend_images, crop_black_borders)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Multi-image panorama canvas computation
# ─────────────────────────────────────────────────────────────

def _compute_multi_canvas(images: list[np.ndarray],
                          H_list: list[np.ndarray | None]) -> tuple[int,int,int,int]:
    """
    Compute shared canvas for all images given their homographies to reference frame.

    H_list[i] maps images[i] → reference coordinate frame.
    H_list[ref_idx] = identity.
    """
    all_corners = []
    for img, H in zip(images, H_list):
        if H is None:
            continue
        h, w = img.shape[:2]
        corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
        mapped  = cv2.perspectiveTransform(corners, H)
        all_corners.append(mapped)

    if not all_corners:
        raise ValueError("No valid homographies — cannot compute canvas size.")

    all_corners = np.concatenate(all_corners, axis=0)
    x_min = int(np.floor(all_corners[:,:,0].min()))
    y_min = int(np.floor(all_corners[:,:,1].min()))
    x_max = int(np.ceil (all_corners[:,:,0].max()))
    y_max = int(np.ceil (all_corners[:,:,1].max()))

    x_off = -x_min if x_min < 0 else 0
    y_off = -y_min if y_min < 0 else 0
    canvas_w = x_max - x_min
    canvas_h = y_max - y_min

    return canvas_w, canvas_h, x_off, y_off


# ─────────────────────────────────────────────────────────────
# Two-image stitch (helper used by multi-image stitcher)
# ─────────────────────────────────────────────────────────────

def stitch_pair(img_src: np.ndarray,
                img_ref: np.ndarray,
                H: np.ndarray,
                alpha_blend: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Stitch two images: warp img_src into img_ref's frame and composite.

    Returns:
        (panorama_bgr, panorama_mask)
    """
    canvas_w, canvas_h, x_off, y_off = compute_canvas(img_src, img_ref, H)

    # Warp source
    warped     = warp_image(img_src, H, canvas_w, canvas_h, x_off, y_off)
    warp_gray  = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _, warp_mask = cv2.threshold(warp_gray, 1, 255, cv2.THRESH_BINARY)

    # Place reference
    ref_canvas, ref_mask = place_reference(img_ref, canvas_w, canvas_h, x_off, y_off)

    # Blend
    panorama = blend_images(ref_canvas, warped, ref_mask, warp_mask, alpha_blend)

    combined_mask = cv2.bitwise_or(ref_mask, warp_mask)
    info = {
        "canvas_w": canvas_w,
        "canvas_h": canvas_h,
        "panorama_w": panorama.shape[1],
        "panorama_h": panorama.shape[0],
        "mask": combined_mask,
    }
    return panorama, info


# ─────────────────────────────────────────────────────────────
# Main multi-image stitcher
# REQ-10: Stitch transformed images into a panorama
# REQ-15: Demonstrate complete pipeline end-to-end
# ─────────────────────────────────────────────────────────────

def stitch_images(images: list[np.ndarray],
                  H_list: list[np.ndarray | None],
                  ref_idx: int,
                  alpha_blend: bool = True,
                  crop_borders: bool = True) -> tuple[np.ndarray, dict]:
    """
    Construct a panorama from N images given their homographies to the reference frame.

    Args:
        images:      List of BGR colour images
        H_list:      Homography for each image → reference frame.
                     H_list[ref_idx] should be np.eye(3).
        ref_idx:     Index of the reference (fixed) image
        alpha_blend: Use distance-weighted alpha blending in overlap regions
        crop_borders: Remove black border regions from final panorama

    Returns:
        (panorama_bgr, info_dict)

    # REQ-10: Stitch transformed images into a panorama
    """
    if len(images) != len(H_list):
        raise ValueError("images and H_list must have the same length.")

    canvas_w, canvas_h, x_off, y_off = _compute_multi_canvas(images, H_list)
    logger.info(f"Panorama canvas: {canvas_w}×{canvas_h}  offset=({x_off},{y_off})")

    # Initialise empty canvas
    canvas      = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas_mask = np.zeros((canvas_h, canvas_w),    dtype=np.uint8)

    T = np.array([[1,0,x_off],[0,1,y_off],[0,0,1]], dtype=np.float64)

    skipped = []

    # Place all images — reference first (priority), then others
    order = [ref_idx] + [i for i in range(len(images)) if i != ref_idx]

    for i in order:
        img = images[i]
        H   = H_list[i]

        if H is None:
            logger.warning(f"Image {i}: no homography — skipping.")
            skipped.append(i)
            continue

        H_adj  = T @ H
        warped = cv2.warpPerspective(img, H_adj, (canvas_w, canvas_h),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(0,0,0))

        gray_w = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        _, wmask = cv2.threshold(gray_w, 1, 255, cv2.THRESH_BINARY)

        if alpha_blend and canvas_mask.any() and wmask.any():
            # Alpha blend new image into existing canvas
            canvas = blend_images(canvas, warped, canvas_mask, wmask, alpha_blend=True)
        else:
            # Simple fill: don't overwrite already-placed pixels
            empty = canvas_mask == 0
            canvas[empty & (wmask > 0)] = warped[empty & (wmask > 0)]

        canvas_mask = cv2.bitwise_or(canvas_mask, wmask)

    if crop_borders:
        panorama = crop_black_borders(canvas)
    else:
        panorama = canvas

    info = {
        "canvas_w":     canvas_w,
        "canvas_h":     canvas_h,
        "panorama_w":   panorama.shape[1],
        "panorama_h":   panorama.shape[0],
        "x_offset":     x_off,
        "y_offset":     y_off,
        "skipped_imgs": skipped,
        "num_images":   len(images) - len(skipped),
    }
    logger.info(f"Panorama complete: {panorama.shape[1]}×{panorama.shape[0]} px")
    return panorama, info
