"""
scripts/prepare_dataset.py
==========================
Utility to generate controlled synthetic experiment images from baseline images.

Usage:
    python scripts/prepare_dataset.py --input data/raw --output data/transformed

Creates rotation, scale, viewpoint, and illumination variants for experiments.
These are clearly labeled as SYNTHETIC EXPERIMENTAL IMAGES.
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import (ROTATION_ANGLES, SCALE_FACTORS,
                        BRIGHTNESS_DELTAS, CONTRAST_FACTORS)
from src.preprocessing import load_image_set

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def generate_rotated(img: np.ndarray, angles: list) -> dict:
    """Generate rotated copies. Ground truth H is also returned."""
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    results = {}
    for angle in angles:
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        cos_a, sin_a = abs(M[0,0]), abs(M[0,1])
        nw = int(h * sin_a + w * cos_a)
        nh = int(h * cos_a + w * sin_a)
        M[0, 2] += nw / 2 - cx
        M[1, 2] += nh / 2 - cy
        rotated = cv2.warpAffine(img, M, (nw, nh))
        # Convert affine M to homography for ground truth
        H_gt = np.eye(3, dtype=np.float64)
        H_gt[:2, :] = M
        results[f"rot_{angle:03d}"] = {"img": rotated, "angle_deg": angle, "H_gt": H_gt.tolist()}
    return results


def generate_scaled(img: np.ndarray, factors: list) -> dict:
    """Generate scale variants (resize + restore to original size)."""
    h, w = img.shape[:2]
    results = {}
    for s in factors:
        nw, nh = max(1, int(w * s)), max(1, int(h * s))
        small    = cv2.resize(img, (nw, nh),  interpolation=cv2.INTER_LINEAR)
        restored = cv2.resize(small, (w, h),  interpolation=cv2.INTER_LINEAR)
        results[f"scale_{s:.2f}".replace(".", "p")] = {
            "img": restored, "scale_factor": s
        }
    return results


def generate_perspective(img: np.ndarray) -> dict:
    """Generate perspective warp variants."""
    h, w = img.shape[:2]
    levels = [("mild", 20), ("moderate", 60), ("extreme", 100)]
    results = {}
    for name, off in levels:
        src = np.float32([[0,0],[w,0],[w,h],[0,h]])
        dst = np.float32([[off,off],[w-off,0],[w,h],[0,h-off]])
        M   = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(img, M, (w, h))
        results[f"viewpoint_{name}"] = {"img": warped, "offset_px": off, "H_gt": M.tolist()}
    return results


def generate_illumination(img: np.ndarray) -> dict:
    """Generate brightness/contrast variants."""
    results = {}
    for beta in BRIGHTNESS_DELTAS:
        tag  = f"bright_p{beta}" if beta >= 0 else f"bright_n{abs(beta)}"
        results[tag] = {
            "img": cv2.convertScaleAbs(img, alpha=1.0, beta=beta),
            "brightness_delta": beta, "contrast_factor": 1.0,
        }
    for alpha in CONTRAST_FACTORS:
        tag = f"contrast_{alpha:.2f}".replace(".", "p")
        results[tag] = {
            "img": cv2.convertScaleAbs(img, alpha=alpha, beta=0),
            "brightness_delta": 0, "contrast_factor": alpha,
        }
    return results


def save_variants(variants: dict, out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in variants.items():
        img_path = out_dir / f"{prefix}_{name}.jpg"
        cv2.imwrite(str(img_path), data["img"])
        logger.info(f"  Saved: {img_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic experiment images")
    parser.add_argument("--input",  default="data/raw",         help="Source images directory")
    parser.add_argument("--output", default="data/transformed",  help="Output directory")
    args = parser.parse_args()

    raw_dir  = Path(args.input)
    xfrm_dir = Path(args.output)

    try:
        img_pairs = load_image_set(raw_dir)
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        sys.exit(1)

    # Use the first image as the source for single-image transformations
    ref_path, ref_img = img_pairs[0]
    stem = ref_path.stem
    logger.info(f"Using reference image: {ref_path.name}")

    logger.info("Generating rotation variants...")
    save_variants(generate_rotated(ref_img, ROTATION_ANGLES),
                  xfrm_dir / "rotation", stem)

    logger.info("Generating scale variants...")
    save_variants(generate_scaled(ref_img, SCALE_FACTORS),
                  xfrm_dir / "scale", stem)

    logger.info("Generating viewpoint variants...")
    save_variants(generate_perspective(ref_img),
                  xfrm_dir / "viewpoint", stem)

    logger.info("Generating illumination variants...")
    save_variants(generate_illumination(ref_img),
                  xfrm_dir / "illumination", stem)

    logger.info(f"\nDone. Synthetic images saved to: {xfrm_dir}")
    logger.info("NOTE: All images in data/transformed/ are SYNTHETIC EXPERIMENTAL IMAGES.")


if __name__ == "__main__":
    main()
