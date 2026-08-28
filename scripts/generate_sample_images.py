"""
scripts/generate_sample_images.py
=================================
Generates a realistic, highly textured 3-image overlapping synthetic scene
and places it into data/raw/ so the pipeline and all experiments can be
executed immediately out-of-the-box.

The scene features multiple geometric patterns, textured objects, text,
and high-frequency details across an extended canvas, cropped with ~50%
overlap to simulate three camera views with horizontal translation/panning.

Usage:
    python scripts/generate_sample_images.py
    python scripts/generate_sample_images.py --output data/raw --count 3
"""

import argparse
import logging
from pathlib import Path
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def create_rich_panoramic_canvas(total_w: int = 1800, h: int = 600) -> np.ndarray:
    """
    Construct a large, feature-rich panoramic canvas with varied shapes,
    textures, grid structures, and text to ensure high-density SIFT & ORB
    keypoints across the entire field of view.
    """
    canvas = np.ones((h, total_w, 3), dtype=np.uint8) * 245
    rng = np.random.RandomState(101)

    # 1. Background gradient
    for x in range(total_w):
        factor = 0.85 + 0.15 * np.sin(x / 150.0)
        canvas[:, x] = (canvas[:, x] * factor).astype(np.uint8)

    # 2. Geometric buildings / architectural facades
    for i in range(12):
        bx = 50 + i * 140 + rng.randint(-15, 15)
        bw = rng.randint(90, 130)
        bh = rng.randint(250, 480)
        by = h - bh
        color = tuple(int(c) for c in rng.randint(60, 180, 3))
        cv2.rectangle(canvas, (bx, by), (bx + bw, h), color, -1)
        cv2.rectangle(canvas, (bx, by), (bx + bw, h), (30, 30, 30), 2)

        # Windows on building
        win_rows = bh // 40
        win_cols = bw // 30
        for wr in range(win_rows):
            for wc in range(win_cols):
                wx = bx + 10 + wc * 28
                wy = by + 15 + wr * 35
                if wx + 20 < bx + bw and wy + 25 < h - 20:
                    wcolor = (240, 240, 200) if (wr + wc) % 2 == 0 else (70, 70, 70)
                    cv2.rectangle(canvas, (wx, wy), (wx + 18, wy + 22), wcolor, -1)
                    cv2.rectangle(canvas, (wx, wy), (wx + 18, wy + 22), (20, 20, 20), 1)

    # 3. High-contrast geometric textures (circles, crosses, stars)
    for _ in range(150):
        cx = rng.randint(30, total_w - 30)
        cy = rng.randint(40, h - 40)
        radius = rng.randint(8, 35)
        color = tuple(int(c) for c in rng.randint(20, 220, 3))
        shape_type = rng.choice(["circle", "rect", "cross", "star"])

        if shape_type == "circle":
            cv2.circle(canvas, (cx, cy), radius, color, -1)
            cv2.circle(canvas, (cx, cy), radius, (0, 0, 0), 1)
        elif shape_type == "rect":
            cv2.rectangle(canvas, (cx - radius, cy - radius),
                          (cx + radius, cy + radius), color, -1)
            cv2.rectangle(canvas, (cx - radius, cy - radius),
                          (cx + radius, cy + radius), (0, 0, 0), 1)
        elif shape_type == "cross":
            cv2.line(canvas, (cx - radius, cy), (cx + radius, cy), color, 3)
            cv2.line(canvas, (cx, cy - radius), (cx, cy + radius), color, 3)
        elif shape_type == "star":
            pts = np.array([[cx, cy - radius], [cx + radius//2, cy - radius//3],
                            [cx + radius, cy], [cx + radius//2, cy + radius//3],
                            [cx, cy + radius], [cx - radius//2, cy + radius//3],
                            [cx - radius, cy], [cx - radius//2, cy - radius//3]], np.int32)
            cv2.fillPoly(canvas, [pts], color)

    # 4. Text labels with sharp corners for corner detectors
    labels = ["CSCD608", "COMPUTER VISION", "PANORAMA", "SIFT", "ORB", "RANSAC", "HOMOGRAPHY", "WARPING"]
    for i, txt in enumerate(labels * 2):
        tx = 40 + i * 110 + rng.randint(-10, 10)
        ty = rng.randint(50, 180)
        scale = 0.6 + rng.rand() * 0.4
        cv2.putText(canvas, txt, (tx, ty), cv2.FONT_HERSHEY_DUPLEX,
                    scale, (20, 20, 20), 2, cv2.LINE_AA)

    return canvas


def generate_overlapping_images(canvas: np.ndarray, n_images: int = 3,
                                img_w: int = 700) -> list[np.ndarray]:
    """
    Extract n_images overlapping perspective views from the wide canvas.
    """
    h, total_w = canvas.shape[:2]
    step = (total_w - img_w) // (n_images - 1)
    images = []

    for i in range(n_images):
        x_start = i * step
        crop = canvas[:, x_start:x_start + img_w].copy()

        # Add subtle natural perspective shift (slight tilt to mimic real camera panning)
        pts_src = np.float32([[0, 0], [img_w, 0], [img_w, h], [0, h]])
        # Slight y-shear depending on image position
        dy = (i - (n_images - 1) / 2.0) * 8.0
        pts_dst = np.float32([
            [0, max(0.0, dy)],
            [img_w, max(0.0, -dy)],
            [img_w, h - max(0.0, dy)],
            [0, h - max(0.0, -dy)]
        ])
        H_subtle = cv2.getPerspectiveTransform(pts_src, pts_dst)
        warped_view = cv2.warpPerspective(crop, H_subtle, (img_w, h), borderMode=cv2.BORDER_REFLECT)
        images.append(warped_view)

    return images


def main():
    parser = argparse.ArgumentParser(description="Generate sample textured dataset for data/raw")
    parser.add_argument("--output", "-o", default="data/raw", help="Output directory")
    parser.add_argument("--count",  "-n", type=int, default=3, help="Number of overlapping images")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating synthetic panoramic scene for {args.count} overlapping views...")
    canvas = create_rich_panoramic_canvas(total_w=1800, h=600)
    images = generate_overlapping_images(canvas, n_images=args.count, img_w=750)

    for i, img in enumerate(images, 1):
        filename = out_dir / f"scene_img{i:02d}.jpg"
        cv2.imwrite(str(filename), img)
        logger.info(f"Saved: {filename} ({img.shape[1]}x{img.shape[0]} px)")

    logger.info("Sample dataset generated successfully in data/raw/")


if __name__ == "__main__":
    main()
