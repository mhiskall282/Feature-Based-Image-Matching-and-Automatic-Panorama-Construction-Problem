"""
experiments/run_rotation.py
============================
Rotation robustness experiment.

Generates synthetic rotated versions of the reference image and measures
how SIFT and ORB perform across increasing rotation angles.

Usage:
    python experiments/run_rotation.py
    python experiments/run_rotation.py --data data/raw --angles 0 30 60 90 180

REQ-12: Investigate performance under rotation.
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config     import SUPPORTED_METHODS, RESULTS_DIR, RANDOM_SEED, ROTATION_ANGLES
from src.preprocessing import load_image_set, preprocess
from src.pipeline   import run_pair_pipeline
from src.evaluation import append_result_csv, save_experiment_config, get_software_versions
from src.visualization import save_experiment_trend

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler(Path(RESULTS_DIR) / "logs" / "rotation.log")])
logger = logging.getLogger(__name__)


def run_rotation(data_dir: str = "data/raw",
                 output_dir: str = "outputs/rotation",
                 angles: list | None = None,
                 methods: list | None = None) -> pd.DataFrame:
    """
    For each rotation angle, generate a rotated copy of the reference image
    and run the full pipeline against the original reference.

    Clearly labeled as SYNTHETIC ROTATION EXPERIMENT.
    """
    np.random.seed(RANDOM_SEED)
    angles  = angles  or ROTATION_ANGLES
    methods = methods or SUPPORTED_METHODS
    data_dir = Path(data_dir)
    out_dir  = Path(output_dir)
    results_csv = Path(RESULTS_DIR) / "rotation_results.csv"
    Path(RESULTS_DIR, "logs").mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("ROTATION EXPERIMENT — Synthetic controlled rotation")
    logger.info("=" * 60)

    try:
        img_pairs = load_image_set(data_dir)
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e)); sys.exit(1)

    ref_bgr   = img_pairs[0][1]
    ref_color, ref_gray = preprocess(ref_bgr)
    h, w = ref_gray.shape[:2]

    config = {"experiment": "rotation", "angles": angles, "methods": methods,
              "random_seed": RANDOM_SEED, **get_software_versions()}
    save_experiment_config(config, out_dir / "experiment_config.json")

    all_rows = []

    for method in methods:
        for angle in angles:
            pair_name = f"ref-rot{angle:03d}"
            # Generate rotated image
            cx, cy = w / 2, h / 2
            M = cv2.getRotationMatrix2D((cx, cy), float(angle), 1.0)
            cos_a, sin_a = abs(M[0,0]), abs(M[0,1])
            nw = int(h * sin_a + w * cos_a)
            nh = int(h * cos_a + w * sin_a)
            M[0,2] += nw/2 - cx; M[1,2] += nh/2 - cy
            rotated = cv2.warpAffine(ref_color, M, (nw, nh))

            row = run_pair_pipeline(
                img1_bgr=ref_color,
                img2_bgr=rotated,
                method=method,
                out_dir=out_dir / method,
                experiment="rotation",
                pair_name=pair_name,
                condition=str(angle),
                save_viz=(angle in [0, 30, 90]),  # save viz for key angles only
            )
            row["rotation_angle_deg"] = angle
            all_rows.append(row)
            append_result_csv(row, results_csv)
            logger.info(f"  [{method}] angle={angle}°  inliers={row.get('num_ransac_inliers',0)}  "
                        f"ratio={row.get('inlier_ratio',0):.3f}")

    df = pd.DataFrame(all_rows)
    if not df.empty and "rotation_angle_deg" in df.columns:
        for metric, ylabel in [("inlier_ratio", "Inlier Ratio"),
                                ("num_ransac_inliers", "RANSAC Inliers"),
                                ("time_total_s", "Total Time (s)")]:
            if metric in df.columns:
                save_experiment_trend(
                    df, "rotation_angle_deg", metric,
                    title=f"Rotation Robustness — {metric}",
                    xlabel="Rotation Angle (°)", ylabel=ylabel,
                    out_path=out_dir / f"rotation_{metric}.png",
                )
    logger.info(f"Rotation results → {results_csv}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",    default="data/raw")
    parser.add_argument("--output",  default="outputs/rotation")
    parser.add_argument("--angles",  nargs="+", type=int, default=None)
    parser.add_argument("--algorithm", default=None)
    args = parser.parse_args()
    methods = [args.algorithm.upper()] if args.algorithm else None
    run_rotation(args.data, args.output, args.angles, methods)
