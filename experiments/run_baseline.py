"""
experiments/run_baseline.py
============================
Baseline experiment: ≥3 overlapping images, standard conditions.

Usage:
    python experiments/run_baseline.py
    python experiments/run_baseline.py --data data/raw --output outputs/baseline

REQ-01–REQ-11, REQ-13, REQ-14, REQ-15
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config    import SUPPORTED_METHODS, RESULTS_DIR, RANDOM_SEED
from src.preprocessing import load_image_set
from src.pipeline  import run_pair_pipeline, run_multi_image_pipeline
from src.evaluation import append_result_csv, save_experiment_config, get_software_versions
from src.visualization import save_input_images, save_preprocessed, save_comparison_bars
from src.preprocessing import preprocess

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(RESULTS_DIR) / "logs" / "baseline.log"),
    ],
)
logger = logging.getLogger(__name__)


def run_baseline(data_dir: str = "data/raw",
                 output_dir: str = "outputs/baseline",
                 methods: list | None = None,
                 ratio_threshold: float = 0.75,
                 ransac_threshold: float = 5.0) -> pd.DataFrame:
    """
    Run the baseline experiment:
      - Load all images from data_dir
      - Run both SIFT and ORB on every adjacent pair
      - Build the full N-image panorama
      - Save results CSV and visualizations
    """
    np.random.seed(RANDOM_SEED)
    methods   = methods or SUPPORTED_METHODS
    data_dir  = Path(data_dir)
    out_dir   = Path(output_dir)
    results_csv = Path(RESULTS_DIR) / "baseline_results.csv"
    Path(RESULTS_DIR, "logs").mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("BASELINE EXPERIMENT — CSCD608 Panorama Pipeline")
    logger.info("=" * 60)

    # Load images
    try:
        img_pairs = load_image_set(data_dir)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Cannot load images: {e}")
        logger.error("Add at least 2 overlapping images to data/raw/ and retry.")
        sys.exit(1)

    images_bgr = [img for _, img in img_pairs]
    names      = [p.name for p, _ in img_pairs]
    n          = len(images_bgr)

    logger.info(f"Loaded {n} images: {names}")

    # Save input images visualization
    save_input_images(images_bgr, names, out_dir / "input_images.png")

    # Save preprocessed visualizations for first image
    c0, g0 = preprocess(images_bgr[0])
    save_preprocessed(c0, g0, names[0], out_dir / f"preprocessed_{names[0]}.png")

    # Save experiment config
    config = {
        "experiment":       "baseline",
        "data_dir":         str(data_dir),
        "num_images":       n,
        "image_names":      names,
        "methods":          methods,
        "ratio_threshold":  ratio_threshold,
        "ransac_threshold": ransac_threshold,
        "random_seed":      RANDOM_SEED,
        **get_software_versions(),
    }
    save_experiment_config(config, out_dir / "experiment_config.json")

    all_rows = []

    for method in methods:
        logger.info(f"\n{'─'*40}")
        logger.info(f"Method: {method}")
        logger.info(f"{'─'*40}")
        method_dir = out_dir / method

        # Run every adjacent pair
        for i in range(n - 1):
            pair_name = f"img{i+1}-img{i+2}"
            row = run_pair_pipeline(
                images_bgr[i], images_bgr[i+1],
                method=method,
                out_dir=method_dir,
                experiment="baseline",
                pair_name=pair_name,
                condition="standard",
                ratio_threshold=ratio_threshold,
                ransac_threshold=ransac_threshold,
                save_viz=True,
            )
            all_rows.append(row)
            append_result_csv(row, results_csv)

        # Full multi-image panorama
        panorama, pano_rows = run_multi_image_pipeline(
            images_bgr, names, method,
            out_dir=method_dir / "panorama",
            experiment="baseline",
        )
        for pr in pano_rows:
            all_rows.append(pr)
            append_result_csv(pr, results_csv)

    # Comparison chart
    if all_rows:
        df = pd.DataFrame(all_rows)
        metrics_to_plot = ["num_kp_img1", "num_good_matches",
                           "num_ransac_inliers", "inlier_ratio", "time_total_s"]
        existing = [m for m in metrics_to_plot if m in df.columns]
        if existing:
            save_comparison_bars(df, existing,
                                 "Baseline: SIFT vs ORB",
                                 out_dir / "comparison_bars.png")
        df.to_csv(out_dir / "baseline_summary.csv", index=False)
        logger.info(f"\nBaseline results saved to {results_csv}")
        return df

    return pd.DataFrame()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baseline panorama experiment")
    parser.add_argument("--data",     default="data/raw",        help="Image directory")
    parser.add_argument("--output",   default="outputs/baseline", help="Output directory")
    parser.add_argument("--algorithm",default=None, help="SIFT, ORB, or both (default)")
    parser.add_argument("--ratio",    type=float, default=0.75,  help="Ratio test threshold")
    parser.add_argument("--ransac",   type=float, default=5.0,   help="RANSAC threshold (px)")
    args = parser.parse_args()
    methods = [args.algorithm.upper()] if args.algorithm else None
    run_baseline(args.data, args.output, methods, args.ratio, args.ransac)
