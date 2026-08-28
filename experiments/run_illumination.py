"""
experiments/run_illumination.py
================================
Illumination robustness experiment. REQ-12D.
Usage: python experiments/run_illumination.py
"""
import argparse, logging, sys
from pathlib import Path
import cv2, numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import SUPPORTED_METHODS, RESULTS_DIR, RANDOM_SEED, BRIGHTNESS_DELTAS, CONTRAST_FACTORS
from src.preprocessing import load_image_set, preprocess
from src.pipeline import run_pair_pipeline
from src.evaluation import append_result_csv, save_experiment_config, get_software_versions
from src.visualization import save_experiment_trend

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler(Path(RESULTS_DIR)/"logs"/"illumination.log")])
logger = logging.getLogger(__name__)


def run_illumination(data_dir="data/raw", output_dir="outputs/illumination",
                     methods=None):
    np.random.seed(RANDOM_SEED)
    methods = methods or SUPPORTED_METHODS
    out_dir = Path(output_dir)
    results_csv = Path(RESULTS_DIR) / "illumination_results.csv"
    Path(RESULTS_DIR, "logs").mkdir(parents=True, exist_ok=True)
    logger.info("=" * 60)
    logger.info("ILLUMINATION EXPERIMENT — Brightness & contrast changes")
    logger.info("=" * 60)

    try:
        img_pairs = load_image_set(Path(data_dir))
    except Exception as e:
        logger.error(str(e)); sys.exit(1)

    ref_bgr   = img_pairs[0][1]
    ref_color, _ = preprocess(ref_bgr)

    # Build all variants
    variants = {}
    for beta in BRIGHTNESS_DELTAS:
        tag = f"bright_p{beta}" if beta >= 0 else f"bright_n{abs(beta)}"
        variants[tag] = {
            "img": cv2.convertScaleAbs(ref_color, alpha=1.0, beta=beta),
            "brightness_delta": beta, "contrast_factor": 1.0, "type": "brightness",
        }
    for alpha in CONTRAST_FACTORS:
        tag = f"contrast_{str(alpha).replace('.','p')}"
        variants[tag] = {
            "img": cv2.convertScaleAbs(ref_color, alpha=alpha, beta=0),
            "brightness_delta": 0, "contrast_factor": alpha, "type": "contrast",
        }

    save_experiment_config({"experiment":"illumination","methods":methods,
        "brightness_deltas":BRIGHTNESS_DELTAS,"contrast_factors":CONTRAST_FACTORS,
        "random_seed":RANDOM_SEED,**get_software_versions()},
        out_dir/"experiment_config.json")
    all_rows = []

    for method in methods:
        for tag, data in variants.items():
            pair_name = f"ref-{tag}"
            row = run_pair_pipeline(ref_color, data["img"], method=method,
                out_dir=out_dir/method, experiment="illumination",
                pair_name=pair_name, condition=tag, save_viz=True)
            row["illumination_tag"]   = tag
            row["brightness_delta"]   = data["brightness_delta"]
            row["contrast_factor"]    = data["contrast_factor"]
            row["illumination_type"]  = data["type"]
            all_rows.append(row)
            append_result_csv(row, results_csv)
            logger.info(f"  [{method}] {tag}  inliers={row.get('num_ransac_inliers',0)}")

    df = pd.DataFrame(all_rows)
    if not df.empty:
        bdf = df[df["illumination_type"]=="brightness"].copy()
        if not bdf.empty and "brightness_delta" in bdf.columns and "inlier_ratio" in bdf.columns:
            save_experiment_trend(bdf,"brightness_delta","inlier_ratio",
                "Illumination: Brightness — Inlier Ratio","Brightness Delta (β)","Inlier Ratio",
                out_dir/"illum_brightness_inlier_ratio.png")
    logger.info(f"Illumination results → {results_csv}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",      default="data/raw")
    ap.add_argument("--output",    default="outputs/illumination")
    ap.add_argument("--algorithm", default=None)
    args = ap.parse_args()
    methods = [args.algorithm.upper()] if args.algorithm else None
    run_illumination(args.data, args.output, methods)
