"""
experiments/run_scale.py
=========================
Scale robustness experiment. REQ-12B.
Usage: python experiments/run_scale.py
"""
import argparse, logging, sys
from pathlib import Path
import cv2, numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import SUPPORTED_METHODS, RESULTS_DIR, RANDOM_SEED, SCALE_FACTORS
from src.preprocessing import load_image_set, preprocess
from src.pipeline import run_pair_pipeline
from src.evaluation import append_result_csv, save_experiment_config, get_software_versions
from src.visualization import save_experiment_trend

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler(Path(RESULTS_DIR)/"logs"/"scale.log")])
logger = logging.getLogger(__name__)


def run_scale(data_dir="data/raw", output_dir="outputs/scale",
              factors=None, methods=None):
    np.random.seed(RANDOM_SEED)
    factors = factors or SCALE_FACTORS
    methods = methods or SUPPORTED_METHODS
    out_dir = Path(output_dir)
    results_csv = Path(RESULTS_DIR) / "scale_results.csv"
    Path(RESULTS_DIR, "logs").mkdir(parents=True, exist_ok=True)
    logger.info("=" * 60)
    logger.info("SCALE EXPERIMENT — Synthetic controlled scale changes")
    logger.info("=" * 60)

    try:
        img_pairs = load_image_set(Path(data_dir))
    except Exception as e:
        logger.error(str(e)); sys.exit(1)

    ref_bgr = img_pairs[0][1]
    ref_color, _ = preprocess(ref_bgr)
    h, w = ref_color.shape[:2]

    save_experiment_config({"experiment": "scale", "factors": factors, "methods": methods,
        "random_seed": RANDOM_SEED, **get_software_versions()},
        out_dir / "experiment_config.json")
    all_rows = []

    for method in methods:
        for s in factors:
            nw, nh   = max(1, int(w * s)), max(1, int(h * s))
            small    = cv2.resize(ref_color, (nw, nh), interpolation=cv2.INTER_LINEAR)
            restored = cv2.resize(small, (w, h),       interpolation=cv2.INTER_LINEAR)
            pair_name = f"ref-scale{str(s).replace('.', 'p')}"
            row = run_pair_pipeline(
                ref_color, restored, method=method,
                out_dir=out_dir / method, experiment="scale",
                pair_name=pair_name, condition=str(s),
                save_viz=(s in [0.5, 1.0, 2.0]),
            )
            row["scale_factor"] = s
            all_rows.append(row)
            append_result_csv(row, results_csv)
            logger.info(f"  [{method}] scale={s}  inliers={row.get('num_ransac_inliers',0)}")

    df = pd.DataFrame(all_rows)
    if not df.empty and "scale_factor" in df.columns:
        for metric, ylabel in [("inlier_ratio", "Inlier Ratio"),
                                ("num_ransac_inliers", "RANSAC Inliers")]:
            if metric in df.columns:
                save_experiment_trend(df, "scale_factor", metric,
                    f"Scale Robustness — {metric}", "Scale Factor", ylabel,
                    out_dir / f"scale_{metric}.png")
    logger.info(f"Scale results → {results_csv}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",      default="data/raw")
    ap.add_argument("--output",    default="outputs/scale")
    ap.add_argument("--algorithm", default=None)
    args = ap.parse_args()
    methods = [args.algorithm.upper()] if args.algorithm else None
    run_scale(args.data, args.output, methods=methods)
