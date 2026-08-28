"""
experiments/run_viewpoint.py
=============================
Viewpoint robustness experiment. REQ-12C.
Usage: python experiments/run_viewpoint.py
"""
import argparse, logging, sys
from pathlib import Path
import cv2, numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import SUPPORTED_METHODS, RESULTS_DIR, RANDOM_SEED
from src.preprocessing import load_image_set, preprocess
from src.pipeline import run_pair_pipeline
from src.evaluation import append_result_csv, save_experiment_config, get_software_versions
from src.visualization import save_experiment_trend

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler(Path(RESULTS_DIR)/"logs"/"viewpoint.log")])
logger = logging.getLogger(__name__)

# Viewpoint levels: name → corner offset in pixels
VIEWPOINT_LEVELS = [("mild", 15), ("moderate", 40), ("extreme", 80)]


def run_viewpoint(data_dir="data/raw", output_dir="outputs/viewpoint",
                  methods=None):
    np.random.seed(RANDOM_SEED)
    methods = methods or SUPPORTED_METHODS
    out_dir = Path(output_dir)
    results_csv = Path(RESULTS_DIR) / "viewpoint_results.csv"
    Path(RESULTS_DIR, "logs").mkdir(parents=True, exist_ok=True)
    logger.info("=" * 60)
    logger.info("VIEWPOINT EXPERIMENT — Synthetic perspective distortion")
    logger.info("=" * 60)

    try:
        img_pairs = load_image_set(Path(data_dir))
    except Exception as e:
        logger.error(str(e)); sys.exit(1)

    ref_bgr = img_pairs[0][1]
    ref_color, _ = preprocess(ref_bgr)
    h, w = ref_color.shape[:2]

    save_experiment_config({"experiment":"viewpoint","levels":VIEWPOINT_LEVELS,
        "methods":methods,"random_seed":RANDOM_SEED,**get_software_versions()},
        out_dir/"experiment_config.json")
    all_rows = []

    for method in methods:
        for level_name, offset in VIEWPOINT_LEVELS:
            src = np.float32([[0,0],[w,0],[w,h],[0,h]])
            dst = np.float32([[offset,offset],[w-offset,0],[w,h],[0,h-offset]])
            M   = cv2.getPerspectiveTransform(src, dst)
            warped = cv2.warpPerspective(ref_color, M, (w, h))

            pair_name = f"ref-viewpoint_{level_name}"
            row = run_pair_pipeline(ref_color, warped, method=method,
                out_dir=out_dir/method, experiment="viewpoint",
                pair_name=pair_name, condition=level_name, save_viz=True)
            row["viewpoint_level"] = level_name
            row["viewpoint_offset_px"] = offset
            all_rows.append(row)
            append_result_csv(row, results_csv)
            logger.info(f"  [{method}] {level_name}(off={offset}px)  "
                        f"inliers={row.get('num_ransac_inliers',0)}")

    df = pd.DataFrame(all_rows)
    if not df.empty and "viewpoint_offset_px" in df.columns:
        for metric, ylabel in [("inlier_ratio","Inlier Ratio"),
                                ("num_ransac_inliers","RANSAC Inliers")]:
            if metric in df.columns:
                save_experiment_trend(df,"viewpoint_offset_px",metric,
                    f"Viewpoint Robustness — {metric}","Perspective Offset (px)",ylabel,
                    out_dir/f"viewpoint_{metric}.png")
    logger.info(f"Viewpoint results → {results_csv}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",      default="data/raw")
    ap.add_argument("--output",    default="outputs/viewpoint")
    ap.add_argument("--algorithm", default=None)
    args = ap.parse_args()
    methods = [args.algorithm.upper()] if args.algorithm else None
    run_viewpoint(args.data, args.output, methods)
