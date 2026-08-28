"""
scripts/run_pipeline.py
========================
CLI entry point for the panorama pipeline.

Usage:
    python scripts/run_pipeline.py --algorithm sift
    python scripts/run_pipeline.py --algorithm orb --data data/raw --output outputs/custom
    python scripts/run_pipeline.py --algorithm sift --ratio 0.7 --ransac 4.0
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import cv2
import numpy as np
from src.config import RESULTS_DIR, RANDOM_SEED
from src.preprocessing import load_image_set, preprocess
from src.pipeline import run_pair_pipeline, run_multi_image_pipeline
from src.evaluation import get_software_versions
from src.visualization import save_input_images, save_panorama

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="CSCD608 Feature-Based Panorama Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_pipeline.py --algorithm sift
  python scripts/run_pipeline.py --algorithm orb --data data/raw
  python scripts/run_pipeline.py --algorithm sift --ratio 0.7 --ransac 4.0
        """
    )
    parser.add_argument("--algorithm", "-a", required=True,
                        choices=["sift", "orb", "SIFT", "ORB"],
                        help="Feature detection/description method")
    parser.add_argument("--data",      "-d", default="data/raw",
                        help="Input image directory (default: data/raw)")
    parser.add_argument("--output",    "-o", default=None,
                        help="Output directory (default: outputs/{algorithm})")
    parser.add_argument("--ratio",     type=float, default=0.75,
                        help="Ratio test threshold for SIFT (default: 0.75)")
    parser.add_argument("--ransac",    type=float, default=5.0,
                        help="RANSAC reprojection threshold in pixels (default: 5.0)")
    parser.add_argument("--resize",    type=int,   default=1280,
                        help="Max image dimension for resize (default: 1280)")
    parser.add_argument("--no-viz",    action="store_true",
                        help="Skip saving visualizations (faster)")
    args = parser.parse_args()

    method     = args.algorithm.upper()
    out_dir    = Path(args.output or f"outputs/{method.lower()}")
    data_dir   = Path(args.data)

    np.random.seed(RANDOM_SEED)

    print(f"\n{'='*60}")
    print(f"CSCD608 Panorama Pipeline  |  Method: {method}")
    print(f"Input:  {data_dir}")
    print(f"Output: {out_dir}")
    print(f"Ratio threshold: {args.ratio}  |  RANSAC: {args.ransac}px")
    versions = get_software_versions()
    print(f"OpenCV: {versions['opencv']}  |  NumPy: {versions['numpy']}")
    print(f"{'='*60}\n")

    # Load images
    try:
        img_pairs = load_image_set(data_dir)
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        logger.error("\nPlease add overlapping images to data/raw/ and retry.")
        logger.error("See README.md §Dataset Setup for instructions.")
        sys.exit(1)

    images_bgr = [img for _, img in img_pairs]
    names      = [p.name for p, _ in img_pairs]

    print(f"Loaded {len(images_bgr)} images: {names}\n")

    if not args.no_viz:
        save_input_images(images_bgr, names, out_dir / "input_images.png")

    # Run all adjacent pairs
    from src.evaluation import append_result_csv
    results_csv = Path(RESULTS_DIR) / f"{method.lower()}_pipeline_results.csv"
    Path(RESULTS_DIR, "logs").mkdir(parents=True, exist_ok=True)

    for i in range(len(images_bgr) - 1):
        pair_name = f"img{i+1}-img{i+2}"
        row = run_pair_pipeline(
            images_bgr[i], images_bgr[i+1],
            method=method,
            out_dir=out_dir,
            experiment="pipeline_run",
            pair_name=pair_name,
            condition="cli",
            ratio_threshold=args.ratio,
            ransac_threshold=args.ransac,
            save_viz=not args.no_viz,
        )
        append_result_csv(row, results_csv)
        print(f"  Pair {pair_name}: "
              f"kp1={row.get('num_kp_img1',0)} "
              f"matches={row.get('num_good_matches',0)} "
              f"inliers={row.get('num_ransac_inliers',0)} "
              f"ratio={row.get('inlier_ratio',0):.2%} "
              f"success={'[OK]' if row.get('homography_success') else '[FAIL]'}")

    # Full panorama
    print("\nBuilding full panorama...")
    panorama, _ = run_multi_image_pipeline(
        images_bgr, names, method,
        out_dir=out_dir / "panorama",
        experiment="pipeline_run",
        ratio_threshold=args.ratio,
        ransac_threshold=args.ransac,
    )
    if panorama is not None:
        pano_path = out_dir / f"final_panorama_{method}.png"
        cv2.imwrite(str(pano_path), panorama)
        if not args.no_viz:
            save_panorama(panorama, method, "pipeline_run", pano_path)
        print(f"\nPanorama saved: {pano_path}")
        print(f"Panorama size:  {panorama.shape[1]}×{panorama.shape[0]} px")
    else:
        print("\nPanorama construction failed — check logs for details.")

    print(f"\nResults CSV: {results_csv}")
    print(f"Outputs:     {out_dir}")
    print("\nDone.")


if __name__ == "__main__":
    main()
