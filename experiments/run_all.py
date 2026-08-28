"""
experiments/run_all.py
=======================
Run all experiments sequentially.

Usage:
    python experiments/run_all.py
    python experiments/run_all.py --data data/raw --skip baseline
"""
import argparse, logging, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import RESULTS_DIR
from scripts.generate_report_tables import generate_all_tables

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run all CSCD608 panorama experiments")
    parser.add_argument("--data",  default="data/raw")
    parser.add_argument("--skip",  nargs="*", default=[],
                        help="Experiments to skip: baseline rotation scale viewpoint illumination")
    args = parser.parse_args()

    Path(RESULTS_DIR, "logs").mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    print("\n" + "="*70)
    print("CSCD608 ADVANCED COMPUTER VISION")
    print("Feature-Based Image Matching & Automatic Panorama Construction")
    print("="*70 + "\n")

    skip = [s.lower() for s in args.skip]

    # 1. Baseline
    if "baseline" not in skip:
        print("\n[1/5] BASELINE EXPERIMENT")
        from experiments.run_baseline import run_baseline
        run_baseline(data_dir=args.data)

    # 2. Rotation
    if "rotation" not in skip:
        print("\n[2/5] ROTATION EXPERIMENT")
        from experiments.run_rotation import run_rotation
        run_rotation(data_dir=args.data)

    # 3. Scale
    if "scale" not in skip:
        print("\n[3/5] SCALE EXPERIMENT")
        from experiments.run_scale import run_scale
        run_scale(data_dir=args.data)

    # 4. Viewpoint
    if "viewpoint" not in skip:
        print("\n[4/5] VIEWPOINT EXPERIMENT")
        from experiments.run_viewpoint import run_viewpoint
        run_viewpoint(data_dir=args.data)

    # 5. Illumination
    if "illumination" not in skip:
        print("\n[5/5] ILLUMINATION EXPERIMENT")
        from experiments.run_illumination import run_illumination
        run_illumination(data_dir=args.data)

    # Generate tables
    print("\n[+] Generating report tables and comparison plots...")
    try:
        generate_all_tables()
    except Exception as e:
        logger.warning(f"Table generation skipped: {e}")

    elapsed = time.perf_counter() - t_start
    print(f"\n{'='*70}")
    print(f"All experiments complete in {elapsed:.1f}s")
    print(f"Results -> {RESULTS_DIR}/")
    print(f"Outputs -> outputs/")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
