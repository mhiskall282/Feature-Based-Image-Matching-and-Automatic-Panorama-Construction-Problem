"""
scripts/generate_report_tables.py
===================================
Aggregate all CSV results into comparison tables and export plots.

Usage:
    python scripts/generate_report_tables.py
"""
import sys, logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd
from src.config import RESULTS_DIR
from src.visualization import save_comparison_bars, save_experiment_trend

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_results(name: str) -> pd.DataFrame | None:
    path = Path(RESULTS_DIR) / f"{name}_results.csv"
    if path.exists():
        df = pd.read_csv(path)
        logger.info(f"Loaded {len(df)} rows from {path}")
        return df
    logger.warning(f"Results file not found: {path}")
    return None


def generate_comparison_table(df: pd.DataFrame, out_path: Path) -> None:
    """Master comparison table: method × experiment × metrics."""
    if df is None or df.empty:
        return
    cols = ["experiment","method","num_kp_img1","num_kp_img2",
            "num_good_matches","num_ransac_inliers","inlier_ratio",
            "time_total_s","homography_success"]
    available = [c for c in cols if c in df.columns]
    summary = df[available].groupby(["experiment","method"], as_index=False).mean(numeric_only=True)
    summary.to_csv(out_path, index=False)
    logger.info(f"Comparison table → {out_path}")


def generate_all_tables():
    tables_dir = Path(RESULTS_DIR) / "tables"
    plots_dir  = Path(RESULTS_DIR) / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    experiments = ["baseline","rotation","scale","viewpoint","illumination"]
    all_dfs = []

    for name in experiments:
        df = load_results(name)
        if df is not None and not df.empty:
            df["experiment"] = name
            all_dfs.append(df)

    if not all_dfs:
        logger.warning("No result CSVs found. Run experiments first.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(tables_dir / "all_results.csv", index=False)

    # Master comparison table
    generate_comparison_table(combined, tables_dir / "comparison_table.csv")

    # Per-experiment tables
    for name in experiments:
        sub = combined[combined["experiment"] == name]
        if not sub.empty:
            sub.to_csv(tables_dir / f"{name}_table.csv", index=False)

    # Comparison bar charts
    key_metrics = [m for m in ["num_kp_img1","num_good_matches",
                                "num_ransac_inliers","inlier_ratio","time_total_s"]
                   if m in combined.columns]
    if key_metrics:
        baseline_df = combined[combined["experiment"]=="baseline"]
        if not baseline_df.empty:
            save_comparison_bars(baseline_df, key_metrics,
                                 "Baseline: SIFT vs ORB Comparison",
                                 plots_dir/"baseline_comparison.png")

    # Trend plots for stress experiments
    trend_cfg = [
        ("rotation",    "rotation_angle_deg",   "Rotation Angle (°)"),
        ("scale",       "scale_factor",          "Scale Factor"),
        ("viewpoint",   "viewpoint_offset_px",   "Perspective Offset (px)"),
        ("illumination","brightness_delta",       "Brightness Delta"),
    ]
    for exp_name, x_col, xlabel in trend_cfg:
        sub = combined[combined["experiment"]==exp_name]
        if not sub.empty and x_col in sub.columns and "inlier_ratio" in sub.columns:
            save_experiment_trend(sub, x_col, "inlier_ratio",
                f"{exp_name.title()} - Inlier Ratio", xlabel, "Inlier Ratio",
                plots_dir / f"{exp_name}_inlier_ratio.png")

    logger.info(f"\nAll tables -> {tables_dir}")
    logger.info(f"All plots  -> {plots_dir}")


if __name__ == "__main__":
    generate_all_tables()
