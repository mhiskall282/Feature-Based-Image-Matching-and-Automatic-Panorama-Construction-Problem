"""
src/pipeline.py
===============
Complete end-to-end pipeline: two images → metrics + panorama.

This is the central orchestrator. It calls each pipeline stage explicitly
so the examiner can trace the full flow:

  Image Preparation
    → Feature Detection
    → Feature Description
    → Descriptor Matching
    → Initial Correspondences (visualization)
    → RANSAC
    → Homography Estimation
    → Image Warping
    → Image Alignment & Stitching
    → Quantitative Evaluation
    → Visual Results

# REQ-15: The final system must clearly demonstrate the complete pipeline.
"""

import logging
import time
import numpy as np
from pathlib import Path

from src.preprocessing  import preprocess, validate_image_pair
from src.features       import detect_and_describe
from src.matching       import match_descriptors
from src.homography     import estimate_homography, diagnose_homography, save_homography
from src.warping        import compute_canvas, warp_image
from src.stitching      import stitch_pair
from src.evaluation     import (compute_metrics, build_result_row,
                                append_result_csv, log_failure,
                                assess_panorama_quality)
from src.visualization  import (save_keypoints, save_raw_matches,
                                save_before_after_ransac, save_warped,
                                save_panorama)
from src.config         import RANDOM_SEED, RESULTS_DIR

logger = logging.getLogger(__name__)


def run_pair_pipeline(img1_bgr: np.ndarray,
                      img2_bgr: np.ndarray,
                      method: str,
                      out_dir: str | Path,
                      experiment: str = "baseline",
                      pair_name: str  = "img1-img2",
                      condition: str  = "standard",
                      ratio_threshold: float = 0.75,
                      ransac_threshold: float = 5.0,
                      equalize: bool = False,
                      save_viz: bool = True,
                      alpha_blend: bool = True) -> dict:
    """
    Run the complete pipeline on one image pair with one feature method.

    Stages (all explicit — no black boxes):
      1. Preprocessing
      2. Feature detection + description
      3. Descriptor matching
      4. RANSAC + homography estimation
      5. Warping
      6. Pair stitching
      7. Metric assembly
      8. Visualizations
      9. CSV logging

    Returns:
        Full metrics dict including method, timing, RANSAC stats, panorama info.
    """
    np.random.seed(RANDOM_SEED)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    failures_csv = Path(RESULTS_DIR) / "logs" / "failures.csv"

    t_total_start = time.perf_counter()
    logger.info(f"\n{'='*60}")
    logger.info(f"Pipeline: {experiment} | {method} | {pair_name}")
    logger.info(f"{'='*60}")

    # ── 1. Preprocessing ─────────────────────────────────────
    validate_image_pair(img1_bgr, img2_bgr)
    img1_color, img1_gray = preprocess(img1_bgr, equalize=equalize)
    img2_color, img2_gray = preprocess(img2_bgr, equalize=equalize)

    # ── 2. Feature detection + description ───────────────────
    # REQ-03, REQ-04
    feat1 = detect_and_describe(img1_gray, method)
    feat2 = detect_and_describe(img2_gray, method)

    if feat1["num_kp"] < 4 or feat2["num_kp"] < 4:
        reason = f"too_few_keypoints: img1={feat1['num_kp']}, img2={feat2['num_kp']}"
        log_failure(experiment, method, pair_name, "feature_detection", reason, failures_csv)
        return _failed_row(experiment, method, pair_name, condition, reason, feat1, feat2)

    if save_viz:
        save_keypoints(img1_gray, feat1["keypoints"], method, "img1",
                       out_dir / f"keypoints_{method}_img1.png")
        save_keypoints(img2_gray, feat2["keypoints"], method, "img2",
                       out_dir / f"keypoints_{method}_img2.png")

    # ── 3. Descriptor matching ────────────────────────────────
    # REQ-05
    match_result = match_descriptors(feat1, feat2, ratio_threshold=ratio_threshold)

    if match_result["num_good_matches"] < 4:
        reason = f"too_few_good_matches: n={match_result['num_good_matches']}"
        log_failure(experiment, method, pair_name, "matching", reason, failures_csv)
        return _failed_row(experiment, method, pair_name, condition, reason, feat1, feat2, match_result)

    # ── 4. Initial correspondences visualization ──────────────
    # REQ-06
    if save_viz:
        save_raw_matches(img1_gray, feat1["keypoints"],
                         img2_gray, feat2["keypoints"],
                         match_result["good_matches"], method, pair_name,
                         out_dir / f"raw_matches_{method}_{pair_name}.png")

    # ── 5. RANSAC + homography ────────────────────────────────
    # REQ-07, REQ-08
    ransac_result = estimate_homography(
        feat1, feat2, match_result["good_matches"],
        reproj_threshold=ransac_threshold,
    )

    # Save H matrices
    save_homography(
        ransac_result["H"], out_dir / "homographies",
        prefix=f"H_{method}_{pair_name}",
        metadata={"method": method, "pair": pair_name, "inliers": ransac_result["num_inliers"]},
    )

    # ── 6. Before vs After RANSAC visualization ───────────────
    # REQ-11
    if save_viz and ransac_result["mask"] is not None:
        save_before_after_ransac(
            img1_gray, feat1["keypoints"],
            img2_gray, feat2["keypoints"],
            match_result["good_matches"],
            ransac_result["inlier_matches"],
            method, pair_name,
            out_dir / f"before_after_ransac_{method}_{pair_name}.png",
        )

    # ── 7. Warping + stitching ────────────────────────────────
    # REQ-09, REQ-10
    panorama = None
    stitch_info = {}

    if ransac_result["success"] and ransac_result["H"] is not None:
        H = ransac_result["H"]
        diag = diagnose_homography(H, img1_color.shape, img2_color.shape)
        if diag.get("degenerate"):
            reason = f"degenerate_homography: {diag.get('reason','')}"
            log_failure(experiment, method, pair_name, "homography", reason, failures_csv)
        else:
            panorama, stitch_info = stitch_pair(img1_color, img2_color, H, alpha_blend)

            if save_viz:
                canvas_w, canvas_h, x_off, y_off = compute_canvas(img1_color, img2_color, H)
                warped = warp_image(img1_color, H, canvas_w, canvas_h, x_off, y_off)
                save_warped(img2_color, warped, method, out_dir / f"warped_{method}_{pair_name}.png")
                save_panorama(panorama, method, experiment, out_dir / f"panorama_{method}_{pair_name}.png")
    else:
        reason = ransac_result.get("failure_reason", "unknown")
        log_failure(experiment, method, pair_name, "ransac", reason, failures_csv)

    # ── 8. Metrics ────────────────────────────────────────────
    t_total = time.perf_counter() - t_total_start
    metrics = compute_metrics(feat1, feat2, match_result, ransac_result,
                              stitch_info, panorama)
    metrics["time_total_s"] = round(t_total, 4)

    row = build_result_row(experiment, method, pair_name, condition, metrics,
                           extra={"ratio_threshold": ratio_threshold,
                                  "ransac_threshold": ransac_threshold})

    logger.info(
        f"[{method}] {pair_name}: kp1={feat1['num_kp']} kp2={feat2['num_kp']} "
        f"matches={match_result['num_good_matches']} "
        f"inliers={ransac_result['num_inliers']} "
        f"ratio={ransac_result['inlier_ratio']:.2%} "
        f"total={t_total:.3f}s"
    )
    return row


def run_multi_image_pipeline(images_bgr: list[np.ndarray],
                              names: list[str],
                              method: str,
                              out_dir: str | Path,
                              experiment: str = "baseline",
                              **kwargs) -> tuple[np.ndarray | None, list[dict]]:
    """
    Run the full pipeline on N images to produce a panorama.

    Strategy:
      - Use the centre image as the reference frame
      - Match every other image to the reference
      - Stitch all images into one panorama

    Returns:
        (panorama_bgr, list_of_result_rows)
    """
    from src.stitching import stitch_images
    from src.homography import estimate_homography

    out_dir  = Path(out_dir)
    n        = len(images_bgr)
    ref_idx  = n // 2   # Centre image is the reference

    logger.info(f"\nMulti-image pipeline: {n} images, ref={ref_idx}, method={method}")

    # Preprocess all images
    all_color = []
    all_gray  = []
    for img in images_bgr:
        c, g = preprocess(img, equalize=kwargs.get("equalize", False))
        all_color.append(c)
        all_gray.append(g)

    # Describe the reference image once
    feat_ref = detect_and_describe(all_gray[ref_idx], method)

    H_list   = [None] * n
    H_list[ref_idx] = np.eye(3, dtype=np.float64)

    all_rows = []

    for i in range(n):
        if i == ref_idx:
            continue
        pair_name = f"img{i+1}-img{ref_idx+1}"
        feat_i = detect_and_describe(all_gray[i], method)
        match_r = match_descriptors(feat_i, feat_ref,
                                    ratio_threshold=kwargs.get("ratio_threshold", 0.75))
        ransac_r = estimate_homography(feat_i, feat_ref, match_r["good_matches"],
                                       reproj_threshold=kwargs.get("ransac_threshold", 5.0))

        if ransac_r["success"]:
            H_list[i] = ransac_r["H"]

        metrics = compute_metrics(feat_i, feat_ref, match_r, ransac_r)
        metrics["time_total_s"] = feat_i["time_s"] + feat_ref["time_s"] + match_r["time_s"] + ransac_r["time_s"]
        row = build_result_row(experiment, method, pair_name, "multi_image", metrics)
        all_rows.append(row)

    # Stitch
    try:
        panorama, s_info = stitch_images(all_color, H_list, ref_idx,
                                         alpha_blend=True, crop_borders=True)
        pano_path = out_dir / f"panorama_{method}_full.png"
        import cv2 as _cv2
        out_dir.mkdir(parents=True, exist_ok=True)
        _cv2.imwrite(str(pano_path), panorama)
        logger.info(f"Panorama saved: {pano_path}")
        return panorama, all_rows
    except Exception as e:
        logger.error(f"Stitching failed: {e}")
        return None, all_rows


def _failed_row(experiment, method, pair_name, condition, reason,
                feat1=None, feat2=None, match_result=None) -> dict:
    """Build a failure result row with whatever data is available."""
    return {
        "experiment":  experiment,
        "method":      method,
        "image_pair":  pair_name,
        "condition":   condition,
        "num_kp_img1": feat1["num_kp"] if feat1 else 0,
        "num_kp_img2": feat2["num_kp"] if feat2 else 0,
        "num_good_matches": match_result["num_good_matches"] if match_result else 0,
        "num_ransac_inliers": 0,
        "inlier_ratio": 0.0,
        "homography_success": False,
        "failure_reason": reason,
    }
