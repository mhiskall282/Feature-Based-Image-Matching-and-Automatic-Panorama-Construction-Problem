"""
app/server.py
=============
Interactive Flask Web Application for CSCD608 Computer Vision Examination.
Serves the web dashboard for interactive image matching, RANSAC visualization,
multi-image panorama construction, and transformation stress testing.

Usage:
    python app/server.py
    python app/server.py --port 5000 --host 0.0.0.0
"""

import argparse
import base64
import io
import json
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file, send_from_directory

# Add parent directory to path so src modules are accessible
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import SUPPORTED_METHODS, RANDOM_SEED
from src.preprocessing import preprocess, validate_image_pair, load_image
from src.features import detect_and_describe
from src.matching import match_descriptors
from src.homography import estimate_homography, diagnose_homography
from src.warping import compute_canvas, warp_image, place_reference, blend_images
from src.stitching import stitch_images, stitch_pair
from src.evaluation import compute_metrics, get_software_versions

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB max upload
UPLOAD_FOLDER = Path(__file__).parent / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)


def np_to_base64(img_bgr: np.ndarray, format: str = ".jpg", quality: int = 90) -> str:
    """Encode OpenCV BGR image to base64 data URI string."""
    if img_bgr is None or img_bgr.size == 0:
        return ""
    params = [int(cv2.IMWRITE_JPEG_QUALITY), quality] if format == ".jpg" else []
    success, buffer = cv2.imencode(format, img_bgr, params)
    if not success:
        return ""
    b64_str = base64.b64encode(buffer).decode("utf-8")
    mime = "image/jpeg" if format == ".jpg" else "image/png"
    return f"data:{mime};base64,{b64_str}"


def draw_keypoints_viz(img_gray: np.ndarray, keypoints: list, method: str) -> np.ndarray:
    """Render rich keypoint circles on grayscale background."""
    kp_img = cv2.drawKeypoints(
        img_gray, keypoints[:600], None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    return kp_img


def draw_matches_viz(img1: np.ndarray, kp1: list,
                     img2: np.ndarray, kp2: list,
                     matches: list, is_inlier: bool = False) -> np.ndarray:
    """Render clean, publication-quality match lines between two views."""
    # For inliers use crisp green; for initial matches use distinct soft gold/cyan
    color = (0, 220, 110) if is_inlier else (0, 165, 255)
    # Display up to 60 evenly distributed matches for visual clarity
    display_matches = matches
    if len(matches) > 60:
        step = len(matches) / 60
        display_matches = [matches[int(i * step)] for i in range(60)]

    img_matches = cv2.drawMatches(
        img1, kp1, img2, kp2, display_matches, None,
        matchColor=color,
        singlePointColor=(120, 120, 120),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    return img_matches


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Render the master SPA dashboard."""
    return render_template("index.html")


@app.route("/docs")
@app.route("/docs/")
@app.route("/documentation")
@app.route("/documentation/")
@app.route("/doc")
@app.route("/doc/")
@app.route("/manual")
@app.route("/guide")
def documentation():
    """Render the in-depth GitBook-style documentation portal."""
    return render_template("docs.html")


@app.route("/outputs/<path:filename>")
def serve_outputs(filename):
    """Serve visual outputs, panoramas, and comparison artifacts."""
    outputs_dir = Path(__file__).parent.parent / "outputs"
    return send_from_directory(outputs_dir, filename)


@app.route("/results/<path:filename>")
def serve_results(filename):
    """Serve benchmark plots and CSV results."""
    results_dir = Path(__file__).parent.parent / "results"
    return send_from_directory(results_dir, filename)


@app.route("/api/presets", methods=["GET"])
def get_presets():
    """Return available sample datasets in data/raw."""
    raw_dir = Path(__file__).parent.parent / "data" / "raw"
    images = sorted([p.name for p in raw_dir.glob("*.*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    return jsonify({
        "status": "success",
        "preset_name": "CSCD608 Panoramic City Scene",
        "image_count": len(images),
        "images": images,
    })


@app.route("/api/stitch", methods=["POST"])
def stitch_pipeline():
    """
    Execute the full panorama construction pipeline from uploaded/preset images.
    Returns rich stage-by-stage visualizations, homographies, and metrics.
    """
    try:
        t_start = time.perf_counter()
        data = request.form
        algorithm = data.get("algorithm", "SIFT").upper()
        ratio_thresh = float(data.get("ratio_threshold", 0.75))
        ransac_thresh = float(data.get("ransac_threshold", 5.0))
        use_alpha = data.get("alpha_blend", "true").lower() == "true"
        use_preset = data.get("use_preset", "true").lower() == "true"

        images_bgr = []
        filenames = []

        if use_preset or "files" not in request.files or len(request.files.getlist("files")) == 0:
            # Use data/raw images
            raw_dir = Path(__file__).parent.parent / "data" / "raw"
            paths = sorted([p for p in raw_dir.glob("*.*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
            if len(paths) < 2:
                return jsonify({"status": "error", "message": "At least 2 images required in data/raw."}), 400
            for p in paths:
                img = load_image(p)
                images_bgr.append(img)
                filenames.append(p.name)
        else:
            # Process uploaded files
            uploaded = request.files.getlist("files")
            for f in uploaded:
                if f.filename:
                    save_path = UPLOAD_FOLDER / f.filename
                    f.save(str(save_path))
                    img = load_image(save_path)
                    images_bgr.append(img)
                    filenames.append(f.filename)

        if len(images_bgr) < 2:
            return jsonify({"status": "error", "message": "At least 2 images are required."}), 400

        n = len(images_bgr)
        ref_idx = n // 2  # Centre image as reference coordinate frame

        # Methods to run: either SIFT, ORB, or BOTH
        methods_to_run = ["SIFT", "ORB"] if algorithm == "BOTH" else [algorithm]
        results_by_method = {}

        for method in methods_to_run:
            m_start = time.perf_counter()
            # 1. Preprocessing
            all_color = []
            all_gray = []
            preproc_previews = []
            for i, img in enumerate(images_bgr):
                c, g = preprocess(img)
                all_color.append(c)
                all_gray.append(g)
                preproc_previews.append({
                    "name": filenames[i],
                    "width": c.shape[1],
                    "height": c.shape[0],
                    "color_preview": np_to_base64(c, quality=80),
                    "gray_preview": np_to_base64(g, quality=80),
                })

            # 2. Keypoints & Descriptors
            features = []
            kp_previews = []
            for i, g in enumerate(all_gray):
                feat = detect_and_describe(g, method)
                features.append(feat)
                kp_viz = draw_keypoints_viz(g, feat["keypoints"], method)
                kp_previews.append({
                    "name": filenames[i],
                    "keypoint_count": feat["num_kp"],
                    "descriptor_shape": feat["desc_shape"],
                    "descriptor_dtype": feat["desc_dtype"],
                    "time_s": round(feat["time_s"], 4),
                    "viz": np_to_base64(kp_viz, quality=85),
                })

            # 3. Pairwise Matching & RANSAC to Reference Image
            pair_results = []
            H_list = [None] * n
            H_list[ref_idx] = np.eye(3, dtype=np.float64)

            for i in range(n):
                if i == ref_idx:
                    continue
                feat_i = features[i]
                feat_ref = features[ref_idx]

                # Match
                match_res = match_descriptors(feat_i, feat_ref, ratio_threshold=ratio_thresh)

                # RANSAC
                ransac_res = estimate_homography(feat_i, feat_ref, match_res["good_matches"],
                                                reproj_threshold=ransac_thresh)

                if ransac_res["success"]:
                    H_list[i] = ransac_res["H"]

                # Generate match visualizations
                raw_m_viz = draw_matches_viz(all_gray[i], feat_i["keypoints"],
                                             all_gray[ref_idx], feat_ref["keypoints"],
                                             match_res["good_matches"], is_inlier=False)
                inlier_m_viz = draw_matches_viz(all_gray[i], feat_i["keypoints"],
                                                all_gray[ref_idx], feat_ref["keypoints"],
                                                ransac_res["inlier_matches"], is_inlier=True)

                # Homography diagnostics
                diag = diagnose_homography(ransac_res["H"], all_color[i].shape, all_color[ref_idx].shape) if ransac_res["H"] is not None else {}

                # Metrics
                pair_metrics = compute_metrics(feat_i, feat_ref, match_res, ransac_res)

                pair_results.append({
                    "pair_name": f"{filenames[i]} -> {filenames[ref_idx]}",
                    "src_index": i,
                    "ref_index": ref_idx,
                    "raw_matches": match_res["num_raw_matches"],
                    "good_matches": match_res["num_good_matches"],
                    "inliers": ransac_res["num_inliers"],
                    "outliers": ransac_res["num_outliers"],
                    "inlier_ratio": round(ransac_res["inlier_ratio"] * 100, 2),
                    "reprojection_error": round(ransac_res["reprojection_error"], 3) if ransac_res["reprojection_error"] != float("inf") else None,
                    "homography_success": ransac_res["success"],
                    "homography_matrix": ransac_res["H"].tolist() if ransac_res["H"] is not None else None,
                    "determinant": round(diag.get("determinant", 0.0), 4) if "determinant" in diag else None,
                    "raw_matches_viz": np_to_base64(raw_m_viz, quality=80),
                    "inliers_viz": np_to_base64(inlier_m_viz, quality=80),
                    "matching_time_s": round(match_res["time_s"], 4),
                    "ransac_time_s": round(ransac_res["time_s"], 4),
                })

            # 4. Multi-Image Stitching
            panorama, stitch_info = stitch_images(all_color, H_list, ref_idx,
                                                 alpha_blend=use_alpha, crop_borders=True)

            m_elapsed = time.perf_counter() - m_start

            results_by_method[method] = {
                "method": method,
                "keypoint_summary": kp_previews,
                "pairs": pair_results,
                "panorama_width": panorama.shape[1] if panorama is not None else 0,
                "panorama_height": panorama.shape[0] if panorama is not None else 0,
                "panorama_preview": np_to_base64(panorama, format=".png") if panorama is not None else "",
                "total_time_s": round(m_elapsed, 3),
                "stitch_info": stitch_info,
            }

        total_elapsed = time.perf_counter() - t_start

        return jsonify({
            "status": "success",
            "images_count": n,
            "reference_image_index": ref_idx,
            "preprocessing": preproc_previews,
            "results": results_by_method,
            "total_execution_time_s": round(total_elapsed, 3),
            "software_versions": get_software_versions(),
        })

    except Exception as e:
        logger.exception("Pipeline execution error")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/stress_test", methods=["POST"])
def live_stress_test():
    """
    Execute real-time transformation stress test on reference image.
    Supports interactive rotation, scale, perspective shear, and illumination sliders.
    """
    try:
        data = request.json or request.form
        angle = float(data.get("rotation_angle", 0.0))
        scale = float(data.get("scale_factor", 1.0))
        shear = float(data.get("viewpoint_shear", 0.0))
        brightness = float(data.get("brightness", 0.0))
        contrast = float(data.get("contrast", 1.0))
        method = data.get("algorithm", "SIFT").upper()

        # Load reference image from data/raw
        raw_dir = Path(__file__).parent.parent / "data" / "raw"
        paths = sorted([p for p in raw_dir.glob("*.*") if p.suffix.lower() in [".jpg", ".png"]])
        if not paths:
            return jsonify({"status": "error", "message": "No base images found."}), 400

        ref_img = load_image(paths[0])
        ref_color, ref_gray = preprocess(ref_img)
        h, w = ref_color.shape[:2]

        # Apply synthetic transformations
        transformed = ref_color.copy()

        # 1. Rotation + Scale
        if angle != 0.0 or scale != 1.0:
            cx, cy = w / 2, h / 2
            M_rot = cv2.getRotationMatrix2D((cx, cy), angle, scale)
            cos_a, sin_a = abs(M_rot[0,0]), abs(M_rot[0,1])
            nw = int(h * sin_a + w * cos_a)
            nh = int(h * cos_a + w * sin_a)
            M_rot[0, 2] += nw / 2 - cx
            M_rot[1, 2] += nh / 2 - cy
            transformed = cv2.warpAffine(transformed, M_rot, (nw, nh), borderMode=cv2.BORDER_REFLECT)

        # 2. Viewpoint Shear
        if shear != 0.0:
            th, tw = transformed.shape[:2]
            src_pts = np.float32([[0,0],[tw,0],[tw,th],[0,th]])
            dst_pts = np.float32([[shear, shear],[tw-shear, 0],[tw, th],[0, th-shear]])
            M_proj = cv2.getPerspectiveTransform(src_pts, dst_pts)
            transformed = cv2.warpPerspective(transformed, M_proj, (tw, th), borderMode=cv2.BORDER_REFLECT)

        # 3. Illumination (Brightness + Contrast)
        if brightness != 0.0 or contrast != 1.0:
            transformed = cv2.convertScaleAbs(transformed, alpha=contrast, beta=brightness)

        _, trans_gray = preprocess(transformed)

        # Run detection & description
        feat_ref = detect_and_describe(ref_gray, method)
        feat_trans = detect_and_describe(trans_gray, method)

        # Match & RANSAC
        match_res = match_descriptors(feat_ref, feat_trans, ratio_threshold=0.75)
        ransac_res = estimate_homography(feat_ref, feat_trans, match_res["good_matches"], reproj_threshold=5.0)

        # Visualizations
        inliers_viz = draw_matches_viz(ref_gray, feat_ref["keypoints"],
                                       trans_gray, feat_trans["keypoints"],
                                       ransac_res["inlier_matches"], is_inlier=True)

        return jsonify({
            "status": "success",
            "method": method,
            "transformations": {
                "rotation_deg": angle,
                "scale_factor": scale,
                "viewpoint_shear_px": shear,
                "brightness_delta": brightness,
                "contrast_factor": contrast,
            },
            "ref_keypoints": feat_ref["num_kp"],
            "transformed_keypoints": feat_trans["num_kp"],
            "good_matches": match_res["num_good_matches"],
            "ransac_inliers": ransac_res["num_inliers"],
            "inlier_ratio": round(ransac_res["inlier_ratio"] * 100, 2),
            "reprojection_error": round(ransac_res["reprojection_error"], 3) if ransac_res["reprojection_error"] != float("inf") else None,
            "transformed_preview": np_to_base64(transformed, quality=80),
            "inliers_viz": np_to_base64(inliers_viz, quality=80),
        })

    except Exception as e:
        logger.exception("Stress test error")
        return jsonify({"status": "error", "message": str(e)}), 500


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSCD608 Panorama Web Dashboard")
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on (default 5000)")
    parser.add_argument("--host", default="127.0.0.1", help="Host address (default 127.0.0.1)")
    parser.add_argument("--debug", action="store_true", help="Run in Flask debug mode")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"CSCD608 ADVANCED COMPUTER VISION - INTERACTIVE WEB APP")
    print(f"Server URL: http://{args.host}:{args.port}/")
    print(f"{'='*70}\n")
    app.run(host=args.host, port=args.port, debug=args.debug)
