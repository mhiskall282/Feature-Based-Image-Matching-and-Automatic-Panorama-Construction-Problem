# Examination Requirements & Requirement Traceability Matrix

## Purpose

This document lists every formal requirement from the CSCD608 examination question and maps each to:
- An **implementation component** (Python module and function)
- An **experiment** (named experiment in `/experiments/`)
- An **output/visualization** (file to be generated)
- An **evaluation metric** (where applicable)
- A **report section** (section in the final academic report)

**Traceability rule**: Every requirement must be demonstrably fulfilled. During implementation, each module must include a comment referencing the requirement ID(s) it satisfies (e.g., `# REQ-03: Detect distinctive keypoints`).

---

## Requirement Traceability Matrix

### REQ-01 — Image Acquisition

**Examination text**: *"Acquire at least three overlapping images of the same scene from different viewpoints."*

| Field | Detail |
|---|---|
| Module | `src/data/loader.py` |
| Function | `load_image_set(directory)` |
| Experiment | `experiments/baseline/` |
| Output | `outputs/baseline/input_images/` (PNG files of all input images) |
| Metric | Count of images loaded; overlap region estimated visually |
| Report section | §3 Dataset / Image Acquisition |

**Implementation notes**:
- At least 3 images required; 4–5 recommended for richer panorama
- Images must have genuine spatial overlap (≥30% overlap recommended for reliable matching)
- Must be documented: camera/device used, scene description, approximate viewpoint angles

---

### REQ-02 — Image Preparation

**Examination text**: *"Perform necessary image preparation."*

| Field | Detail |
|---|---|
| Module | `src/preprocessing/prepare.py` |
| Function | `preprocess_image(img, config)` |
| Experiment | Part of all experiments |
| Output | `outputs/baseline/preprocessed/` |
| Metric | Resolution after resize, histogram stats before/after equalization |
| Report section | §4 Image Preparation |

**Implementation notes**:
- Convert to grayscale for feature detection (keep colour copy for final panorama)
- Optional: resize to uniform dimensions if images differ significantly
- Optional: apply histogram equalization for illumination experiments
- Optional: apply Gaussian blur to reduce noise
- All preprocessing choices must be documented and justified

---

### REQ-03 — Keypoint Detection

**Examination text**: *"Detect distinctive keypoints using an appropriate feature detector studied in the course."*

| Field | Detail |
|---|---|
| Module | `src/features/detection.py` |
| Function | `detect_keypoints(img, method, config)` |
| Experiment | All experiments; comparison experiment |
| Output | `outputs/baseline/keypoints/keypoints_{method}_{image}.png` |
| Metric | Number of detected keypoints (reported per image, per method) |
| Report section | §5 Feature Detection |

**Implementation notes**:
- Must implement at least two methods (SIFT and ORB mandatory)
- Harris or FAST may serve as a third for richer comparison
- Record: keypoint count, response/strength distribution, scale distribution
- Visualize: draw detected keypoints overlaid on the image

---

### REQ-04 — Descriptor Computation

**Examination text**: *"Compute descriptors."*

| Field | Detail |
|---|---|
| Module | `src/features/description.py` |
| Function | `compute_descriptors(img, keypoints, method, config)` |
| Experiment | All experiments |
| Output | Descriptor arrays saved as `.npy` files in `outputs/baseline/descriptors/` |
| Metric | Descriptor dimensionality, descriptor type (float/binary), computation time |
| Report section | §6 Feature Description |

**Implementation notes**:
- SIFT: 128-dim float32 descriptors; L2 distance metric
- ORB: 256-bit binary descriptors; Hamming distance metric
- Do not mix descriptor types with incompatible distance metrics

---

### REQ-05 — Descriptor Matching

**Examination text**: *"Match descriptors between overlapping image pairs."*

| Field | Detail |
|---|---|
| Module | `src/matching/matcher.py` |
| Function | `match_descriptors(desc1, desc2, method, config)` |
| Experiment | All experiments |
| Output | `outputs/baseline/matches/raw_matches_{method}_{pair}.png` |
| Metric | Number of raw matches before filtering |
| Report section | §7 Feature Matching |

**Implementation notes**:
- SIFT: BFMatcher with L2 norm, or FLANN-based matcher
- ORB: BFMatcher with Hamming norm, cross-check enabled
- Apply Lowe's ratio test for SIFT (threshold ~0.7–0.75)
- Apply cross-check for ORB
- Report: raw match count, post-filter match count

---

### REQ-06 — Display Initial Correspondences

**Examination text**: *"Display initial feature correspondences."*

| Field | Detail |
|---|---|
| Module | `src/visualization/draw.py` |
| Function | `visualize_matches(img1, kp1, img2, kp2, matches, title, outpath)` |
| Experiment | All experiments |
| Output | `outputs/baseline/matches/initial_correspondences_{method}_{pair}.png` |
| Metric | Visual quality of displayed correspondences |
| Report section | §7 Feature Matching (figure) |

**Implementation notes**:
- Use `cv2.drawMatches()` to draw lines between corresponding points
- Limit display to top N matches if count is very high (e.g., top 100) for clarity
- Include method name and image pair in the figure title
- Save to file AND display interactively during development

---

### REQ-07 — RANSAC Outlier Rejection

**Examination text**: *"Apply RANSAC to eliminate incorrect correspondences."*

| Field | Detail |
|---|---|
| Module | `src/homography/ransac.py` |
| Function | `apply_ransac(kp1, kp2, matches, config)` |
| Experiment | All experiments |
| Output | `outputs/baseline/ransac/ransac_inliers_{method}_{pair}.png` |
| Metric | Inlier count, outlier count, inlier ratio |
| Report section | §8 RANSAC |

**Implementation notes**:
- Use `cv2.findHomography(points1, points2, cv2.RANSAC, reprojThreshold)`
- Reprojection threshold: typically 4–5 pixels
- Record mask returned by `findHomography` to separate inliers from outliers
- RANSAC is mandatory — never skip and claim results are RANSAC-filtered

---

### REQ-08 — Homography Estimation

**Examination text**: *"Estimate the homography matrix."*

| Field | Detail |
|---|---|
| Module | `src/homography/estimate.py` |
| Function | `estimate_homography(src_pts, dst_pts)` |
| Experiment | All experiments |
| Output | `outputs/baseline/homography/H_{method}_{pair}.txt` (save H matrix) |
| Metric | Reprojection error on inliers; RANSAC inlier count |
| Report section | §9 Homography Estimation |

**Implementation notes**:
- H is a 3×3 matrix; print and save it
- Discuss what the matrix encodes (rotation, translation, shear, perspective)
- For verification: apply H to corner points of image 1 and verify they map to expected locations in image 2
- For synthetic experiments (rotation/scale), compare estimated H to known ground-truth H

---

### REQ-09 — Image Warping

**Examination text**: *"Transform one image into another image's coordinate system."*

| Field | Detail |
|---|---|
| Module | `src/warping/warp.py` |
| Function | `warp_image(img, H, output_size)` |
| Experiment | All experiments |
| Output | `outputs/baseline/warped/warped_{method}_{pair}.png` |
| Metric | Visual alignment quality; overlap region quality |
| Report section | §10 Image Warping |

**Implementation notes**:
- Use `cv2.warpPerspective(img, H, (width, height))`
- Output canvas must be large enough to contain both images after warping
- Compute canvas size from the transformed corners of the source image
- Show warped image alongside original for direct comparison

---

### REQ-10 — Panorama Construction

**Examination text**: *"Stitch transformed images into a panorama."*

| Field | Detail |
|---|---|
| Module | `src/stitching/stitch.py` |
| Function | `stitch_images(images, homographies, config)` |
| Experiment | All experiments |
| Output | `outputs/baseline/panorama/panorama_{method}.png` |
| Metric | Panorama dimensions; seam visibility; ghosting artifacts |
| Report section | §11 Panorama Construction |

**Implementation notes**:
- For 3+ images: chain homographies to a reference image (typically the centre image)
- Handle canvas offset: if warped image extends left of the canvas origin, apply a translation correction
- Simple blending: overwrite or average in overlap regions
- Advanced blending: alpha blending or multi-band (optional but noted)
- Crop or mask black border regions in the final output

---

### REQ-11 — RANSAC Comparison Visualization

**Examination text**: *"Compare feature matching before and after RANSAC."*

| Field | Detail |
|---|---|
| Module | `src/visualization/draw.py` |
| Function | `visualize_before_after_ransac(img1, kp1, img2, kp2, all_matches, inlier_mask, outpath)` |
| Experiment | All experiments |
| Output | `outputs/baseline/ransac/before_after_ransac_{method}_{pair}.png` |
| Metric | Visual outlier density before vs inlier set after |
| Report section | §8 RANSAC (figure) |

**Implementation notes**:
- Side-by-side or top/bottom layout showing raw matches vs RANSAC inliers
- Use distinct colours for inliers (green) and outliers (red) if overlaid on same image
- Include match counts in the figure title

---

### REQ-12 — Robustness Experiments

**Examination text**: *"Investigate performance under: rotation, scale changes, viewpoint changes, illumination changes."*

| Field | Detail |
|---|---|
| Module | `experiments/run_experiments.py` |
| Functions | `run_rotation_experiment()`, `run_scale_experiment()`, `run_viewpoint_experiment()`, `run_illumination_experiment()` |
| Experiments | `experiments/rotation/`, `experiments/scale/`, `experiments/viewpoint/`, `experiments/illumination/` |
| Output | Results CSV + plots for each experiment |
| Metric | All standard metrics across conditions |
| Report sections | §15–18 |

**Implementation notes**: See `/ai/context/evaluation-framework.md` and `/ai/skills/experimental-evaluation.md` for full experiment design.

---

### REQ-13 — Multi-Method Comparison

**Examination text**: *"Experimental comparison must include at least TWO feature detection/description approaches."*

| Field | Detail |
|---|---|
| Module | `experiments/compare_methods.py` |
| Output | `outputs/comparison/comparison_table.csv`, `outputs/comparison/comparison_figure.png` |
| Metric | All 6 metrics: keypoints, initial matches, RANSAC inliers, inlier ratio, time, panorama quality |
| Report section | §14 Algorithm Comparison |

---

### REQ-14 — Per-Method Metrics

**Examination text**: *"For each approach report: number of detected keypoints / initial matches / RANSAC inliers / inlier ratio / processing time / quality of final panorama."*

| Metric | How Measured |
|---|---|
| Detected keypoints | `len(keypoints)` for each image |
| Initial matches | Count of matches before ratio test / cross-check |
| RANSAC inliers | Count of inlier flags in RANSAC mask |
| Inlier ratio | `inliers / initial_matches` |
| Processing time | `time.perf_counter()` around each stage |
| Panorama quality | Visual inspection + optional SSIM/PSNR in overlap region |

---

### REQ-15 — Pipeline Demonstration

**Examination text**: *"The final system must clearly demonstrate the relationship: Feature Detection → Feature Description → Feature Matching → RANSAC → Homography → Image Alignment → Panorama."*

| Field | Detail |
|---|---|
| Output | `outputs/pipeline_diagram.png` — a visual pipeline figure |
| Output | Console/log output showing each stage completing with metrics |
| Report section | §1 Introduction (pipeline figure), or dedicated pipeline diagram |

---

## Summary Checklist

Use this checklist during implementation review to confirm all requirements are satisfied:

- [ ] REQ-01: ≥3 overlapping images acquired and loaded
- [ ] REQ-02: Preprocessing applied (grayscale, resize, optional equalization)
- [ ] REQ-03: Keypoints detected with ≥2 methods (SIFT, ORB required)
- [ ] REQ-04: Descriptors computed for each keypoint
- [ ] REQ-05: Descriptors matched between image pairs
- [ ] REQ-06: Initial correspondences visualized and saved
- [ ] REQ-07: RANSAC applied; inlier mask recorded
- [ ] REQ-08: Homography matrix estimated and saved
- [ ] REQ-09: Image warped using estimated H
- [ ] REQ-10: Multi-image panorama constructed
- [ ] REQ-11: Before/after RANSAC comparison visualized
- [ ] REQ-12: Rotation, scale, viewpoint, illumination experiments run
- [ ] REQ-13: ≥2 methods compared experimentally
- [ ] REQ-14: All 6 metrics reported per method
- [ ] REQ-15: Complete pipeline demonstrated end-to-end
