# Evaluation Framework

## Purpose

This document defines the quantitative evaluation framework for the project. It specifies every metric, how it is measured, what distinguishes measured vs derived vs qualitative values, and how results must be recorded and reported.

**Core principle**: All reported numbers must be computed from actual pipeline output. Nothing must be estimated, rounded up, or fabricated.

---

## 1. Categories of Evidence

All reported evidence must be clearly categorised:

### 1.1 Measured Values
Directly computed from pipeline output:
- Number of detected keypoints (from `len(keypoints)`)
- Number of initial matches (from `len(raw_matches)`)
- Number of RANSAC inliers (from `np.sum(mask)`)
- Processing time (from `time.perf_counter()`)
- Image dimensions (from `img.shape`)

### 1.2 Derived Metrics
Computed mathematically from measured values:
- Inlier ratio = `inliers / raw_matches`
- Match rate = `good_matches / detected_keypoints_in_smaller_image`
- Reprojection error = mean pixel distance of inliers from predicted location

### 1.3 Qualitative Observations
Assessed by visual inspection — explicitly labelled as such:
- "The panorama seam is visible near the left boundary of image 2"
- "SIFT produces more evenly distributed keypoints than ORB in this scene"
- "Ghosting artefacts appear in the overlap region under strong illumination change"

**Never present qualitative observations as if they were measured metrics.**

---

## 2. Standard Metric Definitions

### M1 — Detected Keypoints (per image, per method)

```
metric: num_keypoints_image_N
value:  len(keypoints_N)
unit:   count (integer)
note:   Report separately for each image and each method
```

### M2 — Initial Matches (per image pair, per method)

```
metric: num_initial_matches
value:  len(raw_matches_after_ratio_test_or_crosscheck)
unit:   count (integer)
note:   "Initial" means after basic filtering (ratio test / cross-check)
        but BEFORE RANSAC
```

### M3 — RANSAC Inliers (per image pair, per method)

```
metric: num_ransac_inliers
value:  np.sum(ransac_mask == 1)
unit:   count (integer)
note:   Must use the mask returned by cv2.findHomography()
```

### M4 — Inlier Ratio (per image pair, per method)

```
metric: inlier_ratio
value:  num_ransac_inliers / num_initial_matches
unit:   proportion (0.0 to 1.0)
note:   Report to 3 decimal places
```

### M5 — Processing Time (per stage, per method)

```
metric: processing_time_{stage}
stages: preprocessing, detection, description, matching, ransac,
        warping, stitching, total
value:  time.perf_counter() end - start
unit:   seconds (report to 3 decimal places)
note:   Average over multiple runs if variance is high
        Report hardware: CPU model, RAM, Python/OpenCV versions
```

### M6 — Panorama Quality (qualitative + optional SSIM)

```
metric: panorama_quality
primary: qualitative assessment (visual inspection)
options:
  - ssim: compute in overlap region if reference is available
  - ghost_score: count of visible ghosting artefacts (0=none, 1=minor, 2=major)
  - seam_visibility: 0=invisible, 1=faint, 2=obvious
```

### M7 — Reprojection Error (derived)

```
metric: mean_reprojection_error
value:  mean pixel distance of RANSAC inliers from predicted position
unit:   pixels (report to 2 decimal places)
note:   Lower is better. Should be below the RANSAC threshold.
```

### M8 — Homography Estimation Success

```
metric: homography_success
value:  boolean — True if cv2.findHomography() returns a non-None H
        AND num_ransac_inliers >= minimum_inlier_threshold (e.g., 10)
note:   If H is None or inlier count is below threshold, record as FAILED
```

---

## 3. Standard Results Table (Per Experiment Run)

Every experiment run must produce a CSV file with this schema:

```csv
experiment,method,image_pair,num_kp_img1,num_kp_img2,num_raw_matches,num_ransac_inliers,inlier_ratio,time_detection_s,time_matching_s,time_ransac_s,time_total_s,homography_success,notes
baseline,SIFT,img1-img2,...
baseline,SIFT,img2-img3,...
baseline,ORB,img1-img2,...
baseline,ORB,img2-img3,...
```

This schema applies to all experiments. Experiment-specific columns may be appended:
- Rotation experiments: add column `rotation_angle_deg`
- Scale experiments: add column `scale_factor`
- Illumination experiments: add column `brightness_delta`, `contrast_factor`

---

## 4. Experiment-Specific Metrics

### 4.1 Rotation Experiment Metrics

For each rotation angle θ:
- All standard metrics (M1–M8)
- Additional: `angle_deg` (the applied rotation angle)
- Expected trend: SIFT should maintain performance up to ±180° (fully rotation-invariant). ORB should maintain performance within its pyramid range.

### 4.2 Scale Experiment Metrics

For each scale factor s:
- All standard metrics
- Additional: `scale_factor` (e.g., 0.5, 0.75, 1.25, 1.5, 2.0)
- Expected trend: SIFT should handle significant scale changes (it is scale-invariant by design). ORB (pyramid-based) should handle moderate changes.

### 4.3 Viewpoint Experiment Metrics

For each viewpoint condition:
- All standard metrics
- Additional: `viewpoint_type` (mild, moderate, extreme)
- Note: extreme viewpoint changes may cause homography failure (parallax dominates). Document honestly.

### 4.4 Illumination Experiment Metrics

For each illumination condition:
- All standard metrics
- Additional: `illumination_type` (bright_mild, bright_strong, dark_mild, dark_strong, high_contrast)
- Expected trend: SIFT is more illumination-robust due to descriptor normalization. ORB is more sensitive.

---

## 5. Comparison Table (Method × Condition)

The final comparison must be presented as a structured table:

```
                     SIFT                    ORB
Condition        KP  IM  RI  IR  T(s)    KP  IM  RI  IR  T(s)
Baseline        ...
Rotation 30°    ...
Rotation 90°    ...
Scale 0.5×      ...
Scale 2.0×      ...
Viewpoint mild  ...
Viewpoint extr. ...
Illumin. bright ...
Illumin. dark   ...
```

Where: KP=keypoints, IM=initial matches, RI=RANSAC inliers, IR=inlier ratio, T=total time.

This table must be generated programmatically from the saved CSV files — not typed manually.

---

## 6. Statistical Considerations

For experiments that are non-deterministic (e.g., if RANSAC randomness causes variance):

- Run each configuration **3 times minimum**
- Report **mean ± standard deviation**
- Set a fixed random seed before each run: `np.random.seed(42)`
- Note that `cv2.findHomography` with RANSAC has internal randomness — fixing NumPy seed does not control OpenCV's internal seed
  - Workaround: run multiple times and report statistics

---

## 7. Failure Recording

When the pipeline fails (homography estimation returns None, or fewer than minimum inliers), record:

```csv
...,homography_success=False,notes="Insufficient inliers (N=3). Likely cause: extreme scale change."
```

**Never silently skip failures.** A documented failure is a valid experimental result. It tells the examiner that you understand the algorithm's limitations.

---

## 8. Panorama-Level Quality Assessment

Beyond per-pair metrics, assess the final panorama:

### Visual Quality Checklist
- [ ] No obvious seam lines in the overlap region
- [ ] Colour consistency between images (no strong brightness jump at seam)
- [ ] No ghosting (double-exposure effect) in overlap
- [ ] No missing regions (black holes in canvas)
- [ ] Straight lines in the scene appear straight in the panorama
- [ ] Text in the scene (if any) remains legible

### Quantitative (if reference available)
- SSIM in overlap region: `skimage.metrics.structural_similarity(patch1, patch2)`
- For synthetic experiments with known H: compute reprojection error of ground-truth corners

---

## 9. Output File Requirements

Every experiment must produce:

```
outputs/{experiment_name}/{method}/
├── metrics.csv             ← All numeric results
├── keypoints_img1.png
├── keypoints_img2.png
├── raw_matches.png
├── ransac_inliers.png
├── before_after_ransac.png
├── homography_H.txt        ← The 3×3 H matrix
├── warped_image.png
├── panorama.png
└── experiment_config.json  ← Parameters used (thresholds, seeds, versions)
```

The `experiment_config.json` must record:
```json
{
  "experiment": "baseline",
  "method": "SIFT",
  "images": ["img_01.jpg", "img_02.jpg", "img_03.jpg"],
  "sift_nfeatures": 0,
  "sift_contrast_threshold": 0.04,
  "ratio_test_threshold": 0.75,
  "ransac_reprojection_threshold": 5.0,
  "random_seed": 42,
  "opencv_version": "4.x.x",
  "python_version": "3.x.x",
  "numpy_version": "1.x.x",
  "timestamp": "2026-xx-xx"
}
```

---

*This framework ensures all evaluations are rigorous, reproducible, and honest. Do not deviate without documenting the reason.*
