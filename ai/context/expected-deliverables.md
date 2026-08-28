# Expected Deliverables

## Purpose

This document defines the complete set of deliverables that must exist at the end of the project. It covers the code structure, outputs, results, and report. Use this as a final checklist before submission.

---

## 1. Repository Directory Structure

The final repository must be organised as follows:

```
Feature-Based-Image-Matching-and-Automatic-Panorama-Construction-Problem/
│
├── ai/                              ← AI workspace (this directory — do not modify during impl)
│   ├── README.md
│   ├── context/
│   ├── skills/
│   └── rules/
│
├── data/                            ← Image datasets
│   ├── README.md                   ← Dataset documentation
│   ├── baseline/
│   ├── rotation/
│   ├── scale/
│   ├── viewpoint/
│   └── illumination/
│
├── src/                             ← Source code (modular)
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py               ← Image loading and validation
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── prepare.py              ← Grayscale, resize, equalization
│   ├── features/
│   │   ├── __init__.py
│   │   ├── detection.py            ← Keypoint detection (SIFT, ORB, Harris)
│   │   └── description.py          ← Descriptor computation
│   ├── matching/
│   │   ├── __init__.py
│   │   └── matcher.py              ← BFMatcher, FLANN, ratio test, cross-check
│   ├── homography/
│   │   ├── __init__.py
│   │   ├── ransac.py               ← RANSAC application
│   │   └── estimate.py             ← Homography estimation and verification
│   ├── warping/
│   │   ├── __init__.py
│   │   └── warp.py                 ← Image warping with warpPerspective
│   ├── stitching/
│   │   ├── __init__.py
│   │   └── stitch.py               ← Multi-image panorama construction
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py              ← Metric computation and recording
│   └── visualization/
│       ├── __init__.py
│       └── draw.py                 ← All visualization functions
│
├── experiments/
│   ├── __init__.py
│   ├── config.py                   ← Shared experiment configuration
│   ├── run_baseline.py             ← Baseline experiment runner
│   ├── run_rotation.py             ← Rotation experiment runner
│   ├── run_scale.py                ← Scale experiment runner
│   ├── run_viewpoint.py            ← Viewpoint experiment runner
│   ├── run_illumination.py         ← Illumination experiment runner
│   ├── compare_methods.py          ← Method comparison runner
│   └── run_all.py                  ← Run all experiments sequentially
│
├── outputs/                         ← Generated outputs (gitignored or tracked selectively)
│   ├── baseline/
│   │   ├── SIFT/
│   │   └── ORB/
│   ├── rotation/
│   ├── scale/
│   ├── viewpoint/
│   ├── illumination/
│   └── comparison/
│
├── results/                         ← Aggregated results (tracked in git)
│   ├── baseline_results.csv
│   ├── rotation_results.csv
│   ├── scale_results.csv
│   ├── viewpoint_results.csv
│   ├── illumination_results.csv
│   └── comparison_table.csv
│
├── report/                          ← Academic report
│   ├── report.pdf                  ← Final compiled report
│   └── figures/                    ← Key figures referenced in report
│
├── tests/                           ← Unit tests
│   ├── test_loader.py
│   ├── test_detection.py
│   ├── test_matching.py
│   └── test_homography.py
│
├── requirements.txt                 ← Python dependencies
├── README.md                        ← Main project README
└── .gitignore
```

---

## 2. Source Code Deliverables

### Required Modules

Every module listed in the directory structure above must exist and be functional. Placeholder files with pass statements are not acceptable.

### Module Responsibilities

#### `src/data/loader.py`
- `load_image(path) -> np.ndarray`: Load a single image, verify not None
- `load_image_set(directory, extensions=['.jpg','.png']) -> list`: Load all images from directory in sorted order
- `validate_image(img) -> bool`: Check image is valid, non-empty, correct dtype

#### `src/preprocessing/prepare.py`
- `to_grayscale(img) -> np.ndarray`: BGR to grayscale
- `resize_image(img, target_size) -> np.ndarray`: Resize with aspect ratio handling
- `equalize_histogram(gray) -> np.ndarray`: Apply CLAHE
- `preprocess_pipeline(img, config) -> tuple(color, gray)`: Apply full preprocessing chain

#### `src/features/detection.py`
- `create_detector(method: str, config: dict) -> cv2.Feature2D`: Factory for SIFT, ORB, etc.
- `detect_keypoints(img_gray, detector) -> list`: Detect keypoints
- `detect_all_methods(img_gray, methods, configs) -> dict`: Detect with all configured methods

#### `src/features/description.py`
- `compute_descriptors(img_gray, keypoints, detector) -> tuple(kp, desc)`: Compute descriptors
- `describe_all_methods(img_gray, kp_dict, detectors) -> dict`: Describe with all methods

#### `src/matching/matcher.py`
- `create_matcher(method: str, config: dict) -> cv2.DescriptorMatcher`: Factory for BF/FLANN
- `match_descriptors(desc1, desc2, matcher, method) -> list`: Match and filter
- `apply_ratio_test(matches, threshold=0.75) -> list`: Lowe's ratio test
- `apply_cross_check(matches) -> list`: Cross-check filtering

#### `src/homography/ransac.py`
- `apply_ransac(kp1, kp2, matches, config) -> tuple(H, mask, inlier_kp1, inlier_kp2)`
- `count_inliers(mask) -> int`
- `compute_inlier_ratio(mask, total_matches) -> float`

#### `src/homography/estimate.py`
- `estimate_homography(src_pts, dst_pts) -> np.ndarray`: Wrapper around findHomography
- `verify_homography(H, img1, img2) -> dict`: Map corners, compute reprojection error
- `save_homography(H, path)`: Save 3×3 matrix to text file

#### `src/warping/warp.py`
- `compute_canvas_size(img1, img2, H) -> tuple(width, height, offset)`: Bounding box
- `warp_image(img, H, canvas_size) -> np.ndarray`: Apply warpPerspective
- `create_panorama_canvas(images, H_list) -> np.ndarray`: Multi-image canvas

#### `src/stitching/stitch.py`
- `stitch_pair(img1, img2, H) -> np.ndarray`: Stitch two images
- `stitch_multiple(images, H_list, reference_idx) -> np.ndarray`: Stitch N images
- `blend_images(base, overlay, mask) -> np.ndarray`: Simple blending

#### `src/evaluation/metrics.py`
- `compute_metrics(kp1, kp2, matches, mask, timings) -> dict`: Compute all M1–M8
- `save_metrics(metrics, path)`: Save to CSV
- `load_all_results(results_dir) -> pd.DataFrame`: Aggregate CSV files

#### `src/visualization/draw.py`
- `visualize_keypoints(img, keypoints, title, outpath)`
- `visualize_matches(img1, kp1, img2, kp2, matches, title, outpath)`
- `visualize_before_after_ransac(img1, kp1, img2, kp2, matches, mask, outpath)`
- `visualize_panorama(panorama, title, outpath)`
- `plot_comparison_table(df, outpath)`: Bar charts comparing methods
- `plot_metric_vs_condition(df, metric, condition, outpath)`: Line plots for experiments

---

## 3. Experiment Deliverables

### Required Experiments and Output Files

| Experiment | Runner Script | Output Directory | Required CSV |
|---|---|---|---|
| Baseline | `experiments/run_baseline.py` | `outputs/baseline/` | `results/baseline_results.csv` |
| Rotation | `experiments/run_rotation.py` | `outputs/rotation/` | `results/rotation_results.csv` |
| Scale | `experiments/run_scale.py` | `outputs/scale/` | `results/scale_results.csv` |
| Viewpoint | `experiments/run_viewpoint.py` | `outputs/viewpoint/` | `results/viewpoint_results.csv` |
| Illumination | `experiments/run_illumination.py` | `outputs/illumination/` | `results/illumination_results.csv` |
| Comparison | `experiments/compare_methods.py` | `outputs/comparison/` | `results/comparison_table.csv` |

### Required Visualizations (per experiment, per method)

- `keypoints_img1_{method}.png`
- `keypoints_img2_{method}.png`
- `raw_matches_{method}.png`
- `ransac_inliers_{method}.png`
- `before_after_ransac_{method}.png`
- `warped_{method}.png`
- `panorama_{method}.png`

---

## 4. Report Deliverables

The final academic report must contain these sections (can be Word/PDF):

| Section | Content |
|---|---|
| Abstract | 200–300 words summarising the project |
| 1. Introduction | Problem definition, motivation, objectives |
| 2. Literature Review | Key algorithms with citations (SIFT, ORB, RANSAC, panorama) |
| 3. Dataset / Image Acquisition | Images used, scene description, capture method |
| 4. Image Preparation | Preprocessing steps applied and justification |
| 5. Feature Detection | Methods used, parameter choices, keypoint visualizations |
| 6. Feature Description | Descriptor types, distance metrics, properties |
| 7. Feature Matching | Matching strategy, ratio test, initial correspondence figures |
| 8. RANSAC | Algorithm parameters, before/after figures, inlier analysis |
| 9. Homography Estimation | H matrix discussion, verification method |
| 10. Image Warping | Canvas computation, warpPerspective application |
| 11. Panorama Construction | Stitching strategy, blending, final panorama figures |
| 12. Experimental Design | How experiments were structured and controlled |
| 13. Quantitative Results | Full results tables (from CSV), all M1–M8 metrics |
| 14. Algorithm Comparison | SIFT vs ORB, discussion of trade-offs |
| 15. Rotation Experiment | Results, figures, discussion |
| 16. Scale Experiment | Results, figures, discussion |
| 17. Viewpoint Experiment | Results, figures, discussion |
| 18. Illumination Experiment | Results, figures, discussion |
| 19. Failure Analysis | Documented failures, root causes |
| 20. Limitations | Honest discussion of what the system cannot handle |
| 21. Discussion | Interpretation of results, insights |
| 22. Conclusion | Summary of findings |
| 23. Future Work | What could be improved or extended |
| References | Properly formatted academic references |
| Appendix A | Full source code listing (or reference to GitHub) |
| Appendix B | Experiment configuration parameters |

---

## 5. README.md (Main Project README)

The project root `README.md` must contain:

1. **Project title and description**
2. **Requirements** — Python version, library versions
3. **Installation instructions** — `pip install -r requirements.txt`
4. **Dataset setup** — where to put images, how to acquire them
5. **Running the baseline experiment** — exact command
6. **Running all experiments** — exact command
7. **Interpreting output** — where results are saved
8. **Repository structure** — directory map
9. **Known limitations**
10. **References**

---

## 6. requirements.txt

Must include pinned versions (or minimum versions) of:

```
opencv-python>=4.5.0
numpy>=1.20.0
matplotlib>=3.3.0
pandas>=1.2.0
scikit-image>=0.18.0
scipy>=1.6.0
```

---

*Every item in this deliverables list is non-negotiable for a complete project submission.*
