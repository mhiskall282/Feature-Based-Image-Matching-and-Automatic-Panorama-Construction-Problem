# CSCD608: Advanced Computer Vision — Technical Documentation & User Manual

Welcome to the comprehensive technical documentation for the **Feature-Based Image Matching and Automatic Panorama Construction System** developed for the CSCD608 examination.

---

## 📑 Table of Contents

1. [System Overview & Examination Scope](#1-system-overview--examination-scope)
2. [Examination Requirements Traceability Matrix](#2-examination-requirements-traceability-matrix)
3. [Mathematical Foundations & Projective Geometry](#3-mathematical-foundations--projective-geometry)
4. [Input Image Acquisition & Preprocessing Protocol](#4-input-image-acquisition--preprocessing-protocol)
5. [Feature Detection: SIFT (DoG) vs. ORB (FAST)](#5-feature-detection-sift-dog-vs-orb-fast)
6. [Feature Description: Gradient Histograms vs. Steered rBRIEF](#6-feature-description-gradient-histograms-vs-steered-rbrief)
7. [Descriptor Matching & Verification](#7-descriptor-matching--verification)
8. [Robust Geometric Estimation via RANSAC](#8-robust-geometric-estimation-via-ransac)
9. [Homography Matrix Diagnostics & Degeneracy Verification](#9-homography-matrix-diagnostics--degeneracy-verification)
10. [Dynamic Canvas Geometry & Perspective Warping](#10-dynamic-canvas-geometry--perspective-warping)
11. [Multi-Image Stitching & Distance-Weighted Alpha Blending](#11-multi-image-stitching--distance-weighted-alpha-blending)
12. [Quantitative 8-Metric Evaluation Framework](#12-quantitative-8-metric-evaluation-framework)
13. [Empirical Research Benchmark Results](#13-empirical-research-benchmark-results)
14. [Before vs. After RANSAC In-Depth Analysis](#14-before-vs-after-ransac-in-depth-analysis)
15. [Failure Diagnostics & Edge Case Handling (F1–F5)](#15-failure-diagnostics--edge-case-handling-f1f5)
16. [User Manual: Interactive Web Dashboard](#16-user-manual-interactive-web-dashboard)
17. [CLI Tools & Automated Experiment Runners](#17-cli-tools--automated-experiment-runners)
18. [Cloud Deployment (Render, Vercel, Docker, Hugging Face)](#18-cloud-deployment-render-vercel-docker-hugging-face)
19. [Pytest Test Suite & Validation Report](#19-pytest-test-suite--validation-report)
20. [Academic Integrity Notice & References](#20-academic-integrity-notice--references)

---

## 1. System Overview & Examination Scope

* **Degree:** MPhil / MSc Computer Science
* **Course:** CSCD608: Advanced Computer Vision (3 Credits)
* **Semester:** Second Semester Examinations 2025/2026
* **Question:** Question 1 — Feature-Based Image Matching and Automatic Panorama Construction

The goal of this system is to identify corresponding visual regions across two or more overlapping planar or perspective images and automatically composite them into a seamless panorama. The pipeline is engineered entirely using classical computer vision primitives without black-box stitching functions (`cv2.Stitcher_create`) or deep learning models.

---

## 2. Examination Requirements Traceability Matrix

| Task ID | Examination Specification | Implementation Module | Automated Test / Artifact | Verification Status |
|---|---|---|---|:---:|
| **REQ-01** | Multi-image scene acquisition ($\ge 3$ views) | `src/preprocessing.py` | `data/raw/` | **Verified** |
| **REQ-02** | Aspect-preserving preprocessing & validation | `src/preprocessing.py` | `outputs/baseline/preprocessed_*.png` | **Verified** |
| **REQ-03** | Keypoint detection (SIFT & ORB) | `src/features.py` | `tests/test_features.py` | **Verified** |
| **REQ-04** | Feature description (Float32 & Binary) | `src/features.py` | `tests/test_features.py` | **Verified** |
| **REQ-05** | Descriptor matching with appropriate norms | `src/matching.py` | `tests/test_matching.py` | **Verified** |
| **REQ-06** | Initial correspondences visualization | `src/visualization.py` | `outputs/*/raw_matches_*.png` | **Verified** |
| **REQ-07** | RANSAC outlier rejection (4-pt DLT consensus) | `src/homography.py` | `tests/test_homography.py` | **Verified** |
| **REQ-08** | $3\times3$ Homography matrix estimation & storage | `src/homography.py` | `outputs/*/homographies/` | **Verified** |
| **REQ-09** | Perspective image warping & coordinate shift | `src/warping.py` | `tests/test_warping.py` | **Verified** |
| **REQ-10** | Multi-image panorama stitching & blending | `src/stitching.py` | `outputs/*/panorama/` | **Verified** |
| **REQ-11** | Compare matches Before vs. After RANSAC | `src/visualization.py` | `outputs/*/before_after_*.png` | **Verified** |
| **REQ-12A** | In-plane rotation stress test ($0^\circ$–$180^\circ$) | `experiments/run_rotation.py` | `results/rotation_results.csv` | **Verified** |
| **REQ-12B** | Multi-scale stress test ($0.5\times$–$2.0\times$) | `experiments/run_scale.py` | `results/scale_results.csv` | **Verified** |
| **REQ-12C** | Perspective viewpoint shear stress test | `experiments/run_viewpoint.py` | `results/viewpoint_results.csv` | **Verified** |
| **REQ-12D** | Photometric illumination stress test | `experiments/run_illumination.py` | `results/illumination_results.csv` | **Verified** |
| **REQ-13** | Stage-wise execution timing profiling | `src/evaluation.py` | `results/tables/all_results.csv` | **Verified** |
| **REQ-14** | Quantitative evaluation of SIFT vs. ORB | `scripts/generate_report_tables.py` | `results/tables/comparison_table.csv` | **Verified** |
| **REQ-15** | Complete demonstrable CLI, App & Colab pipeline | `scripts/run_pipeline.py`, `app/server.py` | `tests/test_pipeline.py` | **Verified** |

---

## 3. Mathematical Foundations & Projective Geometry

A homography $H \in \mathbb{R}^{3\times3}$ maps points between two projective image planes:

$$\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} \sim \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

In inhomogeneous Cartesian coordinates:

$$x' = \frac{h_{11}x + h_{12}y + h_{13}}{h_{31}x + h_{32}y + h_{33}}, \quad y' = \frac{h_{21}x + h_{22}y + h_{23}}{h_{31}x + h_{32}y + h_{33}}$$

Because $H$ is defined up to scale, it has **8 degrees of freedom (DOF)**. The Direct Linear Transformation (DLT) sets up the algebraic constraint $A\mathbf{h} = \mathbf{0}$, which is solved via SVD over consensus inliers.

---

## 4. Input Image Acquisition & Preprocessing Protocol

- **Camera Motion**: Pure camera rotation about its optical center or viewing a planar scene.
- **Overlap**: Minimum 30%–50% horizontal overlap between adjacent frames.
- **Preprocessing Pipeline**:
  1. Aspect-preserving resizing (maximum dimension constrained to 1280 px).
  2. Grayscale conversion via ITU-R BT.601 standard ($Y = 0.299R + 0.587G + 0.114B$).
  3. Image buffer integrity validation (ensuring valid non-empty arrays).

---

## 5. Feature Detection: SIFT (DoG) vs. ORB (FAST)

* **SIFT**: Difference-of-Gaussians (DoG) extrema detection across 3 octaves with 3 scale layers per octave. Sub-pixel quadratic Taylor expansion refinement eliminates low-contrast points and unstable edge responses.
* **ORB**: FAST-9 corner detector across an 8-level image pyramid. Harris corner scores rank and retain the top $N=1000$ points.

---

## 6. Feature Description: Gradient Histograms vs. Steered rBRIEF

* **SIFT (128-D Float32)**: 8-bin gradient orientation histograms computed over a $4\times4$ grid of spatial subregions around the keypoint, normalized to unit $L_2$ length.
* **ORB (256-Bit Binary)**: Orientation angle $\theta = \text{atan2}(m_{01}, m_{10})$ computed from image moments $m_{pq}$. Evaluates 256 steered pairwise binary intensity tests $\tau(p; \mathbf{x}_i, \mathbf{y}_i)$.

---

## 7. Descriptor Matching & Verification

* **SIFT**: Euclidean ($L_2$) distance with Lowe's ratio test ($d_1 < 0.75 \cdot d_2$).
* **ORB**: Hamming distance with bidirectional cross-check matching.

---

## 8. Robust Geometric Estimation via RANSAC

1. Randomly sample 4 point pairs.
2. Solve candidate $H_{cand}$ via DLT.
3. Compute transfer reprojection error: $e_i = \|\mathbf{x}'_i - \text{proj}(H_{cand}\mathbf{x}_i)\|_2$.
4. Classify points as inliers if $e_i < 5.0\text{ px}$.
5. Re-estimate $H$ on all inliers via SVD.

---

## 9. Dynamic Canvas Geometry & Warping

To prevent clipping negative warped coordinates:

$$x_{offset} = \max(0, -\min(x_{warped})), \quad y_{offset} = \max(0, -\min(y_{warped}))$$

$$T = \begin{bmatrix} 1 & 0 & x_{offset} \\ 0 & 1 & y_{offset} \\ 0 & 0 & 1 \end{bmatrix}, \quad H_{adjusted} = T \cdot H$$

Images are warped using inverse mapping with bilinear interpolation (`cv2.INTER_LINEAR`).

---

## 10. Multi-Image Stitching & Distance Alpha Blending

In overlap regions, pixel values are blended using Euclidean distance transforms to eliminate seam artifacts:

$$I_{blend}(x, y) = \frac{D_1(x, y)}{D_1(x, y) + D_2(x, y)} I_1(x, y) + \frac{D_2(x, y)}{D_1(x, y) + D_2(x, y)} I_2(x, y)$$

---

## 11. Empirical Research Benchmark Results

Measured data from [`results/tables/comparison_table.csv`](file:///c:/Users/user/Desktop/Feature-Based-Image-Matching-and-Automatic-Panorama-Construction-Problem/results/tables/comparison_table.csv):

| Benchmark Experiment | SIFT Inlier Ratio | ORB Inlier Ratio | SIFT Reproj Error | ORB Reproj Error | SIFT Latency | ORB Latency |
|---|---|---|---|---|---|---|
| **Baseline (3 Images)** | **59.64%** | 37.74% | **0.14 px** | 0.93 px | 1.82 s | 2.06 s |
| **Rotation ($0^\circ$–$180^\circ$)** | **94.57%** | 91.74% | **0.17 px** | 1.15 px | 2.24 s | **1.87 s** |
| **Scale ($0.5\times$–$2.0\times$)** | 96.24% | **97.23%** | **0.18 px** | 0.23 px | 2.52 s | **2.40 s** |
| **Viewpoint Shear** | **90.55%** | 84.87% | **0.28 px** | 0.97 px | 4.90 s | **4.46 s** |
| **Illumination ($\Delta\beta, \alpha$)** | **91.50%** | 87.43% | **0.27 px** | 0.38 px | 4.58 s | **4.36 s** |

---

## 12. Failure Diagnostics Taxonomy (F1–F5)

* **F1: Low Keypoints ($N < 4$)**: Textureless scenes or blur. Aborts estimation.
* **F2: Low Overlap ($N_{matches} < 4$)**: Overlap $< 15\%$. Prevents rank-deficient DLT solver.
* **F3: Outlier Dominance ($> 95\%$)**: RANSAC fails to find consensus. Flags failure in log.
* **F4: Degenerate Homography**: Collinear points ($\det(H) \notin [10^{-6}, 10^6]$).
* **F5: Depth Parallax**: 3D camera translation in non-planar scenes causes local ghosting.

---

## 13. Interactive Web Dashboard User Guide

```bash
# Launch web dashboard
python run_app.py
```
Open **[http://127.0.0.1:5000/](http://127.0.0.1:5000/)** in any browser.

Features:
- 1-click sample scene loading.
- Custom image uploader for 2 to 6 photos.
- Interactive SIFT, ORB, and Dual Benchmark modes.
- Sliders for Lowe's ratio test threshold and RANSAC threshold.
- Interactive step-by-step visualizers and live transformation stress test simulator.
- Direct high-res panorama download.

---

## 14. Cloud Deployment Guide

### Deploy on Render.com (Recommended)
1. In Render Dashboard, click **New → Blueprint**.
2. Connect this repository. Render automatically reads `render.yaml`.

### Deploy on Vercel
1. Import repository into Vercel. `vercel.json` and `api/index.py` handle serverless routing.

### Deploy with Docker
```bash
docker build -t panorama-app .
docker run -p 5000:5000 panorama-app
```

---

## 15. Unit & Integration Test Suite

```bash
pytest tests/ -v
# 31 passed in 1.40s (100% passing)
```

---

## 16. Academic References

1. **Lowe, D. G. (2004).** Distinctive image features from scale-invariant keypoints. *International Journal of Computer Vision*, 60(2), 91–110.
2. **Rublee, E., et al. (2011).** ORB: An efficient alternative to SIFT or SURF. In *IEEE ICCV* (pp. 2564–2571).
3. **Fischler, M. A., & Bolles, R. C. (1981).** Random sample consensus: a paradigm for model fitting. *CACM*, 24(6), 381–395.
4. **Brown, M., & Lowe, D. G. (2007).** Automatic panoramic image stitching using invariant features. *IJCV*, 74(1), 59–73.
5. **Hartley, R., & Zisserman, A. (2004).** *Multiple View Geometry in Computer Vision* (2nd ed.). Cambridge University Press.
