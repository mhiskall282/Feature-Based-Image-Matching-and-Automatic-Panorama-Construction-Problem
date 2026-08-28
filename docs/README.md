# CSCD608: Advanced Computer Vision — Technical Documentation & User Manual

Welcome to the comprehensive technical documentation for the **Feature-Based Image Matching and Automatic Panorama Construction System** developed for the CSCD608 postgraduate examination.

---

## 📑 Table of Contents

### Part I: Getting Started
1. [Project Mandate & Scope](#1-project-mandate--scope)
2. [Examination Requirements Traceability Matrix](#2-examination-requirements-traceability-matrix)
3. [Installation & Quickstart Guide](#3-installation--quickstart-guide)

### Part II: Computer Vision Mathematics & Pipeline
4. [Projective Geometry & 8-DOF Homography](#4-projective-geometry--8-dof-homography)
5. [Image Preprocessing & Normalization Protocol](#5-image-preprocessing--normalization-protocol)
6. [Feature Detection: SIFT (DoG) vs. ORB (FAST)](#6-feature-detection-sift-dog-vs-orb-fast)
7. [Feature Description: Gradient Histograms vs. Steered rBRIEF](#7-feature-description-gradient-histograms-vs-steered-rbrief)
8. [Descriptor Matching & Lowe's Ratio Test](#8-descriptor-matching--lowes-ratio-test)
9. [Robust Geometric Estimation via RANSAC](#9-robust-geometric-estimation-via-ransac)
10. [Homography Matrix Diagnostics & Degeneracy Checks](#10-homography-matrix-diagnostics--degeneracy-checks)
11. [Dynamic Canvas Geometry & Perspective Warping](#11-dynamic-canvas-geometry--perspective-warping)
12. [Multi-Image Stitching & Distance-Weighted Alpha Blending](#12-multi-image-stitching--distance-weighted-alpha-blending)

### Part III: Empirical Research & Benchmarks
13. [Quantitative 8-Metric Evaluation Framework](#13-quantitative-8-metric-evaluation-framework)
14. [5 Research Benchmark Experiments & Comparative Analysis](#14-5-research-benchmark-experiments--comparative-analysis)
15. [Before vs. After RANSAC In-Depth Analysis](#15-before-vs-after-ransac-in-depth-analysis)

### Part IV: Diagnostics & Recovery
16. [Failure Diagnostics Taxonomy (F1–F5)](#16-failure-diagnostics-taxonomy-f1f5)

### Part V: Application Interfaces & User Manual
17. [Interactive Web Dashboard Manual](#17-interactive-web-dashboard-manual)
18. [CLI Utilities & Automated Benchmark Runners](#18-cli-utilities--automated-benchmark-runners)
19. [Google Colab Integration Guide](#19-google-colab-integration-guide)

### Part VI: Production Cloud Deployment & Verification
20. [Cloud Deployment Guide (Render, Vercel, Docker)](#20-cloud-deployment-guide-render-vercel-docker)
21. [Pytest Test Suite & Validation Report](#21-pytest-test-suite--validation-report)
22. [Academic References](#22-academic-references)

---

## Part I: Getting Started

### 1. Project Mandate & Scope
* **Degree:** MPhil / MSc Computer Science
* **Course:** CSCD608: Advanced Computer Vision (3 Credits)
* **Semester:** Second Semester Examinations 2025/2026
* **Question:** Question 1 — Feature-Based Image Matching and Automatic Panorama Construction

The goal of this system is to identify corresponding visual regions across two or more overlapping planar or perspective images and automatically composite them into a seamless panorama. The pipeline is engineered entirely using classical computer vision primitives without black-box stitching functions (`cv2.Stitcher_create`) or deep learning models.

### 2. Examination Requirements Traceability Matrix

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

### 3. Installation & Quickstart Guide

```bash
# 1. Clone repository
git clone https://github.com/mhiskall282/Feature-Based-Image-Matching-and-Automatic-Panorama-Construction-Problem.git
cd Feature-Based-Image-Matching-and-Automatic-Panorama-Construction-Problem

# 2. Create and activate virtual environment
python -m venv venv
# Windows: .\venv\Scripts\activate
# Linux/macOS: source venv/bin/activate

# 3. Install requirements
pip install -r requirements.txt

# 4. Launch interactive dashboard
python run_app.py
```

---

## Part II: Computer Vision Mathematics & Pipeline

### 4. Projective Geometry & 8-DOF Homography
A homography $H \in \mathbb{R}^{3\times3}$ maps points between two projective image planes:

$$\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} \sim \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

In inhomogeneous Cartesian coordinates:

$$x' = \frac{h_{11}x + h_{12}y + h_{13}}{h_{31}x + h_{32}y + h_{33}}, \quad y' = \frac{h_{21}x + h_{22}y + h_{23}}{h_{31}x + h_{32}y + h_{33}}$$

Because $H$ is defined up to scale, it has **8 degrees of freedom (DOF)**. The Direct Linear Transformation (DLT) sets up the algebraic constraint $A\mathbf{h} = \mathbf{0}$, which is solved via SVD over consensus inliers.

### 5. Image Preprocessing & Normalization Protocol
1. Aspect-preserving resizing (maximum dimension constrained to 1280 px).
2. Grayscale conversion via ITU-R BT.601 standard ($Y = 0.299R + 0.587G + 0.114B$).
3. Image buffer integrity validation.

### 6. Feature Detection: SIFT (DoG) vs. ORB (FAST)
* **SIFT**: Difference-of-Gaussians (DoG) extrema detection across 3 octaves with 3 scale layers per octave. Sub-pixel quadratic Taylor expansion refinement eliminates low-contrast points and unstable edge responses.
* **ORB**: FAST-9 corner detector across an 8-level image pyramid. Harris corner scores rank and retain the top $N=1000$ points.

### 7. Feature Description: Gradient Histograms vs. Steered rBRIEF
* **SIFT (128-D Float32)**: 8-bin gradient orientation histograms computed over a $4\times4$ grid of spatial subregions around the keypoint, normalized to unit $L_2$ length.
* **ORB (256-Bit Binary)**: Orientation angle $\theta = \text{atan2}(m_{01}, m_{10})$ computed from image moments $m_{pq}$. Evaluates 256 steered pairwise binary intensity tests $\tau(p; \mathbf{x}_i, \mathbf{y}_i)$.

### 8. Descriptor Matching & Lowe's Ratio Test
* **SIFT**: Euclidean ($L_2$) distance with Lowe's ratio test ($d_1 < 0.75 \cdot d_2$).
* **ORB**: Hamming distance with bidirectional cross-check matching.

### 9. Robust Geometric Estimation via RANSAC
1. Randomly sample 4 point pairs.
2. Solve candidate $H_{cand}$ via DLT.
3. Compute transfer reprojection error: $e_i = \|\mathbf{x}'_i - \text{proj}(H_{cand}\mathbf{x}_i)\|_2$.
4. Classify points as inliers if $e_i < 5.0\text{ px}$.
5. Re-estimate $H$ on all inliers via SVD least-squares refinement.

### 10. Homography Matrix Diagnostics & Degeneracy Checks
* **Determinant Check**: $\det(H) \in [10^{-6}, 10^6]$.
* **Condition Number**: $\text{cond}(H) < 10^8$.
* **Corner Bounds Sanity**: Verifies warped corners remain within canvas boundaries.

### 11. Dynamic Canvas Geometry & Perspective Warping
$$x_{offset} = \max(0, -\min(x_{warped})), \quad y_{offset} = \max(0, -\min(y_{warped}))$$
$$T = \begin{bmatrix} 1 & 0 & x_{offset} \\ 0 & 1 & y_{offset} \\ 0 & 0 & 1 \end{bmatrix}, \quad H_{adjusted} = T \cdot H$$

### 12. Multi-Image Stitching & Distance-Weighted Alpha Blending
In overlap regions, pixel values are blended using Euclidean distance transforms:
$$I_{blend}(x, y) = \frac{D_1(x, y)}{D_1(x, y) + D_2(x, y)} I_1(x, y) + \frac{D_2(x, y)}{D_1(x, y) + D_2(x, y)} I_2(x, y)$$

---

## Part III: Empirical Research & Benchmarks

### 13. Quantitative 8-Metric Evaluation Framework
* **M1: Keypoint Count**: Total detected interest points per view.
* **M2: Match Count**: Initial filtered correspondences.
* **M3: RANSAC Inlier Count**: Number of geometrically verified correspondences.
* **M4: Inlier Ratio**: Ratio of inliers to good matches ($N_{inliers} / N_{matches}$).
* **M5: Execution Timing**: Stage-wise profiling (detection, matching, RANSAC, total).
* **M6: Panorama Quality**: Non-black pixel coverage ratio and Laplacian gradient sharpness.
* **M7: Mean Reprojection Error**: Average pixel distance of inlier projections.
* **M8: Homography Success**: Boolean status with failure root cause logging.

### 14. 5 Research Benchmark Experiments & Comparative Analysis

| Benchmark Experiment | SIFT Inlier Ratio | ORB Inlier Ratio | SIFT Reproj Error | ORB Reproj Error | SIFT Latency | ORB Latency |
|---|---|---|---|---|---|---|
| **Baseline (3 Images)** | **59.64%** | 37.74% | **0.14 px** | 0.93 px | 1.82 s | 2.06 s |
| **Rotation ($0^\circ$–$180^\circ$)** | **94.57%** | 91.74% | **0.17 px** | 1.15 px | 2.24 s | **1.87 s** |
| **Scale ($0.5\times$–$2.0\times$)** | 96.24% | **97.23%** | **0.18 px** | 0.23 px | 2.52 s | **2.40 s** |
| **Viewpoint Shear** | **90.55%** | 84.87% | **0.28 px** | 0.97 px | 4.90 s | **4.46 s** |
| **Illumination ($\Delta\beta, \alpha$)** | **91.50%** | 87.43% | **0.27 px** | 0.38 px | 4.58 s | **4.36 s** |

### 15. Before vs. After RANSAC In-Depth Analysis
Before RANSAC, raw matches contain false correspondences. RANSAC eliminates outliers and ensures only valid geometric consensus pairs determine the projective warp.

---

## Part IV: Diagnostics & Recovery

### 16. Failure Diagnostics Taxonomy (F1–F5)
* **F1: Low Keypoints ($N < 4$)**: Textureless scenes or blur. Aborts estimation.
* **F2: Low Overlap ($N_{matches} < 4$)**: Overlap $< 15\%$. Prevents rank-deficient DLT solver.
* **F3: Outlier Dominance ($> 95\%$)**: RANSAC fails to find consensus. Flags failure in log.
* **F4: Degenerate Homography**: Collinear points ($\det(H) \notin [10^{-6}, 10^6]$).
* **F5: Depth Parallax**: 3D camera translation in non-planar scenes causes local ghosting.

---

## Part V: Application Interfaces & User Manual

### 17. Interactive Web Dashboard Manual
* **Preset Scenes**: 1-click sample scene loading.
* **Custom Upload**: Drag-and-drop 2 to 6 photos.
* **Dual Benchmark Mode**: Runs SIFT and ORB side-by-side.
* **Live Stress Playground**: Interactive sliders for rotation, scale, shear, and illumination.

### 18. CLI Utilities & Automated Benchmark Runners
```bash
# Run SIFT CLI pipeline
python scripts/run_pipeline.py --algorithm sift --data data/raw --output outputs/sift

# Run all 5 research experiments
python experiments/run_all.py
```

### 19. Google Colab Integration Guide
Open `notebooks/analysis.ipynb` in Google Colab to run all pipeline stages interactively.

---

## Part VI: Production Cloud Deployment & Verification

### 20. Cloud Deployment Guide (Render, Vercel, Docker)
* **Render**: Detects `render.yaml` and deploys automatically.
* **Vercel**: Routes through serverless entrypoint `api/index.py`.
* **Docker**: `docker build -t panorama-app . && docker run -p 5000:5000 panorama-app`.

### 21. Pytest Test Suite & Validation Report
```bash
pytest tests/ -v
# 31 passed in 1.40s (100% passing)
```

### 22. Academic References
1. **Lowe, D. G. (2004).** Distinctive image features from scale-invariant keypoints. *IJCV*, 60(2), 91–110.
2. **Rublee, E., et al. (2011).** ORB: An efficient alternative to SIFT or SURF. In *IEEE ICCV* (pp. 2564–2571).
3. **Fischler, M. A., & Bolles, R. C. (1981).** Random sample consensus: a paradigm for model fitting. *CACM*, 24(6), 381–395.
4. **Brown, M., & Lowe, D. G. (2007).** Automatic panoramic image stitching using invariant features. *IJCV*, 74(1), 59–73.
5. **Hartley, R., & Zisserman, A. (2004).** *Multiple View Geometry in Computer Vision* (2nd ed.). Cambridge University Press.
