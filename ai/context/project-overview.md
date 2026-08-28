# Project Overview — Feature-Based Image Matching & Automatic Panorama Construction

## Course and Examination Context

- **Course**: CSCD608 Advanced Computer Vision (3 Credits)
- **Programme**: MPhil/MSc Computer Science
- **Institution**: [University — as specified on examination paper]
- **Semester**: Second Semester Examinations, 2025/2026
- **Question**: 1 — Feature-Based Image Matching and Automatic Panorama Construction

This project is a practical examination project requiring implementation of a complete image stitching pipeline from scratch using classical computer vision methods. The work will be assessed on:
- Correctness and depth of the implemented pipeline
- Quality and rigour of the experimental evaluation
- Clarity of the academic report
- Honest treatment of limitations and failures

---

## Problem Statement

**Panoramic image construction** is the process of combining multiple overlapping photographs taken from different viewpoints into a single wide-field image. This is a classical and practically important problem in computer vision with applications in:

- Photography (mobile phone panorama modes)
- Remote sensing and satellite image mosaicking
- Medical imaging (wide-field retinal imaging)
- Robotics (environment mapping)
- Cultural heritage digitization (wide-area scene capture)
- Augmented reality (background scene reconstruction)

The fundamental challenge is: **given two or more images of the same scene taken from different positions/orientations, how do we determine which pixels correspond to the same physical point in 3D space, and how do we transform and blend the images into a coherent whole?**

This is solved using the **feature-based image matching paradigm**, which avoids dense pixel-by-pixel comparison in favour of detecting, describing, and matching sparse but highly distinctive image regions.

---

## The Classical Pipeline

The pipeline this project must implement is:

### Stage 1 — Image Acquisition and Preparation
Obtain at least three overlapping images of the same scene. Apply necessary preprocessing (grayscale conversion, resizing, histogram equalization where relevant).

**Why this stage matters**: The quality of the input images directly determines whether any subsequent stage can succeed. Poor overlap, extreme blur, or near-identical (textureless) scenes will cause pipeline failure.

### Stage 2 — Feature Detection
Identify keypoints — distinctive, locally identifiable image locations. A good detector finds points that are:
- **Repeatable**: Found in both images even under changes in viewpoint, scale, or illumination
- **Localized**: Precisely positioned so correspondence is meaningful
- **Distinctive**: Surrounding regions are unique enough to be matched reliably

Methods studied in CSCD608 and applicable here:
- **SIFT** (Scale-Invariant Feature Transform — Lowe, 2004): Detects scale-space extrema in Difference-of-Gaussian pyramid
- **ORB** (Oriented FAST and Rotated BRIEF — Rublee et al., 2011): Uses FAST keypoint detector with orientation assignment
- **Harris Corner Detector**: Classic edge/corner detection based on second-moment matrix eigenvalues
- **FAST** (Features from Accelerated Segment Test): High-speed corner detection

### Stage 3 — Feature Description
For each detected keypoint, compute a compact numerical descriptor encoding the local image neighbourhood. The descriptor enables matching across images.

- **SIFT descriptor**: 128-dimensional float vector encoding gradient orientations in 4×4 spatial bins around keypoint. Rotation-invariant (orientation normalized). Scale-invariant (computed at detected scale).
- **ORB descriptor (rBRIEF)**: 256-bit binary string encoding intensity comparisons. Extremely fast. Hamming distance for matching.

### Stage 4 — Feature Matching
Find correspondences between descriptors from different images. Two main strategies:
- **Brute-Force Matching (BFMatcher)**: Compare every descriptor against every other. Guaranteed to find the nearest neighbour.
- **FLANN (Fast Library for Approximate Nearest Neighbours)**: Approximate matching using optimized data structures. Faster for large descriptor sets.

**Lowe's Ratio Test** (SIFT): Accept a match only if the distance to the best match is significantly smaller than the distance to the second-best match. Threshold typically 0.7–0.75. Rejects ambiguous matches.

**Cross-check matching** (ORB): Accept a match only if image A's best match in image B is also image B's best match in image A.

### Stage 5 — RANSAC and Homography Estimation
Raw matches contain many false correspondences (outliers). RANSAC (Random Sample Consensus) robustly estimates the geometric transformation (homography matrix H) that best fits the inlier matches.

**Homography**: A 3×3 projective transformation matrix that maps points from one image plane to another. Valid for scenes that are approximately planar OR when the camera undergoes pure rotation (no translation).

RANSAC algorithm:
1. Randomly sample the minimum number of point correspondences needed (4 for homography)
2. Compute the homography from those 4 points
3. Count inliers: points whose reprojection error is below a threshold
4. Repeat for N iterations
5. Return the homography with the most inliers
6. Refine using all inliers

### Stage 6 — Image Warping
Apply the estimated homography to transform one image into the coordinate frame of another using `cv2.warpPerspective()`. This is a perspective (projective) transformation.

### Stage 7 — Image Alignment and Panorama Construction
Composite the warped image(s) into a single canvas:
- Determine the output canvas size (must accommodate all images)
- Apply offset translation if needed so no image goes out of frame
- Handle overlapping regions (simple averaging, or multi-band blending)
- Extend to three or more images by chaining homographies

### Stage 8 — Quantitative Evaluation and Comparison
Measure pipeline performance across methods and conditions. Report standardized metrics for rigorous comparison.

---

## Scope Boundaries

### In Scope
- Classical feature detectors and descriptors (SIFT, ORB, Harris, FAST, BRIEF)
- OpenCV-based implementation in Python
- Controlled experimental conditions (rotation, scale, viewpoint, illumination)
- Quantitative metrics (keypoints, matches, inliers, time, panorama quality)
- Comparison of at least two approaches on identical data

### Out of Scope
- Deep learning feature extractors (SuperPoint, DELF, D2-Net, LoFTR, etc.)
- Dense optical flow methods for stitching
- Black-box panorama APIs (`cv2.Stitcher_create()` used without understanding)
- GPS/IMU-aided alignment
- 3D reconstruction or Structure from Motion
- Video panorama / streaming

---

## Key Academic References to Consult

> **Important**: Do not cite these without reading them. Only include references you have genuinely engaged with.

1. Lowe, D.G. (2004). "Distinctive image features from scale-invariant keypoints." *International Journal of Computer Vision*, 60(2), 91–110.
2. Rublee, E., et al. (2011). "ORB: An efficient alternative to SIFT or SURF." *ICCV 2011*.
3. Fischler, M.A. & Bolles, R.C. (1981). "Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography." *Communications of the ACM*, 24(6), 381–395.
4. Brown, M. & Lowe, D.G. (2007). "Automatic Panoramic Image Stitching using Invariant Features." *International Journal of Computer Vision*, 74(1), 59–73.
5. Harris, C. & Stephens, M. (1988). "A combined corner and edge detector." *Alvey Vision Conference*.
6. Hartley, R. & Zisserman, A. (2004). *Multiple View Geometry in Computer Vision*. 2nd ed. Cambridge University Press.
7. Szeliski, R. (2022). *Computer Vision: Algorithms and Applications*. 2nd ed. Springer. (Chapters on feature detection, image stitching.)

---

## Project Constraints Summary

| Constraint | Requirement |
|---|---|
| Minimum images | ≥ 3 overlapping images |
| Feature detectors | ≥ 2 classical methods (e.g., SIFT + ORB) |
| Outlier rejection | RANSAC required |
| Transformation model | Homography (projective) |
| Experiment conditions | Rotation, Scale, Viewpoint, Illumination |
| Programming language | Python 3.x |
| Core library | OpenCV |
| Report | Academic format with quantitative results |
| Fabrication | Absolutely prohibited |

---

*This document provides project scope and background. See `/ai/context/examination-requirements.md` for the full requirement traceability matrix.*
