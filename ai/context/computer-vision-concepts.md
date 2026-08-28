# Core Computer Vision Concepts — CSCD608 Reference

## Purpose

This document describes the computer vision concepts that are central to this project. It serves as an implementation-oriented reference — not a textbook — for the agent implementing the pipeline. Each concept is described with enough depth to implement it correctly and justify algorithmic choices in a report or oral defence.

---

## 1. Image Representation and Colour Spaces

### Grayscale Conversion
Feature detectors (SIFT, ORB, Harris) operate on single-channel intensity images.

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

OpenCV stores images in BGR order (not RGB). Be consistent throughout the codebase.

### Working with Colour
- Detect keypoints and compute descriptors on **grayscale**
- Warp and stitch in **colour** (BGR) for the final panorama
- Keep a colour copy of every input image alongside the grayscale version

### Histogram Equalization
Used in illumination experiments to normalize brightness:
```python
equalized = cv2.equalizeHist(gray_img)
```
CLAHE (Contrast Limited Adaptive Histogram Equalization) is a more robust alternative:
```python
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
equalized = clahe.apply(gray_img)
```

---

## 2. Scale Space and Image Pyramids

### Why Scale Space?
Natural images contain structures at many scales. A small textured region and a large blob may both be meaningful. Feature detectors must find points that are distinctive at their natural scale.

### Gaussian Scale Space
The Gaussian scale space L(x, y, σ) is created by convolving the image with Gaussian kernels of increasing σ:
```
L(x, y, σ) = G(x, y, σ) * I(x, y)
```
where G is a Gaussian kernel and σ controls the scale of blurring.

### Difference of Gaussian (DoG)
SIFT uses the DoG approximation to the Laplacian of Gaussian (LoG):
```
D(x, y, σ) = L(x, y, kσ) − L(x, y, σ)
```
Extrema (maxima and minima) in the DoG scale space are keypoint candidates.

### Image Pyramid
A sequence of images at progressively lower resolution (each level half the previous). Used in ORB, FAST, and many tracking algorithms.

---

## 3. Harris Corner Detector

### Principle
A corner is a point where intensity changes significantly in multiple directions. The Harris detector computes the structure tensor M (second-moment matrix):

```
M = Σ w(x,y) [ Ix²    Ix·Iy ]
               [ Ix·Iy  Iy²  ]
```

where Ix, Iy are image gradients and w is a weighting window.

### Corner Response Function
```
R = det(M) − k·trace(M)²
```
- R >> 0: corner (keypoint)
- R << 0: edge
- |R| small: flat region

Typical k = 0.04–0.06.

### Limitations
- Not scale-invariant (detects corners at a single scale)
- Not descriptor-producing (must pair with a separate descriptor)
- Reference: Harris & Stephens (1988)

---

## 4. SIFT — Scale-Invariant Feature Transform

### Reference
Lowe, D.G. (2004). "Distinctive image features from scale-invariant keypoints." *IJCV*.

### Detection
1. Build Gaussian scale pyramid with multiple octaves
2. Compute DoG between adjacent scales in each octave
3. Find extrema (local min/max) in 3D (x, y, scale) DoG space
4. Filter low-contrast extrema and edge responses (ratio of Hessian eigenvalues)
5. Refine keypoint location to sub-pixel accuracy

### Description
1. For each keypoint, determine dominant orientation from gradient histogram in local region
2. Divide 16×16 region into 4×4 cells
3. Compute 8-bin gradient orientation histogram for each cell
4. Concatenate: 4×4×8 = 128-dimensional descriptor
5. Normalize and threshold to reduce illumination sensitivity

### Properties

| Property | SIFT |
|---|---|
| Rotation invariant | Yes (orientation assignment) |
| Scale invariant | Yes (scale-space detection) |
| Illumination robust | Partially (normalization) |
| Descriptor type | Float32, 128-dim |
| Distance metric | L2 (Euclidean) |
| Matching strategy | Ratio test (Lowe's 0.7) |
| Speed | Slow (computationally expensive) |
| Patent status | Patent expired; free since ~2020 |

### OpenCV Usage
```python
sift = cv2.SIFT_create(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
kp, des = sift.detectAndCompute(gray, None)
```

---

## 5. ORB — Oriented FAST and Rotated BRIEF

### Reference
Rublee, E., et al. (2011). "ORB: An efficient alternative to SIFT or SURF." *ICCV*.

### Detection (FAST + Orientation)
1. Use FAST (Features from Accelerated Segment Test) to detect corners
2. FAST tests 16 pixels on a circle around a candidate point
3. Apply Harris score to rank keypoints
4. Compute orientation using intensity centroid method (gives rotation invariance)

### Description (rBRIEF)
1. BRIEF (Binary Robust Independent Elementary Features) compares pairs of pixel intensities
2. ORB uses "steered" BRIEF — rotated according to keypoint orientation
3. Result: 256-bit binary string

### Properties

| Property | ORB |
|---|---|
| Rotation invariant | Yes (via orientation) |
| Scale invariant | Partial (image pyramid) |
| Illumination robust | Less than SIFT |
| Descriptor type | Binary, 256-bit |
| Distance metric | Hamming distance |
| Matching strategy | BFMatcher(NORM_HAMMING) + cross-check |
| Speed | Very fast |
| Patent status | Free (Apache 2.0) |

### OpenCV Usage
```python
orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8, edgeThreshold=31, patchSize=31)
kp, des = orb.detectAndCompute(gray, None)
```

---

## 6. Feature Matching

### Brute-Force Matcher (BFMatcher)
Compares each descriptor in set A against all descriptors in set B.
- Guaranteed to find the true nearest neighbour
- O(n²) complexity — slow for large descriptor sets

```python
# For SIFT (float descriptors, L2 norm)
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
matches = bf.knnMatch(des1, des2, k=2)

# For ORB (binary descriptors, Hamming norm)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
```

### FLANN Matcher
Fast approximate nearest neighbour search. Better for large datasets.

```python
# For SIFT
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)
```

### Lowe's Ratio Test (for float descriptors)
Rejects ambiguous matches where the best match is not clearly better than the second-best:
```python
good = [m for m, n in matches if m.distance < 0.75 * n.distance]
```

### Cross-Check (for binary descriptors)
Accept match (A→B) only if B also matches back to A:
```python
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
```

---

## 7. RANSAC — Random Sample Consensus

### Reference
Fischler & Bolles (1981). "Random Sample Consensus." *CACM*.

### Problem
Given a set of point correspondences that contains outliers (wrong matches), estimate a geometric model (homography H) that fits only the inliers.

### Algorithm
```
Input: Set of N point correspondences (some are outliers)
Output: Model H, inlier set S

for i = 1 to max_iterations:
    sample = randomly select minimum sample size (s=4 for homography)
    H_candidate = compute_homography(sample)
    inliers = {(p1, p2) : reprojection_error(p1, H_candidate, p2) < threshold}
    if |inliers| > |best_inliers|:
        best_inliers = inliers
        H_best = H_candidate

H_refined = compute_homography(best_inliers)  # refit on all inliers
return H_refined, best_inliers
```

### Key Parameters
- **Minimum sample size (s)**: 4 point correspondences for homography (minimum to compute unique H)
- **Reprojection threshold**: Typical 3–5 pixels. Points within this distance of the predicted location are inliers.
- **Max iterations (N)**: Related to expected outlier ratio p and desired confidence:
  ```
  N = log(1 - confidence) / log(1 - (1 - outlier_ratio)^s)
  ```
  For 60% outliers, s=4, 99% confidence: N ≈ 572 iterations

### OpenCV Usage
```python
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=5.0)
```
`mask` is an array where 1 = inlier, 0 = outlier.

---

## 8. Homography Matrix

### Definition
A 3×3 matrix H that maps points from one image plane to another under a projective transformation:

```
[x']     [h00 h01 h02] [x]
[y']  =  [h10 h11 h12] [y]
[w']     [h20 h21 h22] [1]

x_final = x'/w',  y_final = y'/w'
```

### Valid Under
- Pure rotation of the camera (no translation)
- Planar scene (all 3D points lie on a plane)
- Panorama construction (camera rotation only) is the common use case

### Degrees of Freedom
H has 8 degrees of freedom (9 elements minus 1 for scale). Minimum 4 point correspondences to compute.

### Decomposition
H can encode combinations of:
- Rotation
- Translation (within the plane)
- Scale
- Shear
- Perspective distortion

### Verification
After computing H, verify by mapping corners of image 1 to image 2 and checking they land in expected positions:
```python
corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
mapped = cv2.perspectiveTransform(corners, H)
```

---

## 9. Image Warping and Blending

### Perspective Transform
```python
warped = cv2.warpPerspective(img, H, (canvas_width, canvas_height))
```

### Canvas Size Calculation
The output canvas must contain both images:
1. Map the four corners of the source image through H
2. Find the bounding box of the mapped corners + the reference image size
3. Use the bounding box dimensions as the canvas size
4. Apply a translation offset if any corner maps to a negative coordinate

### Blending Strategies

**Simple (overwrite)**: Reference image pixels overwrite warped pixels in overlap. Creates hard seams.

**Alpha blending**: Weight pixels by distance from seam. Reduces visible seam.

**Multi-band blending**: Different frequency components blended at different scales. Best quality but complex. (Optional for this project; note if not implemented.)

---

## 10. Image Quality Metrics

### SSIM — Structural Similarity Index
Measures perceptual similarity between two image patches:
- Values: 0 (no similarity) to 1 (identical)
- Applicable to overlap regions where both images are visible
- Implemented in `skimage.metrics.structural_similarity`

### PSNR — Peak Signal-to-Noise Ratio
```
PSNR = 10 * log10(MAX² / MSE)
```
Higher is better. Meaningful only when a ground-truth reference exists.

### Inlier Ratio
```
inlier_ratio = num_inliers / num_raw_matches
```
Reflects the quality of initial matching. Higher ratio = fewer outliers = better feature matching.

### Reprojection Error
```
reprojection_error = mean(||p2 - H·p1||) over inliers
```
Should be < reprojection threshold. Lower is better.

---

*This document covers the key concepts required for implementation. Consult textbooks (Hartley & Zisserman, Szeliski) for full mathematical derivations.*
