# Skill: RANSAC and Homography Estimation

## Overview

This skill guides the implementation of RANSAC-based outlier rejection and homography estimation. This is the most critical algorithmic step in the pipeline — it separates correct correspondences (inliers) from false ones (outliers) and estimates the geometric transformation between images.

**Requirements satisfied**: REQ-07, REQ-08, REQ-11

---

## Why RANSAC is Necessary

Raw feature matches — even after ratio test or cross-check — contain many false correspondences (outliers). Estimating a homography directly from all matches using least squares would be severely corrupted by these outliers.

**RANSAC (Random Sample Consensus)** is a robust estimation framework that:
1. Assumes that the dataset contains both inliers (matches consistent with some H) and outliers (random noise)
2. Iteratively hypothesizes a model from a minimal set of points
3. Selects the hypothesis that is consistent with the most data points
4. Returns the best model and identifies which points are inliers

Without RANSAC, homography estimation from noisy matches will fail or produce an incorrect H.

---

## RANSAC Algorithm Detail

### Minimum sample size for homography
A homography has 8 degrees of freedom. Each point correspondence provides 2 equations. Therefore, the minimum number of correspondences needed to compute a unique homography is **4 point pairs**.

### Number of iterations
The required number of iterations N for a desired confidence P, given expected outlier ratio ε and sample size s=4:

```
N = log(1 - P) / log(1 - (1 - ε)^s)
```

| Outlier ratio (ε) | Confidence (P) | Required iterations (N) |
|---|---|---|
| 10% | 99% | 5 |
| 30% | 99% | 34 |
| 50% | 99% | 145 |
| 70% | 99% | 1177 |
| 80% | 99% | 4787 |

OpenCV handles iteration count automatically based on the reprojection threshold.

### Reprojection threshold
A match (p1, p2) is an inlier if:
```
||p2 - H * p1|| < threshold  (in pixels)
```

Typical values: 3–5 pixels. Smaller = stricter = fewer inliers but better quality. Larger = more lenient = more inliers but potentially noisier.

---

## Implementation

### Core RANSAC + Homography call

```python
# src/homography/ransac.py

import cv2
import numpy as np
import time

def apply_ransac(
    kp1: list,
    kp2: list, 
    matches: list,
    config: dict
) -> tuple:
    """
    Apply RANSAC to find inlier matches and estimate homography.
    
    Args:
        kp1, kp2: Keypoints from image 1 and image 2
        matches: List of cv2.DMatch objects (good matches from matching stage)
        config: dict with keys:
            - ransac_threshold: reprojection threshold in pixels (default 5.0)
            - min_inliers: minimum inliers to accept result (default 10)
    
    Returns:
        (H, mask, inlier_matches, outlier_matches, metrics)
        H = None if estimation failed
        
    # REQ-07: Apply RANSAC to eliminate incorrect correspondences
    # REQ-08: Estimate the homography matrix
    """
    if len(matches) < 4:
        return None, None, [], matches, {'error': 'insufficient_matches'}
    
    # Extract point coordinates
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    threshold = config.get('ransac_threshold', 5.0)
    
    start = time.perf_counter()
    H, mask = cv2.findHomography(
        src_pts, dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=threshold,
        confidence=0.995,        # Desired probability that result is correct
        maxIters=2000            # Maximum RANSAC iterations
    )
    elapsed = time.perf_counter() - start
    
    if H is None or mask is None:
        return None, None, [], matches, {
            'error': 'homography_failed',
            'ransac_time_s': elapsed
        }
    
    # Separate inliers and outliers
    mask_flat = mask.ravel()
    inlier_matches  = [m for m, flag in zip(matches, mask_flat) if flag == 1]
    outlier_matches = [m for m, flag in zip(matches, mask_flat) if flag == 0]
    
    min_inliers = config.get('min_inliers', 10)
    success = len(inlier_matches) >= min_inliers
    
    metrics = {
        'num_input_matches': len(matches),
        'num_ransac_inliers': len(inlier_matches),
        'num_ransac_outliers': len(outlier_matches),
        'inlier_ratio': len(inlier_matches) / len(matches) if matches else 0.0,
        'ransac_time_s': elapsed,
        'homography_success': success,
        'ransac_threshold': threshold,
    }
    
    if not success:
        return None, mask, inlier_matches, outlier_matches, metrics
    
    return H, mask, inlier_matches, outlier_matches, metrics
```

### Homography verification and analysis

```python
# src/homography/estimate.py

def verify_homography(
    H: np.ndarray, 
    img1_shape: tuple,
    img2_shape: tuple
) -> dict:
    """
    Verify H by mapping corners of image 1 and checking output.
    
    # REQ-08: Estimate the homography matrix
    """
    h1, w1 = img1_shape[:2]
    
    # Four corners of image 1
    corners = np.float32([[0,0],[w1,0],[w1,h1],[0,h1]]).reshape(-1, 1, 2)
    
    # Map through H
    mapped = cv2.perspectiveTransform(corners, H)
    
    # Check: mapped corners should be within reasonable bounds of image 2
    h2, w2 = img2_shape[:2]
    margin = max(h2, w2) * 0.5  # Allow 50% outside frame
    
    mapped_flat = mapped.reshape(-1, 2)
    all_in_bounds = all(
        -margin <= x <= w2 + margin and -margin <= y <= h2 + margin
        for x, y in mapped_flat
    )
    
    return {
        'mapped_corners': mapped_flat.tolist(),
        'corners_in_bounds': all_in_bounds,
        'determinant': float(np.linalg.det(H)),
        'condition_number': float(np.linalg.cond(H)),
    }


def save_homography(H: np.ndarray, path: str, metadata: dict = None):
    """Save homography matrix to text file."""
    with open(path, 'w') as f:
        f.write("# Homography Matrix H (3x3)\n")
        if metadata:
            for k, v in metadata.items():
                f.write(f"# {k}: {v}\n")
        f.write("#\n")
        np.savetxt(f, H, fmt='%.8f')
```

---

## Computing Reprojection Error

After RANSAC, compute the mean reprojection error over all inliers:

```python
def compute_reprojection_error(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    H: np.ndarray,
    mask: np.ndarray
) -> float:
    """
    Compute mean reprojection error for inlier matches.
    
    Lower is better. Should be less than the RANSAC threshold.
    """
    mask_flat = mask.ravel().astype(bool)
    inlier_src = src_pts[mask_flat]
    inlier_dst = dst_pts[mask_flat]
    
    # Project src points through H
    projected = cv2.perspectiveTransform(inlier_src, H)
    
    # Compute per-point distances
    errors = np.linalg.norm(inlier_dst - projected, axis=2).ravel()
    
    return float(np.mean(errors)) if len(errors) > 0 else float('inf')
```

---

## Before/After RANSAC Visualisation (REQ-11)

```python
# src/visualization/draw.py

def visualize_before_after_ransac(
    img1, kp1, img2, kp2, 
    all_matches, inlier_matches, outlier_matches,
    outpath
):
    """
    Side-by-side comparison: all matches vs RANSAC inliers only.
    
    # REQ-11: Compare feature matching before and after RANSAC
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    
    fig = plt.figure(figsize=(20, 8))
    gs = gridspec.GridSpec(1, 2, figure=fig)
    
    # Before RANSAC — show all matches
    ax1 = fig.add_subplot(gs[0])
    img_all = cv2.drawMatches(
        img1, kp1, img2, kp2, all_matches[:200], None,
        matchColor=(255, 0, 0),  # Red for all matches
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    ax1.imshow(cv2.cvtColor(img_all, cv2.COLOR_BGR2RGB))
    ax1.set_title(f'Before RANSAC: {len(all_matches)} matches', fontsize=13)
    ax1.axis('off')
    
    # After RANSAC — show inliers only
    ax2 = fig.add_subplot(gs[1])
    img_inliers = cv2.drawMatches(
        img1, kp1, img2, kp2, inlier_matches, None,
        matchColor=(0, 255, 0),  # Green for inliers
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    ax2.imshow(cv2.cvtColor(img_inliers, cv2.COLOR_BGR2RGB))
    ratio = len(inlier_matches) / len(all_matches) if all_matches else 0
    ax2.set_title(f'After RANSAC: {len(inlier_matches)} inliers ({ratio:.1%})', fontsize=13)
    ax2.axis('off')
    
    plt.suptitle('Feature Matching: Before vs After RANSAC', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
```

---

## Diagnosing RANSAC Failures

| Symptom | Likely Cause | Remedy |
|---|---|---|
| H is None | Fewer than 4 good matches | Check matching stage; lower ratio threshold |
| Very low inlier ratio (<10%) | Poor initial matching | Check descriptor compatibility; tune parameters |
| H is non-None but panorama is wrong | H estimated from wrong geometric model | Verify scene is planar or camera is rotating; check images |
| Inlier ratio OK but H maps corners way out of frame | Degenerate configuration (collinear points) | Acquire better-distributed matches across the scene |
| RANSAC succeeds but reprojection error is high | Threshold too large; over-permissive | Lower `ransac_threshold` |

---

## Parameter Documentation Requirements

For every RANSAC call, save to the experiment config JSON:
```json
{
  "ransac_reprojection_threshold": 5.0,
  "ransac_confidence": 0.995,
  "ransac_max_iterations": 2000,
  "min_acceptable_inliers": 10
}
```

---

## Common Mistakes to Avoid

- **Do not** skip RANSAC and claim the result is RANSAC-filtered — this is a fundamental error
- **Do not** call RANSAC with fewer than 4 point correspondences — it will fail or crash
- **Do not** ignore the `mask` returned by `findHomography` — it is the inlier/outlier classification
- **Do not** use a very large reprojection threshold (e.g., 50 pixels) to inflate inlier counts artificially
- **Do not** report the homography as successful when `H is None`
- **Do not** forget to check for degenerate H (e.g., all zeros, rank < 3)
