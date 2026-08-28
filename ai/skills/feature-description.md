# Skill: Feature Description

## Overview

This skill guides the implementation of the descriptor computation stage. A descriptor is a compact numerical representation of the image neighbourhood around a keypoint. The descriptor must be distinctive enough to match the same physical point across different images, and robust enough to tolerate the expected changes (viewpoint, scale, illumination).

**Requirement satisfied**: REQ-04

---

## Descriptor Fundamentals

A feature descriptor answers: **"What does the region around this keypoint look like — in a way that another image of the same point would agree on?"**

A good descriptor is:
- **Discriminative** — different physical points produce different descriptors
- **Invariant** — the same physical point produces the same (or similar) descriptor under viewpoint/scale/lighting changes
- **Compact** — small enough for fast comparison and storage
- **Efficient** — fast to compute

---

## SIFT Descriptor

### Construction
1. Take a 16×16 pixel region centred on the keypoint, normalised to the keypoint's dominant orientation
2. Divide into 4×4 grid of cells
3. In each cell, compute an 8-bin gradient orientation histogram (weighted by magnitude)
4. Concatenate all histograms: 4×4×8 = **128-dimensional** float vector
5. Normalize the 128-vector to unit length (L2 norm)
6. Clamp values above 0.2 (reduces illumination sensitivity)
7. Re-normalize

### Properties
| Property | Value |
|---|---|
| Dimensions | 128 |
| Type | float32 |
| Norm | L2 (Euclidean) |
| Rotation invariant | Yes (orientation normalized) |
| Scale invariant | Yes (computed at detected scale) |
| Illumination robust | Partial (clamping + normalization) |
| Affine invariant | Partially |

### Distance Metric
Use **L2 (Euclidean) distance** for SIFT:
```python
distance = np.linalg.norm(desc1 - desc2)  # manual
# or via BFMatcher(cv2.NORM_L2)
```

### OpenCV Usage
```python
sift = cv2.SIFT_create()
# Compute descriptors for pre-detected keypoints:
kp, desc = sift.compute(gray, keypoints)
# Or detect + compute in one call:
kp, desc = sift.detectAndCompute(gray, None)
# desc.shape: (N, 128), dtype=float32
```

---

## ORB Descriptor (rBRIEF)

### Construction
1. Take a 31×31 pixel patch centred on the keypoint
2. Rotate the patch according to the keypoint's assigned orientation
3. Apply a set of pre-learned binary tests: compare pixel intensities at 256 pairs of locations within the patch
4. Each test produces 1 bit (1 if I(p) < I(q), else 0)
5. Concatenate 256 bits → **256-bit binary string**

### Properties
| Property | Value |
|---|---|
| Dimensions | 256 bits (32 bytes) |
| Type | uint8 (packed bits) |
| Norm | Hamming distance |
| Rotation invariant | Yes (patch rotated before testing) |
| Scale invariant | Partial (pyramid-based) |
| Illumination robust | Less than SIFT |
| Speed | Very fast |

### Distance Metric
Use **Hamming distance** for ORB (number of differing bits):
```python
# via BFMatcher(cv2.NORM_HAMMING)
```
**Never use L2 distance with binary descriptors** — it is mathematically meaningless.

### OpenCV Usage
```python
orb = cv2.ORB_create()
kp, desc = orb.detectAndCompute(gray, None)
# desc.shape: (N, 32), dtype=uint8 (packed 256-bit binary)
```

---

## Descriptor-Matcher Compatibility Table

| Detector | Descriptor | Distance Metric | Matcher |
|---|---|---|---|
| SIFT | SIFT (128-dim float) | L2 | BFMatcher(NORM_L2) or FLANN |
| ORB | rBRIEF (256-bit binary) | Hamming | BFMatcher(NORM_HAMMING) |
| Harris | SIFT or ORB (computed separately) | L2 or Hamming | Depends on descriptor |

**Critical rule**: Always match the distance metric to the descriptor type. Mismatching (e.g., Hamming on float32) produces meaningless results and is a fundamental error.

---

## Implementation Instructions

### Combined Detect + Describe

In most cases, use the unified `detectAndCompute` call:

```python
# src/features/description.py

def compute_descriptors(
    img_gray: np.ndarray, 
    detector: cv2.Feature2D,
    mask: np.ndarray = None
) -> tuple:
    """
    Detect keypoints and compute descriptors jointly.
    
    Args:
        img_gray: Grayscale image
        detector: A cv2.Feature2D detector (SIFT, ORB, etc.)
        mask: Optional binary mask (detect only in mask==255 regions)
    
    Returns:
        (keypoints, descriptors, elapsed_time)
        
    # REQ-04: Compute descriptors
    """
    import time
    start = time.perf_counter()
    keypoints, descriptors = detector.detectAndCompute(img_gray, mask)
    elapsed = time.perf_counter() - start
    
    if descriptors is None:
        raise ValueError("No descriptors computed — image may have no detectable features")
    
    return keypoints, descriptors, elapsed
```

### Saving Descriptors

For reproducibility, save descriptor arrays:
```python
def save_descriptors(descriptors: np.ndarray, path: str):
    np.save(path, descriptors)  # saves as .npy file

def load_descriptors(path: str) -> np.ndarray:
    return np.load(path)
```

---

## Descriptor Analysis for the Report

When writing the report section on feature description, include:

1. **Descriptor type**: Float32 or binary — why does it matter for distance metric choice?
2. **Dimensionality**: 128 (SIFT) vs 256-bit/32-byte (ORB) — what are the storage and speed implications?
3. **Invariance properties**: How does SIFT's gradient orientation histogram achieve rotation invariance? How does ORB's patch rotation achieve similar?
4. **Illumination robustness**: Explain how SIFT's L2 normalization and clamping reduce sensitivity to illumination changes. Explain why ORB is more sensitive.
5. **Descriptor visualisation** (optional but impressive): Visualise a small set of descriptor vectors as a heatmap for a few keypoints.

---

## Common Mistakes to Avoid

- **Do not** call `detect()` and `compute()` separately when the same detector is used — use `detectAndCompute()` to ensure consistency
- **Do not** use float descriptors with Hamming distance or binary descriptors with L2 distance
- **Do not** pass colour images to `detectAndCompute()` without converting to grayscale first
- **Do not** assume all keypoints will receive a valid descriptor — some may be filtered; check `len(keypoints) == len(descriptors)`
- **Do not** claim SIFT "always produces better descriptors" — document the actual experimental results

---

## Edge Cases

| Situation | What Happens | How to Handle |
|---|---|---|
| Image is blank/uniform | `descriptors = None` | Raise ValueError with informative message |
| Image is very small | Too few keypoints for reliable homography | Warn user; resize or use different image |
| `nfeatures` cap reached | Only top N keypoints retained | Document this in experiment config |
| Keypoints near border | ORB's `edgeThreshold` filters them out | Reduce `edgeThreshold` if needed |
