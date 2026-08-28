# Skill: Feature Detection

## Overview

This skill guides the implementation of the keypoint detection stage of the pipeline. Feature detection is the first active algorithmic stage — it identifies locations in an image that are distinctive and likely to be found again in another image of the same scene.

**Requirement satisfied**: REQ-03

---

## What Makes a Good Keypoint?

A keypoint must be:
1. **Repeatable** — detectable in both images even under viewpoint, scale, or lighting changes
2. **Informative** — located at a region with enough local structure to support a unique descriptor
3. **Localized** — precisely positioned (not a broad, ambiguous region)
4. **Numerous enough** — enough keypoints to support reliable homography estimation (minimum 4 inliers needed, practically need hundreds of raw matches)

---

## Methods to Implement

### Method A: SIFT (Required)

**Full name**: Scale-Invariant Feature Transform (Lowe, 2004)

**Detection principle**: Find extrema (maxima/minima) of the Difference-of-Gaussian (DoG) function in scale space. These correspond to blob-like structures at their natural scale.

**Parameters and their meaning**:
```python
sift = cv2.SIFT_create(
    nfeatures=0,              # Max keypoints to retain (0=unlimited)
    nOctaveLayers=3,          # Layers per octave in scale space (default 3)
    contrastThreshold=0.04,   # Filter low-contrast keypoints (lower=more keypoints)
    edgeThreshold=10,         # Filter edge-like responses (higher=more keypoints)
    sigma=1.6                 # Initial Gaussian blur sigma
)
kp = sift.detect(gray, None)
```

**Expected output characteristics**:
- Keypoints have position (x,y), scale (size), and orientation (angle)
- Detects blob-like regions across multiple scales
- Fewer keypoints than ORB by default, but generally higher quality
- Good distribution across the image for textured scenes

**When SIFT fails**:
- Very smooth/textureless regions (no extrema in DoG)
- Heavily blurred images (DoG smoothing removes structure)
- Very small images (insufficient scale pyramid depth)

---

### Method B: ORB (Required)

**Full name**: Oriented FAST and Rotated BRIEF (Rublee et al., 2011)

**Detection principle**: 
1. Detect corners using FAST (circle-based pixel comparison test)
2. Score corners with Harris response to select the best N
3. Assign orientation using the intensity centroid of the local patch
4. Build image pyramid for multi-scale detection

**Parameters and their meaning**:
```python
orb = cv2.ORB_create(
    nfeatures=500,            # Maximum number of keypoints to detect
    scaleFactor=1.2,          # Scale factor between pyramid levels (1.2 is default)
    nlevels=8,                # Number of pyramid levels
    edgeThreshold=31,         # Border size where features are not detected
    firstLevel=0,             # Level of the pyramid where original image is placed
    WTA_K=2,                  # Number of points to compute BRIEF descriptor orientation
    patchSize=31,             # Patch size for computing oriented BRIEF
    fastThreshold=20          # Threshold for FAST detector
)
kp = orb.detect(gray, None)
```

**Expected output characteristics**:
- Returns exactly `nfeatures` keypoints (or fewer if insufficient)
- Keypoints tend to cluster in high-contrast regions (corners, edges)
- Faster than SIFT but potentially less uniformly distributed
- Rotation-invariant via the orientation assignment

**When ORB fails**:
- Images with very few sharp corners (smooth, blurry, or textureless)
- Large viewpoint changes that exceed the scale pyramid range
- When `nfeatures` is set too low for the scene complexity

---

### Method C: Harris Corner Detector (Optional / for comparison)

**Detection principle**: Second-moment matrix (structure tensor) eigenvalue analysis. A corner has large eigenvalues in both directions.

```python
gray_float = np.float32(gray)
harris = cv2.cornerHarris(gray_float, blockSize=2, ksize=3, k=0.04)
harris = cv2.dilate(harris, None)  # Enhance corners
# Threshold
keypoint_mask = harris > 0.01 * harris.max()
```

Harris does not produce keypoints in the OpenCV `KeyPoint` format directly — you must convert threshold locations to KeyPoint objects. It has no built-in descriptor, so pair with BRIEF or SIFT descriptor separately.

**Note**: Harris is useful to discuss in the report as a foundational concept, even if SIFT/ORB are the primary methods.

---

## Implementation Instructions

### Step 1: Create detector factory

```python
# src/features/detection.py

def create_detector(method: str, config: dict) -> cv2.Feature2D:
    """
    Factory function to create a feature detector.
    
    Args:
        method: One of 'SIFT', 'ORB', 'HARRIS'
        config: Dictionary of method-specific parameters
        
    Returns:
        OpenCV Feature2D detector object
        
    # REQ-03: Detect distinctive keypoints using an appropriate feature detector
    """
    method = method.upper()
    if method == 'SIFT':
        return cv2.SIFT_create(**config.get('sift', {}))
    elif method == 'ORB':
        return cv2.ORB_create(**config.get('orb', {}))
    else:
        raise ValueError(f"Unknown method: {method}. Choose from: SIFT, ORB")
```

### Step 2: Detect keypoints

```python
def detect_keypoints(img_gray: np.ndarray, detector: cv2.Feature2D) -> list:
    """
    Detect keypoints in a grayscale image.
    
    Returns:
        List of cv2.KeyPoint objects
    """
    import time
    start = time.perf_counter()
    keypoints = detector.detect(img_gray, None)
    elapsed = time.perf_counter() - start
    return keypoints, elapsed
```

### Step 3: Record and report

After detection, always record:
```python
metrics = {
    'num_keypoints': len(keypoints),
    'detection_time_s': elapsed,
    'keypoint_scales': [kp.size for kp in keypoints],         # scale distribution
    'keypoint_responses': [kp.response for kp in keypoints],  # strength distribution
}
```

---

## Parameter Tuning Guidelines

| Scenario | Recommendation |
|---|---|
| Too few keypoints for matching | SIFT: lower `contrastThreshold`; ORB: increase `nfeatures` |
| Too many weak keypoints | SIFT: raise `contrastThreshold`; ORB: lower `nfeatures` |
| Missing large-scale features | SIFT: increase `nOctaveLayers`; ORB: increase `scaleFactor` |
| Edge responses dominating | SIFT: lower `edgeThreshold` (more aggressive edge filtering) |

---

## Comparison Considerations

When comparing SIFT and ORB in the report, discuss:
1. **Detection scale**: SIFT detects across scales; ORB uses discrete pyramid levels
2. **Spatial distribution**: SIFT tends to be more spread across the image; ORB focuses on corners
3. **Speed**: ORB is significantly faster — record actual times
4. **Robustness**: SIFT is more robust to scale and illumination changes in general
5. **Context**: ORB is preferred in real-time applications (robotics, AR); SIFT in offline, accuracy-critical tasks

---

## Common Mistakes to Avoid

- **Do not** detect on colour images — always convert to grayscale first
- **Do not** compare keypoint counts from SIFT and ORB without accounting for `nfeatures` limits
- **Do not** skip the keypoint visualization — it reveals spatial distribution problems early
- **Do not** use the same `nfeatures` value in both methods and claim a fair comparison without noting the cap
