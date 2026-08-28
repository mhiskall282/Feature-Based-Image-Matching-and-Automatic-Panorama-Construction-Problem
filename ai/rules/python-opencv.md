# Rules: Python and OpenCV Coding Standards

## Status
These rules define the Python and OpenCV coding standards for this project. They are **mandatory** for all source files in `src/` and `experiments/`.

---

## RULE-PY01: Python Version

Use **Python 3.8 or later**. All code must be compatible with Python 3.8+.

```python
# At top of every script/module entry point:
import sys
assert sys.version_info >= (3, 8), "Python 3.8+ required"
```

---

## RULE-PY02: Required Libraries

The project must use these libraries and no others for core functionality:

| Library | Version | Purpose |
|---|---|---|
| `opencv-python` | >= 4.5.0 | All computer vision operations |
| `numpy` | >= 1.20.0 | Numerical arrays |
| `matplotlib` | >= 3.3.0 | All visualizations |
| `pandas` | >= 1.2.0 | Results tables, CSV I/O |
| `scikit-image` | >= 0.18.0 | Optional: SSIM metric |
| `scipy` | >= 1.6.0 | Optional: statistical functions |

**Do not** add dependencies for tasks already covered by the above (e.g., no `Pillow` for image loading when `cv2.imread` works, no `seaborn` for charts when `matplotlib` works).

---

## RULE-PY03: Image Channel Convention

OpenCV loads images in **BGR** (not RGB). Apply conversions consistently:

```python
# Loading
img = cv2.imread(path)           # BGR
# For matplotlib display
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# For grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# NEVER pass a BGR image to matplotlib directly — it will show incorrect colours
```

**Rule**: Always convert BGR → RGB before any matplotlib display call.

---

## RULE-PY04: Image Loading Validation

Every image load must be validated:

```python
def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(
            f"Failed to load image: {path}\n"
            f"Check that the file exists and is a valid image format."
        )
    return img
```

Do not silently proceed with `None` images — they will cause cryptic errors downstream.

---

## RULE-PY05: Floating Point Precision for Geometry

All homography and point coordinate computations must use `float32` or `float64`:

```python
# For point arrays passed to findHomography
src_pts = np.float32([...]).reshape(-1, 1, 2)
dst_pts = np.float32([...]).reshape(-1, 1, 2)

# Homography matrices use float64 by default from findHomography — keep as-is
```

**Do not** use integer arrays for geometric computations — they produce silent truncation errors.

---

## RULE-PY06: Descriptor Type and Distance Metric Enforcement

Enforce the correct pairing at the matcher creation level:

```python
DESCRIPTOR_NORMS = {
    'SIFT': cv2.NORM_L2,
    'ORB':  cv2.NORM_HAMMING,
}

def get_norm_for_method(method: str) -> int:
    norm = DESCRIPTOR_NORMS.get(method.upper())
    if norm is None:
        raise ValueError(f"Unknown method '{method}'. Add to DESCRIPTOR_NORMS.")
    return norm
```

**Do not** hardcode `cv2.NORM_L2` or `cv2.NORM_HAMMING` in multiple places — use this function.

---

## RULE-PY07: Timing All Pipeline Stages

Every pipeline stage that takes measurable time must be timed:

```python
import time

start = time.perf_counter()
# ... operation ...
elapsed = time.perf_counter() - start
```

Use `time.perf_counter()` (high-resolution timer). Do not use `time.time()` for benchmarking (it has lower resolution on Windows).

Store timings in the metrics dict for every trial.

---

## RULE-PY08: Path Handling

Use `pathlib.Path` for all file path operations:

```python
from pathlib import Path

output_dir = Path('outputs') / 'baseline' / 'SIFT'
output_dir.mkdir(parents=True, exist_ok=True)

img_path = Path('data') / 'baseline' / 'img_01.jpg'
```

**Do not** use hardcoded string paths like `'outputs\\baseline\\SIFT'` — use Path and let it handle OS-specific separators.

---

## RULE-PY09: Configuration Management

All tunable parameters must come from a configuration dictionary or file — not hardcoded inline:

```python
# experiments/config.py

DEFAULT_CONFIG = {
    'sift': {
        'nfeatures': 0,
        'nOctaveLayers': 3,
        'contrastThreshold': 0.04,
        'edgeThreshold': 10,
        'sigma': 1.6,
    },
    'orb': {
        'nfeatures': 500,
        'scaleFactor': 1.2,
        'nlevels': 8,
        'edgeThreshold': 31,
        'patchSize': 31,
        'fastThreshold': 20,
    },
    'matching': {
        'ratio_threshold': 0.75,
    },
    'ransac': {
        'reprojection_threshold': 5.0,
        'confidence': 0.995,
        'max_iters': 2000,
        'min_inliers': 10,
    },
    'random_seed': 42,
}
```

Experiment-specific configs override defaults using `{**DEFAULT_CONFIG, 'sift': {...}}`.

---

## RULE-PY10: Type Hints for Public Functions

All public functions in `src/` must have type hints:

```python
def detect_keypoints(
    img_gray: np.ndarray,
    detector: cv2.Feature2D
) -> tuple[list, float]:
    ...
```

Private helper functions (prefixed with `_`) may omit type hints.

---

## RULE-PY11: Docstrings for All Public Functions

All public functions must have a docstring:

```python
def apply_ransac(kp1, kp2, matches, config):
    """
    Apply RANSAC to estimate a homography and classify inliers.
    
    Args:
        kp1 (list): Keypoints from image 1
        kp2 (list): Keypoints from image 2
        matches (list): List of cv2.DMatch objects
        config (dict): Configuration dict with 'ransac' keys
    
    Returns:
        tuple: (H, mask, inlier_matches, outlier_matches, metrics_dict)
               H is None if estimation failed.
    
    # REQ-07: Apply RANSAC to eliminate incorrect correspondences
    """
```

---

## RULE-PY12: Error Handling for All I/O Operations

All file I/O must be wrapped in error handling:

```python
try:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
except Exception as e:
    print(f"[ERROR] Failed to load {path}: {e}")
    raise
```

---

## RULE-PY13: No Global State

Do not use module-level global variables that change during execution. All state must be passed as function arguments or encapsulated in a configuration object.

---

## RULE-PY14: Print Statements for Pipeline Progress

All experiment scripts must print progress to stdout:

```python
print(f"[SIFT] Image 1: {len(kp1)} keypoints detected in {t_det1:.3f}s")
print(f"[SIFT] Matches after ratio test: {len(good_matches)}")
print(f"[SIFT] RANSAC: {ransac_inliers} inliers / {len(good_matches)} ({inlier_ratio:.1%})")
print(f"[SIFT] Homography: {'SUCCESS' if H is not None else 'FAILED'}")
```

This allows quick visual inspection of results without opening CSV files every run.

---

## RULE-PY15: Do Not Suppress Errors Silently

```python
# FORBIDDEN
try:
    H, mask = cv2.findHomography(...)
except:
    pass  # Silent failure — never do this

# REQUIRED
try:
    H, mask = cv2.findHomography(...)
    if H is None:
        log_failure('RANSAC_RETURNED_NONE', ...)
except cv2.error as e:
    log_failure('OPENCV_ERROR', str(e), ...)
    raise
```
