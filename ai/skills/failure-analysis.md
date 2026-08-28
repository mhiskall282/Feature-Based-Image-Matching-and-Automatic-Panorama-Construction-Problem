# Skill: Failure Analysis

## Overview

This skill guides the identification, documentation, and analysis of failure cases. Documenting failures honestly is a sign of scientific rigour and is explicitly expected in an MPhil/MSc examination project. A system that pretends never to fail is dishonest and academically unacceptable.

**Requirement satisfied**: Part of REQ-12 (robustness investigation), report section §19 Failure Analysis

---

## Why Document Failures?

1. **Academic integrity**: Fabricating success is dishonest. Examiners expect to see where and why algorithms fail.
2. **Understanding**: Failure cases reveal the assumptions underlying each algorithm. Understanding when SIFT fails demonstrates deeper knowledge than only showing when it succeeds.
3. **Practical value**: Real-world applications need to know the failure modes before deployment.
4. **Oral defence**: Failure cases are favourite questions in oral examinations — "What happens when your system encounters X?"

---

## Categories of Failure

### F1 — Insufficient Keypoints

**Symptom**: `len(keypoints)` is very small (< 20) for one or both images.

**Root causes**:
- Textureless scene (blank wall, sky, flat surface)
- Image is too blurry (DoG cannot find extrema; FAST cannot detect corners)
- Image is too dark or overexposed
- `nfeatures` limit in ORB is too restrictive

**Detection**:
```python
if len(keypoints) < MIN_KEYPOINTS_THRESHOLD:
    log_failure('INSUFFICIENT_KEYPOINTS', len(keypoints), image_name, method)
```

**What to report**: Image pair, method, keypoint count, visual inspection of image quality.

---

### F2 — Insufficient Matches

**Symptom**: After ratio test or cross-check, `len(good_matches) < 4`.

**Root causes**:
- Scenes have no common content (wrong image pair)
- Scene is textureless — keypoints detected but descriptors are all similar (ambiguous matching)
- Extreme condition (large rotation, scale change, illumination change) exceeds algorithm invariance range
- Ratio test threshold too strict

**Detection**:
```python
if len(good_matches) < 4:
    log_failure('INSUFFICIENT_MATCHES', len(good_matches), pair_name, method)
    return None  # Cannot proceed to RANSAC
```

---

### F3 — RANSAC Failure

**Symptom**: `cv2.findHomography()` returns `H = None`, or returns H but with very few inliers (< `min_inliers` threshold).

**Root causes**:
- Matches are dominated by outliers (> 95% outlier ratio) — RANSAC cannot find a consistent model
- Matches are in a degenerate configuration (all collinear, all near-coplanar) — H is underdetermined
- Fewer than 4 valid correspondences
- Scene has significant 3D depth variation (parallax) — homography model invalid

**Detection**:
```python
if H is None:
    log_failure('RANSAC_RETURNED_NONE', len(good_matches), pair_name, method)
elif ransac_inliers < MIN_RANSAC_INLIERS:
    log_failure('INSUFFICIENT_RANSAC_INLIERS', ransac_inliers, pair_name, method)
```

---

### F4 — Degenerate Homography

**Symptom**: H is returned but the resulting warp is visually nonsensical (image warped to a tiny line, or warped to the wrong region entirely).

**Root causes**:
- All matches came from a single small region of the image (not distributed)
- Scene contains a dominant repeated pattern causing many incorrect matches that happen to be geometrically consistent
- Scene is truly planar with near-zero in-plane structure (e.g., a blank wall)

**Detection**:
```python
det = np.linalg.det(H)
if abs(det) < 1e-6 or abs(det) > 1e6:
    log_failure('DEGENERATE_HOMOGRAPHY', det, pair_name, method)
    
# Also check corner mapping
mapped_corners = verify_homography(H, img1.shape, img2.shape)
if not mapped_corners['corners_in_bounds']:
    log_failure('OUT_OF_BOUNDS_WARP', mapped_corners['mapped_corners'], pair_name, method)
```

---

### F5 — Panorama Stitching Artefacts

**Symptom**: The panorama is constructed but has visible problems:
- **Ghosting**: The same object appears twice (double exposure) in the overlap region
- **Seam lines**: A hard edge is visible where images join
- **Misalignment**: Lines or edges don't match up across the stitch boundary
- **Colour mismatch**: Brightness or colour tone differs noticeably between panels
- **Black holes**: Large unfilled regions in the canvas

**Root causes** and mitigations:

| Artefact | Root cause | Mitigation |
|---|---|---|
| Ghosting | H is slightly wrong; two versions of the scene appear | Improve matching; use better blending |
| Seam lines | Simple overwrite compositing; brightness mismatch | Alpha blending; histogram matching |
| Misalignment | RANSAC inlier ratio too low; H inaccurate | Improve initial matching; lower RANSAC threshold |
| Colour mismatch | Images shot under different exposure | Histogram matching before stitching |
| Black holes | Canvas offset wrong; H maps to wrong region | Debug canvas computation |

---

## Failure Logging System

Implement a consistent failure logger:

```python
# src/evaluation/metrics.py

import json
from datetime import datetime
from pathlib import Path

FAILURE_CODES = {
    'INSUFFICIENT_KEYPOINTS': 'Fewer keypoints than minimum threshold',
    'INSUFFICIENT_MATCHES': 'Fewer than 4 good matches — cannot run RANSAC',
    'RANSAC_RETURNED_NONE': 'cv2.findHomography() returned None',
    'INSUFFICIENT_RANSAC_INLIERS': 'RANSAC inlier count below minimum threshold',
    'DEGENERATE_HOMOGRAPHY': 'H matrix has degenerate determinant',
    'OUT_OF_BOUNDS_WARP': 'Homography maps corners outside reasonable bounds',
    'STITCHING_ARTEFACT': 'Visible stitching artefact in panorama',
}

def log_failure(
    code: str,
    value,
    context: str,
    method: str,
    output_dir: Path
):
    """
    Log a pipeline failure to a JSON file.
    """
    failure = {
        'timestamp': datetime.utcnow().isoformat(),
        'code': code,
        'description': FAILURE_CODES.get(code, 'Unknown failure'),
        'value': str(value),
        'context': context,
        'method': method,
    }
    
    failure_log_path = output_dir / 'failures.json'
    
    # Append to existing failures
    failures = []
    if failure_log_path.exists():
        with open(failure_log_path) as f:
            failures = json.load(f)
    
    failures.append(failure)
    
    with open(failure_log_path, 'w') as f:
        json.dump(failures, f, indent=2)
    
    print(f"[FAILURE] {code}: {value} ({context}, {method})")
```

---

## Failure Visualization

For every failure case, generate a figure showing what went wrong:

```python
def visualize_failure(img1, img2, failure_code, method, details, outpath):
    """
    Create a failure case visualization for the report.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Image 1')
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Image 2')
    axes[1].axis('off')
    
    plt.suptitle(
        f'FAILURE CASE — {method}\n'
        f'Code: {failure_code}\n'
        f'Details: {details}',
        fontsize=11, color='red', fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
```

**Output**: `outputs/{exp}/{method}/failure_{failure_code}_{pair}.png`

---

## Report Section: Failure Analysis (§19)

The failure analysis section of the report must:

1. **List every failure type encountered** with the experiment and condition that triggered it
2. **Provide the failure visualization** (figure showing the problematic images and/or partial output)
3. **Explain the root cause** — connect the failure to the algorithm's theoretical assumptions
4. **Compare methods**: Does SIFT fail in situations where ORB succeeds, or vice versa?
5. **Quantify failures**: How many trials out of total resulted in failure per method?

Example structure:
```
§19 Failure Analysis

§19.1 Failure under Extreme Scale Change (ORB)
  - Condition: scale factor 0.25× (image reduced to 25% then restored)
  - Method: ORB
  - Failure code: INSUFFICIENT_RANSAC_INLIERS (N=2)
  - Root cause: ORB's image pyramid (8 levels, scaleFactor=1.2) covers only
    approximately 4× scale range. A 4× scale change exceeds this range.
  - SIFT performance: X inliers (not failed) — SIFT's continuous scale space
    handles this range.
  - [Figure: failure visualization]

§19.2 Failure under Textureless Scene (Both methods)
  ...
```

---

## Common Failure Scenarios to Test Deliberately

Include these in your experiments:

1. **Extreme rotation (>90°)** — test failure boundary of ORB
2. **Very low overlap (<15%)** — insufficient matches
3. **Textureless scene** — blank wall or uniform sky (include as a negative result)
4. **Large depth variation with camera translation** — homography model violation
5. **Very dark image pair** — illumination experiment failure case

---

## Important: Failure ≠ Bad Project

A project that encounters, documents, and explains failures demonstrates **deeper understanding** than one that only shows successful cases. An examiner will ask: "What happens when your system fails?" — be ready to answer from documented evidence.
