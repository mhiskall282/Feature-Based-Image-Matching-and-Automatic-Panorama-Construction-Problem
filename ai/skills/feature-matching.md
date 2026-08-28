# Skill: Feature Matching

## Overview

This skill guides the implementation of the descriptor matching stage. Matching finds pairs of descriptors — one from each image — that describe the same physical point. This is a critical stage: poor matching produces too many false correspondences (outliers), making RANSAC harder or causing homography failure.

**Requirements satisfied**: REQ-05, REQ-06

---

## Matching Strategies

### Strategy 1: Brute-Force Matcher (BFMatcher)

Compares every descriptor in set A against every descriptor in set B. Guaranteed to find the true nearest neighbour.

**When to use**: Small descriptor sets (< 2000 keypoints per image), or when accuracy is paramount.

**Complexity**: O(N × M) where N, M are descriptor counts.

```python
# For SIFT (float32, L2 norm)
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

# For ORB (binary uint8, Hamming norm)  
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
```

### Strategy 2: FLANN Matcher

Fast Library for Approximate Nearest Neighbours. Uses kd-trees or LSH for fast approximate search.

**When to use**: Large descriptor sets (> 2000 keypoints per image), or when speed matters.

**Caveat**: Approximate — may occasionally miss the true nearest neighbour.

```python
# For SIFT (float descriptors)
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# For ORB (binary descriptors)
FLANN_INDEX_LSH = 6
index_params = dict(
    algorithm=FLANN_INDEX_LSH,
    table_number=6,       # 12 is recommended in docs
    key_size=12,          # 20 is recommended
    multi_probe_level=1   # 2 is recommended
)
flann = cv2.FlannBasedMatcher(index_params, search_params)
```

---

## Outlier Filtering After Matching

Raw matches contain many false correspondences. Two primary filtering strategies:

### Lowe's Ratio Test (for SIFT and float descriptors)

Accept a match only if the nearest neighbour distance is significantly smaller than the second-nearest neighbour distance. This rejects ambiguous matches where two descriptors look similarly close to the query.

```python
# Requires knnMatch with k=2
matches = matcher.knnMatch(desc1, desc2, k=2)

# Apply ratio test
good_matches = []
for m, n in matches:
    if m.distance < ratio_threshold * n.distance:
        good_matches.append(m)
```

**Typical threshold**: 0.7–0.75. Lower = stricter (fewer but more reliable matches). Higher = more matches but more outliers.

**Tuning guidance**:
- Start with 0.75 (Lowe's original recommendation)
- If getting too few inliers after RANSAC, try raising to 0.80
- If RANSAC is still failing with many matches, lower to 0.65

### Cross-Check (for ORB and binary descriptors)

Accept a match (A→B) only if the reverse match (B→A) also selects the same pair. Eliminates many spurious matches.

```python
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(desc1, desc2)
matches = sorted(matches, key=lambda x: x.distance)
```

**Note**: Cross-check with `BFMatcher.match()` does NOT return k-nearest neighbours. Do NOT apply ratio test on top of cross-check matches — they are single matches, not pairs.

---

## Implementation

```python
# src/matching/matcher.py

import cv2
import numpy as np
import time

def create_matcher(method: str, config: dict) -> cv2.DescriptorMatcher:
    """
    Factory for matchers appropriate to the descriptor type.
    
    # REQ-05: Match descriptors between overlapping image pairs
    """
    method = method.upper()
    if method == 'SIFT':
        return cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    elif method == 'ORB':
        return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    else:
        raise ValueError(f"Unknown method: {method}")


def match_descriptors(
    desc1: np.ndarray,
    desc2: np.ndarray, 
    method: str,
    config: dict
) -> tuple:
    """
    Match descriptors and apply appropriate filtering.
    
    Returns:
        (raw_matches, good_matches, elapsed_time)
    """
    matcher = create_matcher(method, config)
    
    start = time.perf_counter()
    
    if method.upper() == 'SIFT':
        # kNN match then ratio test
        raw_matches_pairs = matcher.knnMatch(desc1, desc2, k=2)
        raw_matches = [m for m, n in raw_matches_pairs]
        ratio_threshold = config.get('ratio_threshold', 0.75)
        good_matches = apply_ratio_test(raw_matches_pairs, ratio_threshold)
        
    elif method.upper() == 'ORB':
        # Cross-check match (already filtered)
        good_matches = matcher.match(desc1, desc2)
        good_matches = sorted(good_matches, key=lambda x: x.distance)
        raw_matches = good_matches  # raw == filtered for cross-check
    
    elapsed = time.perf_counter() - start
    return raw_matches, good_matches, elapsed


def apply_ratio_test(knn_matches: list, threshold: float = 0.75) -> list:
    """
    Lowe's ratio test to filter ambiguous matches.
    """
    good = []
    for match_pair in knn_matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < threshold * n.distance:
                good.append(m)
    return good
```

---

## Extracting Point Coordinates from Matches

After matching, extract the matched point coordinates for RANSAC:

```python
def extract_matched_points(
    kp1: list,
    kp2: list,
    matches: list
) -> tuple:
    """
    Extract (x,y) coordinates of matched keypoints.
    
    Returns:
        (src_pts, dst_pts) as float32 arrays of shape (N, 1, 2)
    """
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    return src_pts, dst_pts
```

The `.reshape(-1, 1, 2)` format is required by `cv2.findHomography()`.

---

## Visualising Initial Correspondences (REQ-06)

```python
# src/visualization/draw.py

def visualize_matches(img1, kp1, img2, kp2, matches, title, outpath, max_display=100):
    """
    Draw match lines between two images and save to file.
    
    # REQ-06: Display initial feature correspondences
    """
    import matplotlib.pyplot as plt
    
    # Limit display for clarity
    display_matches = matches[:max_display] if len(matches) > max_display else matches
    
    img_matches = cv2.drawMatches(
        img1, kp1, img2, kp2, display_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    plt.figure(figsize=(16, 6))
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.title(f"{title} — {len(matches)} matches (showing {len(display_matches)})")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
```

---

## Metrics to Record

For every matching run, record:

```python
matching_metrics = {
    'method': method,
    'num_raw_matches': len(raw_matches),
    'num_good_matches': len(good_matches),
    'ratio_threshold': config.get('ratio_threshold', 0.75),
    'matching_time_s': elapsed,
    'avg_match_distance': float(np.mean([m.distance for m in good_matches])) if good_matches else 0.0,
}
```

---

## Expected Match Quality by Scenario

| Scenario | Expected Good Matches | Notes |
|---|---|---|
| Similar images (baseline) | High (50–500+) | Depends on scene texture |
| Rotation 30° | High | SIFT rotation-invariant; ORB should also handle |
| Rotation 90° | Moderate | ORB may struggle beyond pyramid range |
| Scale 0.5× | Moderate–High | SIFT handles well; ORB more sensitive |
| Strong illumination change | Low–Moderate | Both degrade; SIFT more robust |
| Extreme viewpoint | Low | Homography model may not hold |

---

## Common Mistakes to Avoid

- **Do not** apply ratio test to ORB matches (it uses cross-check, not k-NN)
- **Do not** apply cross-check to SIFT k-NN results (use ratio test instead)
- **Do not** forget to check that `len(good_matches) >= 4` before attempting RANSAC
- **Do not** display all matches when there are thousands — limit to top 100–200 for visualisation
- **Do not** sort SIFT matches by distance alone without first applying ratio test
- **Do not** use `cv2.NORM_L2` for ORB or `cv2.NORM_HAMMING` for SIFT
