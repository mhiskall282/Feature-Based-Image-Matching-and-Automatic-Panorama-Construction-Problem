# Skill: Experimental Evaluation

## Overview

This skill guides the design, execution, and recording of the four controlled experiments required by the examination: rotation, scale, viewpoint, and illumination. Each experiment follows the same structure but varies one controlled variable while holding others constant.

**Requirements satisfied**: REQ-12, REQ-13, REQ-14

---

## Experiment Design Principles

1. **One variable at a time**: Each experiment changes only one condition. All other parameters (detector settings, matching thresholds, RANSAC parameters, reference image) remain constant.

2. **Same codebase for all experiments**: The same pipeline code is used for every experiment. There is no experiment-specific hacking of parameters unless that change is documented and justified.

3. **Both methods per experiment**: Every experiment runs with both SIFT and ORB. This enables direct comparison under each condition.

4. **Fixed random seed**: Set `np.random.seed(42)` at the start of every experiment run.

5. **Save everything**: Results to CSV, visualizations to PNG, config to JSON. Nothing computed only in memory.

6. **Document failures**: If a method fails under a condition, record it as a failure — do not skip or adjust parameters post-hoc without documenting.

---

## Experiment Template Structure

Every experiment script follows this template:

```python
# experiments/run_{experiment_name}.py

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Project imports
from src.data.loader import load_image
from src.preprocessing.prepare import preprocess_pipeline
from src.features.detection import create_detector
from src.features.description import compute_descriptors
from src.matching.matcher import match_descriptors
from src.homography.ransac import apply_ransac
from src.warping.warp import compute_canvas_size, warp_image
from src.stitching.stitch import stitch_multiple
from src.evaluation.metrics import compute_metrics, save_metrics
from src.visualization.draw import (
    visualize_keypoints, visualize_matches,
    visualize_before_after_ransac, visualize_warp
)
from experiments.config import EXPERIMENT_CONFIG

def run_single_trial(img1, img2, method, config, output_dir, trial_id):
    """
    Run the full pipeline for one image pair, one method, one condition.
    Returns a metrics dict.
    """
    np.random.seed(config.get('random_seed', 42))
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Preprocessing
    img1_color, img1_gray = preprocess_pipeline(img1, config)
    img2_color, img2_gray = preprocess_pipeline(img2, config)
    
    # Detection + Description
    detector = create_detector(method, config)
    kp1, desc1, t_det1 = compute_descriptors(img1_gray, detector)
    kp2, desc2, t_det2 = compute_descriptors(img2_gray, detector)
    
    # Matching
    raw_matches, good_matches, t_match = match_descriptors(desc1, desc2, method, config)
    
    # RANSAC + Homography
    H, mask, inliers, outliers, ransac_metrics = apply_ransac(kp1, kp2, good_matches, config)
    
    # Warping + Panorama (only if homography succeeded)
    pano = None
    if H is not None:
        canvas_w, canvas_h, x_off, y_off = compute_canvas_size(img1_color, img2_color, H)
        warped = warp_image(img1_color, H, canvas_w, canvas_h, x_off, y_off)
        # ... stitch and save ...
    
    # Collect all metrics
    metrics = {
        'trial_id': trial_id,
        'method': method,
        'num_kp_img1': len(kp1),
        'num_kp_img2': len(kp2),
        'num_raw_matches': len(raw_matches),
        'num_good_matches': len(good_matches),
        **ransac_metrics,
        'time_detection_s': t_det1 + t_det2,
        'time_matching_s': t_match,
    }
    
    # Save visualizations
    visualize_keypoints(img1_gray, kp1, f"{method} Keypoints - Image 1", output_dir / f"kp1_{method}.png")
    visualize_matches(img1_gray, kp1, img2_gray, kp2, good_matches[:200], 
                      f"Raw Matches - {method}", output_dir / f"raw_matches_{method}.png")
    if mask is not None:
        visualize_before_after_ransac(img1_gray, kp1, img2_gray, kp2,
                                      good_matches, inliers, outliers,
                                      output_dir / f"ransac_{method}.png")
    
    return metrics


def run_experiment(conditions, methods, config, output_base, results_path):
    """
    Run the full experiment across all conditions and methods.
    """
    all_results = []
    
    for condition_id, condition_data in conditions.items():
        img1 = load_image(condition_data['img1_path'])
        img2 = load_image(condition_data['img2_path'])
        
        for method in methods:
            output_dir = Path(output_base) / condition_id / method
            metrics = run_single_trial(img1, img2, method, config, output_dir, condition_id)
            metrics['condition'] = condition_id
            metrics.update(condition_data.get('metadata', {}))
            all_results.append(metrics)
            print(f"  [{condition_id}] [{method}] inliers={metrics.get('num_ransac_inliers', 'N/A')} "
                  f"ratio={metrics.get('inlier_ratio', 0.0):.3f}")
    
    df = pd.DataFrame(all_results)
    df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    return df
```

---

## Experiment A: Rotation

### Purpose
Measure how each method degrades as the angular difference between images increases.

### Image Preparation
Generate rotated versions of a reference image programmatically:

```python
# In experiments/run_rotation.py

def generate_rotated_images(ref_img: np.ndarray, angles: list) -> dict:
    """
    Generate rotated copies of ref_img.
    Returns dict: {angle_str: rotated_img}
    """
    h, w = ref_img.shape[:2]
    cx, cy = w // 2, h // 2
    
    rotated = {}
    for angle in angles:
        M = cv2.getRotationMatrix2D((cx, cy), angle, scale=1.0)
        
        # Compute new canvas size to avoid cropping
        cos_a = abs(M[0, 0])
        sin_a = abs(M[0, 1])
        new_w = int(h * sin_a + w * cos_a)
        new_h = int(h * cos_a + w * sin_a)
        
        # Adjust translation to centre the rotated image
        M[0, 2] += (new_w / 2) - cx
        M[1, 2] += (new_h / 2) - cy
        
        rot_img = cv2.warpAffine(ref_img, M, (new_w, new_h))
        rotated[f'rot_{angle:03d}'] = {
            'img': rot_img,
            'angle': angle,
            'ground_truth_M': M  # Save for verification
        }
    
    return rotated

ROTATION_ANGLES = [0, 15, 30, 45, 60, 90, 120, 180]
```

### What to Report
For each rotation angle, for each method:
- All standard metrics (M1–M8)
- Additional column: `rotation_angle_deg`

### Expected Findings
- SIFT: should maintain high inlier ratio up to 180° (fully rotation-invariant descriptor)
- ORB: should handle up to ~360° via orientation assignment, but may degrade at extreme angles
- Both may fail if the rotation creates too little overlap (for large angles, the rotated image shares very few regions)

### Ground Truth (Synthetic Rotation Only)
For synthetic rotation, the true H can be derived from the rotation matrix M:
```
H_true = [M_2x2  | t] where M_2x2 is the rotation block and t is translation
         [0  0   | 1]
```
Compare `H_estimated` to `H_true` by computing `np.linalg.norm(H_est - H_true)` (after normalisation).

---

## Experiment B: Scale

### Purpose
Measure performance when images represent the same scene at different scales.

### Image Preparation

```python
def generate_scaled_images(ref_img: np.ndarray, scale_factors: list) -> dict:
    """
    Resize ref_img by each scale factor.
    Resize back to original size to normalize dimensions for fair comparison.
    """
    h, w = ref_img.shape[:2]
    scaled = {}
    
    for s in scale_factors:
        new_w, new_h = int(w * s), int(h * s)
        # Scale down
        small = cv2.resize(ref_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        # Scale back up to original size (introduces blur/aliasing)
        restored = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        scaled[f'scale_{s:.2f}'] = {
            'img': restored,
            'scale_factor': s
        }
    
    return scaled

SCALE_FACTORS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
```

### What to Report
All standard metrics + `scale_factor` column. Plot inlier_ratio vs scale_factor for each method.

---

## Experiment C: Viewpoint

### Purpose
Test matching under perspective distortion / viewpoint change.

### Image Preparation (Synthetic)

```python
def generate_perspective_images(ref_img: np.ndarray, levels: list) -> dict:
    """
    Apply controlled perspective distortion.
    """
    h, w = ref_img.shape[:2]
    results = {}
    
    for level_name, offset in levels:
        # Source points (corners of original)
        src = np.float32([[0,0], [w,0], [w,h], [0,h]])
        # Destination points (simulate perspective tilt)
        dst = np.float32([[offset, offset], [w-offset, 0], [w, h], [0, h-offset]])
        
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(ref_img, M, (w, h))
        
        results[level_name] = {
            'img': warped,
            'viewpoint_offset': offset,
            'ground_truth_H': M
        }
    
    return results

VIEWPOINT_LEVELS = [('mild', 20), ('moderate', 60), ('extreme', 120)]
```

---

## Experiment D: Illumination

### Purpose
Test matching under different lighting conditions.

### Image Preparation

```python
def generate_illumination_variants(ref_img: np.ndarray) -> dict:
    """
    Create illumination variants of the reference image.
    """
    variants = {}
    
    # Brightness changes
    for beta in [-100, -50, 50, 100]:
        name = f"bright_{'p' if beta >= 0 else 'n'}{abs(beta)}"
        variant = cv2.convertScaleAbs(ref_img, alpha=1.0, beta=beta)
        variants[name] = {'img': variant, 'brightness_delta': beta, 'contrast_factor': 1.0}
    
    # Contrast changes  
    for alpha in [0.5, 0.75, 1.25, 1.5]:
        name = f"contrast_{alpha:.2f}"
        variant = cv2.convertScaleAbs(ref_img, alpha=alpha, beta=0)
        variants[name] = {'img': variant, 'brightness_delta': 0, 'contrast_factor': alpha}
    
    # Histogram equalization
    gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized_gray = clahe.apply(gray)
    equalized = cv2.cvtColor(equalized_gray, cv2.COLOR_GRAY2BGR)
    variants['equalized'] = {'img': equalized, 'brightness_delta': 0, 'contrast_factor': 'CLAHE'}
    
    return variants
```

---

## Results Aggregation and Plotting

After all experiments, aggregate and compare:

```python
def plot_metric_vs_condition(df: pd.DataFrame, metric: str, condition_col: str,
                              title: str, outpath: str):
    """
    Line plot: metric vs condition value, one line per method.
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for method in df['method'].unique():
        method_df = df[df['method'] == method].sort_values(condition_col)
        ax.plot(method_df[condition_col], method_df[metric], 
                marker='o', label=method, linewidth=2)
    
    ax.set_xlabel(condition_col.replace('_', ' ').title())
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
```

---

## Run All Experiments Script

```python
# experiments/run_all.py

def main():
    print("="*60)
    print("CSCD608 Panorama Construction — Full Experiment Suite")
    print("="*60)
    
    print("\n[1/5] Running baseline experiment...")
    # from experiments.run_baseline import run_baseline
    # run_baseline()
    
    print("\n[2/5] Running rotation experiment...")
    print("\n[3/5] Running scale experiment...")
    print("\n[4/5] Running viewpoint experiment...")
    print("\n[5/5] Running illumination experiment...")
    print("\n[6/6] Running method comparison...")
    
    print("\nAll experiments complete. Results in: results/")

if __name__ == '__main__':
    main()
```
