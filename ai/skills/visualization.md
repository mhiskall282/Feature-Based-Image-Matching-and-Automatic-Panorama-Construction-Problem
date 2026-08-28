# Skill: Visualization

## Overview

This skill defines all visualization outputs required by the project — their content, style, filenames, and output directory conventions. Consistent, well-labelled visualizations are essential for both the experimental analysis and the academic report.

**Requirements satisfied**: REQ-06, REQ-11, REQ-15 (visual evidence for all pipeline stages)

---

## Visualization Philosophy

1. **Every output figure must be self-explanatory**: Include a title, method name, image pair identifier, and relevant metric count in the figure title.
2. **Consistent colour coding across all figures**: Use the same colour conventions for all figures in the project.
3. **Save at high resolution**: Use `dpi=150` minimum for paper-quality figures. Use `dpi=200` for the final report figures.
4. **Both display and save**: During development, show figures interactively. For experiment runs, save only (no `plt.show()`).
5. **Reproducible**: Figures must be generated programmatically from data — never hand-drawn or edited in external tools.

---

## Colour Conventions

Apply these consistently across all figures:

| Element | Colour |
|---|---|
| SIFT keypoints | Blue (`(255, 150, 0)` BGR or `#0096FF` hex) |
| ORB keypoints | Orange (`(0, 150, 255)` BGR or `#FF9600` hex) |
| Match lines (raw/all) | Red |
| RANSAC inlier match lines | Green |
| RANSAC outlier match lines | Red |
| Warped region boundary | Yellow |
| Reference image region | Cyan outline |

---

## Required Visualizations

### VIZ-01: Input Images

```python
def visualize_input_images(images: list, names: list, outpath: str):
    """
    Display all input images side by side.
    """
    import matplotlib.pyplot as plt
    
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5))
    if n == 1:
        axes = [axes]
    
    for ax, img, name in zip(axes, images, names):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(f'{name}\n{img.shape[1]}×{img.shape[0]} px')
        ax.axis('off')
    
    plt.suptitle('Input Images', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
```

**Output**: `outputs/{exp}/{method}/input_images.png`

---

### VIZ-02: Preprocessed Images

```python
def visualize_preprocessed(img_color, img_gray, name, outpath):
    """
    Show original colour and grayscale preprocessed versions.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f'{name} — Colour (BGR→RGB)')
    axes[0].axis('off')
    
    axes[1].imshow(img_gray, cmap='gray')
    axes[1].set_title(f'{name} — Grayscale (for feature detection)')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
```

**Output**: `outputs/{exp}/{method}/preprocessed_{img_name}.png`

---

### VIZ-03: Detected Keypoints

```python
def visualize_keypoints(img_gray, keypoints, method, img_name, outpath, max_kp=500):
    """
    Draw detected keypoints on the image.
    Rich keypoints (with scale/orientation) are drawn as circles with orientation line.
    """
    # Draw keypoints with scale and orientation
    img_kp = cv2.drawKeypoints(
        img_gray, 
        keypoints[:max_kp],
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    
    plt.figure(figsize=(10, 7))
    plt.imshow(cv2.cvtColor(img_kp, cv2.COLOR_BGR2RGB))
    plt.title(f'{method} Keypoints — {img_name}\n'
              f'{len(keypoints)} detected (showing {min(len(keypoints), max_kp)})',
              fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
```

**Output**: `outputs/{exp}/{method}/keypoints_{method}_{img_name}.png`

**Note**: `DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS` draws circles sized to the keypoint scale with an orientation line — this is much more informative than simple dots.

---

### VIZ-04: Keypoint Size Distribution (Optional but valuable)

```python
def plot_keypoint_size_distribution(kp_dict: dict, outpath: str):
    """
    Histogram of keypoint sizes (scales) for each method.
    """
    fig, axes = plt.subplots(1, len(kp_dict), figsize=(6*len(kp_dict), 5))
    if len(kp_dict) == 1:
        axes = [axes]
    
    for ax, (method, keypoints) in zip(axes, kp_dict.items()):
        sizes = [kp.size for kp in keypoints]
        ax.hist(sizes, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        ax.set_title(f'{method} — Scale Distribution\nN={len(keypoints)}')
        ax.set_xlabel('Keypoint Size (scale)')
        ax.set_ylabel('Count')
    
    plt.suptitle('Keypoint Scale Distributions', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
```

**Output**: `outputs/{exp}/comparison/keypoint_scale_distribution.png`

---

### VIZ-05: Raw Matches (Initial Correspondences)

```python
def visualize_matches(img1, kp1, img2, kp2, matches, method, pair_name, outpath, max_display=150):
    """
    # REQ-06: Display initial feature correspondences
    """
    display = matches[:max_display]
    
    img_matches = cv2.drawMatches(
        img1, kp1, img2, kp2, display, None,
        matchColor=(0, 0, 255),    # Red lines for matches
        singlePointColor=(200, 200, 200),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    plt.figure(figsize=(16, 6))
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.title(f'{method} — Initial Matches: {pair_name}\n'
              f'{len(matches)} matches (showing {len(display)})', fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
```

**Output**: `outputs/{exp}/{method}/raw_matches_{method}_{pair}.png`

---

### VIZ-06: Before/After RANSAC Comparison

```python
def visualize_before_after_ransac(img1, kp1, img2, kp2,
                                   all_matches, inlier_matches,
                                   method, pair_name, outpath):
    """
    # REQ-11: Compare feature matching before and after RANSAC
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    
    # Before RANSAC
    img_before = cv2.drawMatches(
        img1, kp1, img2, kp2, all_matches[:200], None,
        matchColor=(0, 0, 255),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    axes[0].imshow(cv2.cvtColor(img_before, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f'Before RANSAC: {len(all_matches)} matches', fontsize=12)
    axes[0].axis('off')
    
    # After RANSAC (inliers only)
    img_after = cv2.drawMatches(
        img1, kp1, img2, kp2, inlier_matches, None,
        matchColor=(0, 255, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    ratio = len(inlier_matches) / len(all_matches) if all_matches else 0.0
    axes[1].imshow(cv2.cvtColor(img_after, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'After RANSAC: {len(inlier_matches)} inliers ({ratio:.1%})', fontsize=12)
    axes[1].axis('off')
    
    plt.suptitle(f'{method} — Before vs After RANSAC: {pair_name}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
```

**Output**: `outputs/{exp}/{method}/before_after_ransac_{method}_{pair}.png`

---

### VIZ-07: Warped Image

```python
def visualize_warped(img_ref, img_warped, method, outpath):
    """Show reference image alongside warped source image."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Reference Image')
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(img_warped, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'Warped Source Image ({method})')
    axes[1].axis('off')
    
    plt.suptitle('Image Warping Result', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
```

**Output**: `outputs/{exp}/{method}/warped_{method}.png`

---

### VIZ-08: Final Panorama

```python
def visualize_panorama(panorama, method, experiment, outpath):
    plt.figure(figsize=(20, 8))
    plt.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
    plt.title(f'Final Panorama — {method}\nExperiment: {experiment}\n'
              f'Dimensions: {panorama.shape[1]}×{panorama.shape[0]} px', fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
```

**Output**: `outputs/{exp}/{method}/panorama_{method}.png`

---

### VIZ-09: Method Comparison Bar Chart

```python
def plot_method_comparison(df: pd.DataFrame, metrics: list, outpath: str):
    """
    Grouped bar chart comparing methods across key metrics.
    """
    methods = df['method'].unique()
    n_metrics = len(metrics)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 6))
    if n_metrics == 1:
        axes = [axes]
    
    colors = {'SIFT': '#2196F3', 'ORB': '#FF9800'}
    
    for ax, metric in zip(axes, metrics):
        for i, method in enumerate(methods):
            val = df[df['method'] == method][metric].values
            ax.bar(i, val.mean(), color=colors.get(method, '#9E9E9E'),
                   label=method, alpha=0.85, edgecolor='black')
        
        ax.set_title(metric.replace('_', '\n'), fontsize=10)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods)
        ax.set_ylabel(metric)
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('SIFT vs ORB — Metric Comparison (Baseline)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
```

**Output**: `outputs/comparison/method_comparison_bar.png`

---

### VIZ-10: Experiment Line Plots

One line plot per experiment × metric, showing how performance changes with the condition:

```python
def plot_experiment_trend(df, x_col, y_col, title, outpath):
    """
    Line plot: y_col vs x_col, one line per method.
    Used for rotation, scale, viewpoint, illumination experiments.
    """
    # ... (see experimental-evaluation.md for implementation)
```

**Outputs**:
- `outputs/rotation/inlier_ratio_vs_angle.png`
- `outputs/scale/inlier_ratio_vs_scale.png`
- `outputs/viewpoint/inlier_ratio_vs_viewpoint.png`
- `outputs/illumination/inlier_ratio_vs_illumination.png`

---

## Output Directory Naming Convention

```
outputs/
├── {experiment_name}/
│   ├── {method}/
│   │   ├── VIZ-01: input_images.png
│   │   ├── VIZ-02: preprocessed_{img}.png
│   │   ├── VIZ-03: keypoints_{method}_{img}.png
│   │   ├── VIZ-05: raw_matches_{method}_{pair}.png
│   │   ├── VIZ-06: before_after_ransac_{method}_{pair}.png
│   │   ├── VIZ-07: warped_{method}.png
│   │   └── VIZ-08: panorama_{method}.png
│   └── comparison/
│       ├── VIZ-04: keypoint_scale_distribution.png
│       └── VIZ-09: method_comparison_bar.png
└── report_figures/              ← High-resolution copies for the report (dpi=200)
```

---

## Common Mistakes to Avoid

- **Do not** save figures without titles — an untitled figure in the report is unprofessional
- **Do not** use `plt.show()` in experiment scripts — it blocks execution; use `plt.savefig()` only
- **Do not** use default matplotlib colours without a consistent scheme — establish colour conventions and stick to them
- **Do not** generate figures without axis labels or legends where they are needed
- **Do not** use very low DPI (< 100) — figures will look blurry in the report
