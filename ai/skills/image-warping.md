# Skill: Image Warping

## Overview

This skill guides the implementation of the image warping stage. Warping applies the estimated homography H to transform one image into the coordinate frame of another, which is the mathematical foundation of image alignment in panorama construction.

**Requirements satisfied**: REQ-09

---

## Mathematical Background

### Projective Transformation (Homography)
Given a homography matrix H (3×3), a point **p = (x, y)** in image 1 maps to point **p' = (x', y')** in image 2's coordinate frame:

```
[x_w]   [h00 h01 h02]   [x]
[y_w] = [h10 h11 h12] × [y]
[w  ]   [h20 h21 h22]   [1]

x' = x_w / w
y' = y_w / w
```

This is a perspective (projective) transformation — the most general linear transformation of image coordinates. It encodes rotation, translation, scale, shear, and perspective distortion.

### Forward vs Inverse Warping
OpenCV's `warpPerspective` applies the transformation using **inverse warping** (backward mapping):
- For each pixel in the OUTPUT image, compute where it comes from in the INPUT image
- Sample the input image at that location (with interpolation)
- This avoids holes in the output (unlike forward mapping)

---

## Canvas Size Computation

Before warping, you must determine how large the output canvas needs to be.

### The Problem
When you warp image 1 into image 2's coordinate frame, the warped result may extend:
- To the right (if the panorama is wider than image 2)
- To the left (if warped image 1 extends into negative x coordinates)
- Above or below

The output canvas must accommodate both images.

### Algorithm

```python
# src/warping/warp.py

import cv2
import numpy as np

def compute_canvas_size(img1: np.ndarray, img2: np.ndarray, H: np.ndarray) -> tuple:
    """
    Compute the output canvas dimensions and offset needed to contain
    both the reference image (img2) and the warped source image (img1 warped by H).
    
    Returns:
        (canvas_width, canvas_height, x_offset, y_offset)
        x_offset, y_offset: Translation to apply so no image has negative coordinates
        
    # REQ-09: Transform one image into another image's coordinate system
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Corners of image 1 (source)
    corners_img1 = np.float32([[0,0], [w1,0], [w1,h1], [0,h1]]).reshape(-1, 1, 2)
    
    # Map corners of image 1 through H into image 2's frame
    corners_img1_warped = cv2.perspectiveTransform(corners_img1, H)
    
    # Corners of image 2 (reference — stays fixed)
    corners_img2 = np.float32([[0,0], [w2,0], [w2,h2], [0,h2]]).reshape(-1, 1, 2)
    
    # All corners together
    all_corners = np.concatenate([corners_img2, corners_img1_warped], axis=0)
    
    # Bounding box
    x_min, y_min = np.floor(all_corners.min(axis=0).ravel()).astype(int)
    x_max, y_max = np.ceil(all_corners.max(axis=0).ravel()).astype(int)
    
    # Offset to make all coordinates non-negative
    x_offset = -x_min if x_min < 0 else 0
    y_offset = -y_min if y_min < 0 else 0
    
    canvas_width  = x_max - x_min
    canvas_height = y_max - y_min
    
    return canvas_width, canvas_height, x_offset, y_offset
```

---

## Warping the Image

```python
def warp_image(
    img: np.ndarray,
    H: np.ndarray,
    canvas_width: int,
    canvas_height: int,
    x_offset: int,
    y_offset: int
) -> np.ndarray:
    """
    Warp img using homography H onto a canvas of specified size.
    Apply (x_offset, y_offset) translation to handle negative coordinates.
    
    # REQ-09: Transform one image into another image's coordinate system
    """
    # Translation matrix to handle negative coordinate offset
    T = np.array([
        [1, 0, x_offset],
        [0, 1, y_offset],
        [0, 0, 1        ]
    ], dtype=np.float64)
    
    # Compose translation with homography: T @ H
    H_adjusted = T @ H
    
    warped = cv2.warpPerspective(
        img,
        H_adjusted,
        (canvas_width, canvas_height),
        flags=cv2.INTER_LINEAR,         # Bilinear interpolation
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)           # Black fill for out-of-bounds areas
    )
    
    return warped
```

### Interpolation Options

| Flag | Method | Use case |
|---|---|---|
| `cv2.INTER_NEAREST` | Nearest-neighbour | Fast but blocky; avoid for final panorama |
| `cv2.INTER_LINEAR` | Bilinear | Good default for photographic images |
| `cv2.INTER_CUBIC` | Bicubic | Higher quality, slower |
| `cv2.INTER_LANCZOS4` | Lanczos | Highest quality, slowest |

**Recommendation**: Use `cv2.INTER_LINEAR` for experiments (fast), `cv2.INTER_CUBIC` for the final panorama output.

---

## Placing the Reference Image on the Canvas

The reference image (image 2) also needs to be placed on the larger canvas with the correct offset:

```python
def place_reference_on_canvas(
    img2: np.ndarray,
    canvas_width: int,
    canvas_height: int,
    x_offset: int,
    y_offset: int
) -> np.ndarray:
    """
    Place the reference image (image 2) onto the output canvas with offset applied.
    """
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    h2, w2 = img2.shape[:2]
    canvas[y_offset:y_offset+h2, x_offset:x_offset+w2] = img2
    return canvas
```

---

## Simple Composite (Stitch Two Images)

```python
def composite_images(warped_img1: np.ndarray, canvas_with_img2: np.ndarray) -> np.ndarray:
    """
    Composite the warped image onto the canvas containing the reference image.
    Uses simple overwrite: reference image pixels take priority in overlap.
    
    For a more advanced result, use alpha blending (see stitching skill).
    """
    result = warped_img1.copy()
    
    # Mask: where does the canvas (reference image) have valid pixels?
    ref_mask = np.any(canvas_with_img2 > 0, axis=2)
    
    # Place reference image pixels over warped image
    result[ref_mask] = canvas_with_img2[ref_mask]
    
    return result
```

---

## Cropping Black Borders

After stitching, the canvas may have large black border regions. Crop them:

```python
def crop_black_borders(img: np.ndarray) -> np.ndarray:
    """
    Remove black border rows/columns from panorama.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return img
    
    # Bounding box of the largest contour
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return img[y:y+h, x:x+w]
```

---

## Verifying the Warp Visually

Always generate a visualization of the warped image alongside the reference:

```python
def visualize_warp(img2, warped_img1, composite, outpath):
    """
    Side-by-side: reference | warped source | composite result
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    axes[0].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Reference Image')
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(warped_img1, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Warped Source Image')
    axes[1].axis('off')
    
    axes[2].imshow(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB))
    axes[2].set_title('Composite Result')
    axes[2].axis('off')
    
    plt.suptitle('Image Warping Result', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
```

---

## Multi-Image Warping (3+ Images)

For 3 or more images, chain homographies with respect to a reference (typically the centre image):

```
H_left_to_centre = (estimated between left and centre image)
H_right_to_centre = (estimated between right and centre image)

canvas = large canvas
place centre image on canvas (at offset position)
warp left image → canvas using H_left_to_centre
warp right image → canvas using H_right_to_centre
composite all three
```

See `/ai/skills/panorama-stitching.md` for the multi-image strategy.

---

## Common Mistakes to Avoid

- **Do not** use the raw H without accounting for negative-coordinate offsets — the result will be cropped or misaligned
- **Do not** assume the canvas size equals image 2's size — it must be large enough to contain both images
- **Do not** use INTER_NEAREST for the final panorama output — visible blocky artefacts are unacceptable
- **Do not** forget that warped images will have large black regions (border fill) — handle appropriately in compositing
- **Do not** warp in grayscale and then try to stitch in colour — maintain colour throughout
