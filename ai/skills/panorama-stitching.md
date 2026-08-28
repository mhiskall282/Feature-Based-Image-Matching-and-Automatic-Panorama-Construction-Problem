# Skill: Panorama Stitching

## Overview

This skill guides the implementation of the final panorama construction stage — compositing multiple warped images into a single wide-field image. This stage builds on the warping skill and adds multi-image ordering, canvas management, and blending strategies.

**Requirements satisfied**: REQ-10, REQ-15

---

## Panorama Construction Strategy

### Two-Image Panorama (Minimum Viable)
1. Match image 1 → image 2 (estimate H)
2. Warp image 1 into image 2's frame
3. Place image 2 on a canvas
4. Composite warped image 1 onto canvas

### Three-Image Panorama (Required by exam)
Two approaches:

**Approach A — Chain from reference**
Choose a reference image (typically the centre image). Estimate all homographies with respect to the reference.
```
H_left  = H(left → centre)
H_right = H(right → centre)

warp left  onto canvas via H_left
place centre on canvas (identity transform)
warp right  onto canvas via H_right
```

**Approach B — Sequential stitching**
Stitch pairs sequentially:
```
Step 1: Stitch img1 + img2 → panorama_12
Step 2: Match panorama_12 with img3
Step 3: Stitch panorama_12 + img3 → panorama_123
```

**Recommendation**: Approach A (reference-based) produces less cumulative error. Approach B is simpler to implement but errors compound across pairwise stitches.

---

## Reference Image Selection

For a sequence of overlapping images [img1, img2, img3, ...]:
- Choose the **centre image** as reference when possible
- This minimises the maximum warping distance for any image
- For 3 images: img2 (centre) is the reference
- For 4 images: img2 or img3 (either centre image)
- For 5 images: img3 (centre)

---

## Canvas Size for N Images

For N images with a reference image R:
1. Map all four corners of every image through their respective H matrices (H_i = H(img_i → R))
2. Find the global bounding box of all mapped corners
3. The canvas size = bounding box dimensions
4. x_offset, y_offset = negate minimum x and y if they are negative

```python
# src/stitching/stitch.py

def compute_panorama_canvas_size(images: list, H_list: list, ref_idx: int) -> tuple:
    """
    Compute the canvas size for a multi-image panorama.
    
    Args:
        images: List of image arrays
        H_list: List of homographies H_i where H_list[i] maps images[i] → reference
                H_list[ref_idx] should be the identity matrix
        ref_idx: Index of the reference image
    
    Returns:
        (canvas_width, canvas_height, x_offset, y_offset)
    """
    all_corners = []
    
    for i, (img, H) in enumerate(zip(images, H_list)):
        h, w = img.shape[:2]
        corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1, 1, 2)
        
        if H is None:
            # This image cannot be placed — skip
            continue
        
        mapped = cv2.perspectiveTransform(corners, H)
        all_corners.append(mapped)
    
    all_corners = np.concatenate(all_corners, axis=0)
    
    x_min, y_min = np.floor(all_corners.min(axis=0).ravel()).astype(int)
    x_max, y_max = np.ceil(all_corners.max(axis=0).ravel()).astype(int)
    
    x_offset = -x_min if x_min < 0 else 0
    y_offset = -y_min if y_min < 0 else 0
    
    canvas_width  = x_max - x_min
    canvas_height = y_max - y_min
    
    return canvas_width, canvas_height, x_offset, y_offset
```

---

## Building the Homography Chain

For Approach A (reference-based), you need H_i for each image relative to the reference:

```python
def build_homography_chain(images: list, ref_idx: int, detector_name: str, config: dict) -> list:
    """
    Estimate H_i for each image relative to the reference image.
    
    H_list[ref_idx] = identity matrix
    H_list[i] = H mapping images[i] → images[ref_idx]
    
    For non-adjacent images, chain homographies:
    H(img0 → ref) = H(img0 → img1) @ H(img1 → ref)   [if ref > 0]
    """
    # This is a simplified version — pair-wise estimation from each image to reference
    # More robust: sequential chain with error propagation analysis
    
    H_list = [None] * len(images)
    H_list[ref_idx] = np.eye(3, dtype=np.float64)  # Reference maps to itself
    
    ref_img = images[ref_idx]
    
    for i, img in enumerate(images):
        if i == ref_idx:
            continue
        
        # Match img[i] to reference
        # (This calls the full feature detection → matching → RANSAC pipeline)
        H, _, _, _, metrics = run_full_pipeline_pair(img, ref_img, detector_name, config)
        H_list[i] = H  # May be None if failed
        
    return H_list
```

---

## Multi-Image Stitching

```python
def stitch_multiple(
    images: list,
    H_list: list,
    ref_idx: int,
    config: dict
) -> np.ndarray:
    """
    Stitch multiple images into a panorama using precomputed homographies.
    
    # REQ-10: Stitch transformed images into a panorama
    # REQ-15: Demonstrate complete pipeline
    """
    canvas_w, canvas_h, x_off, y_off = compute_panorama_canvas_size(images, H_list, ref_idx)
    
    # Translation matrix for offset
    T = np.array([[1, 0, x_off], [0, 1, y_off], [0, 0, 1]], dtype=np.float64)
    
    # Initialize canvas
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    # Track which pixels have been filled (for blending)
    canvas_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    
    # Warp and place all images in order (reference first, then others)
    # Process in this order to give reference image priority in overlap
    order = [ref_idx] + [i for i in range(len(images)) if i != ref_idx]
    
    for i in order:
        img = images[i]
        H = H_list[i]
        
        if H is None:
            print(f"[WARNING] Skipping image {i} — homography not available")
            continue
        
        H_adjusted = T @ H
        
        warped = cv2.warpPerspective(
            img, H_adjusted, (canvas_w, canvas_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        
        # Create mask for this warped image (non-black pixels)
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        _, warped_mask = cv2.threshold(warped_gray, 1, 255, cv2.THRESH_BINARY)
        
        # Simple compositing: only place where canvas is not yet filled
        # (reference image placed first has priority)
        empty_regions = canvas_mask == 0
        canvas[empty_regions & (warped_mask > 0)] = warped[empty_regions & (warped_mask > 0)]
        canvas_mask[warped_mask > 0] = 255
    
    return canvas
```

---

## Blending Strategies

### 1. Simple Overwrite (Implemented above)
- Reference pixels dominate
- Creates hard seams at image boundaries
- Simple but often has visible seam lines

### 2. Alpha Blending in Overlap Region

```python
def alpha_blend_pair(warped1: np.ndarray, warped2: np.ndarray) -> np.ndarray:
    """
    Alpha blend two warped images in their overlap region.
    Each pixel in the overlap is a weighted average based on distance from seam.
    """
    # Compute masks
    mask1 = (np.any(warped1 > 0, axis=2)).astype(np.float32)
    mask2 = (np.any(warped2 > 0, axis=2)).astype(np.float32)
    
    # Distance transform for smooth blending weights
    dist1 = cv2.distanceTransform((mask1 * 255).astype(np.uint8), cv2.DIST_L2, 5)
    dist2 = cv2.distanceTransform((mask2 * 255).astype(np.uint8), cv2.DIST_L2, 5)
    
    # Normalize weights
    total = dist1 + dist2
    total[total == 0] = 1
    alpha1 = dist1 / total
    alpha2 = dist2 / total
    
    # Blend
    result = np.zeros_like(warped1, dtype=np.float32)
    for c in range(3):
        result[:,:,c] = (alpha1 * warped1[:,:,c] + alpha2 * warped2[:,:,c])
    
    return result.astype(np.uint8)
```

### 3. Multi-band Blending (Optional)
Applies blending at multiple frequency scales using Gaussian and Laplacian pyramids. Best quality but significantly more complex. If implemented, cite the technique. If not implemented, acknowledge the limitation.

---

## Pipeline Demonstration Visualization (REQ-15)

Create a pipeline summary visualization showing all stages:

```python
def create_pipeline_visualization(
    img1, img2, kp1, kp2, good_matches, inlier_matches,
    mask, H, warped, panorama, outpath
):
    """
    Multi-panel figure showing the complete pipeline stages.
    """
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(24, 16))
    
    panels = [
        (img1, 'Input Image 1', 231),
        (img2, 'Input Image 2', 232),
    ]
    # ... build panels for each stage and save
    plt.savefig(outpath, dpi=120, bbox_inches='tight')
    plt.close()
```

---

## Output Files

```
outputs/{experiment}/{method}/
├── panorama_{method}.png             ← Final panorama (cropped)
├── panorama_{method}_uncropped.png   ← Full canvas (with black borders)
├── pipeline_summary_{method}.png     ← Multi-panel pipeline visualization
└── stitching_stages/
    ├── stage1_input.png
    ├── stage2_preprocessed.png
    ├── stage3_keypoints.png
    ├── stage4_raw_matches.png
    ├── stage5_ransac_inliers.png
    ├── stage6_warped.png
    └── stage7_panorama.png
```

---

## Common Mistakes to Avoid

- **Do not** use `cv2.Stitcher_create().stitch()` as the sole method — it is a black box that does not demonstrate understanding of the pipeline
- **Do not** assume that more images always produces a better panorama — each pairwise H estimation introduces error
- **Do not** forget to handle failed homographies (H is None) gracefully — skip that image and log the failure
- **Do not** composite all warped images without considering order — the first image placed has priority in simple overwrite
- **Do not** report panorama quality without at least a visual quality checklist (ghosting, seams, alignment)
