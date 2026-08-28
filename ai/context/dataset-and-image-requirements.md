# Dataset and Image Requirements

## Purpose

This document defines the requirements for acquiring, organising, and validating the image dataset used in this project. Image quality and dataset design directly affect whether the pipeline can succeed and whether experimental conclusions are meaningful.

---

## 1. Baseline Dataset Requirements

### Minimum Specification

| Requirement | Minimum | Recommended |
|---|---|---|
| Number of images | 3 | 4–5 |
| Scene overlap | ~30% | 40–60% |
| Resolution | 640×480 | 1280×720 or higher |
| Format | JPEG or PNG | PNG (lossless) |
| Colour | RGB/BGR | RGB/BGR |
| Scene type | Textured outdoor/indoor | Textured outdoor scene |

### Scene Selection Guidelines

**Good scene characteristics:**
- Rich texture (buildings, brickwork, foliage, bookshelves, market stalls, street scenes)
- Approximately planar dominant surface OR purely rotational camera movement
- Sufficient overlap between adjacent images (30–60%)
- Consistent lighting conditions (for baseline)
- Static scene (no moving objects during capture)

**Poor scene characteristics (avoid for baseline):**
- Textureless scenes (blank walls, clear skies, flat water surfaces)
- Heavily repeating patterns (may cause ambiguous matching)
- Scenes with significant depth variation under large translation (violates homography assumption)
- Moving objects in the scene (people, vehicles)
- Extreme exposure differences between images

---

## 2. Image Capture Protocol

### Camera Setup
- Use a consistent camera device (smartphone camera is acceptable)
- Keep the camera at a fixed height and orientation (level with ground)
- For pure panorama: **rotate the camera about its optical centre** (not translate the camera bodily). This satisfies the pure-rotation assumption for homography.
- If slight translation is unavoidable, ensure the dominant scene plane is far enough to minimise parallax

### Overlap Strategy
- Image N and Image N+1 must share approximately 30–50% of their visual content
- More overlap is generally safer for matching but produces redundancy
- Less than 20% overlap risks insufficient matches for reliable homography

### Recommended Capture Sequence (3-image panorama)
```
Left image:   capture at position 0° (reference)
Centre image: capture at position ~25–35° to the right
Right image:  capture at position ~50–70° to the right
```

### File Naming Convention
```
data/
└── baseline/
    ├── img_01.jpg      (leftmost)
    ├── img_02.jpg      (centre)
    └── img_03.jpg      (rightmost)
```

---

## 3. Experiment-Specific Image Requirements

### A. Rotation Experiment
**Goal**: Test matching under controlled image rotation.

**Method 1 — Synthetic**: Apply `cv2.getRotationMatrix2D()` to create rotated copies of existing images programmatically.

**Method 2 — Real**: Capture same scene with camera rotated ±30°, ±45°, ±60°, ±90°.

**Required rotation angles**: 15°, 30°, 45°, 60°, 90° (test robustness gradient)

**Ground truth**: For synthetic rotation, the true H is known analytically and can be used to validate the estimated H.

```
data/
└── rotation/
    ├── ref.jpg          (reference image)
    ├── rot_015.jpg      (15° rotation)
    ├── rot_030.jpg      (30° rotation)
    ├── rot_045.jpg      (45° rotation)
    ├── rot_060.jpg      (60° rotation)
    └── rot_090.jpg      (90° rotation)
```

### B. Scale Experiment
**Goal**: Test matching under scale changes.

**Method — Synthetic**: Resize one image to different fractions of original size, then upscale back to original size (this introduces resolution loss and simulates distance change).

**Alternatively**: Capture same scene at different zoom levels / from different distances.

**Required scale factors**: 0.5×, 0.75×, 1.25×, 1.5×, 2.0× (relative to reference)

```
data/
└── scale/
    ├── ref.jpg
    ├── scale_050.jpg   (50% of original size)
    ├── scale_075.jpg
    ├── scale_125.jpg
    ├── scale_150.jpg
    └── scale_200.jpg
```

### C. Viewpoint Experiment
**Goal**: Test matching under viewpoint/perspective change.

**Method 1 — Real images**: Capture the same scene from different horizontal positions (simulates parallax and perspective change).

**Method 2 — Synthetic**: Apply `cv2.getPerspectiveTransform()` with controlled perspective distortion to simulate tilt.

```
data/
└── viewpoint/
    ├── ref.jpg
    ├── viewpoint_mild.jpg    (slight perspective change)
    ├── viewpoint_moderate.jpg
    └── viewpoint_extreme.jpg
```

### D. Illumination Experiment
**Goal**: Test matching under different lighting conditions.

**Method 1 — Real images**: Capture same scene at different times of day (morning, midday, afternoon) or with different artificial lighting.

**Method 2 — Synthetic**: Apply brightness/contrast adjustments:
```python
# Brightness change
bright = cv2.convertScaleAbs(img, alpha=1.0, beta=50)    # +50 brightness
dark   = cv2.convertScaleAbs(img, alpha=1.0, beta=-50)   # -50 brightness

# Contrast change
high_contrast = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
low_contrast  = cv2.convertScaleAbs(img, alpha=0.5, beta=0)
```

```
data/
└── illumination/
    ├── ref.jpg
    ├── bright_mild.jpg     (+50 brightness)
    ├── bright_strong.jpg   (+100 brightness)
    ├── dark_mild.jpg       (-50 brightness)
    ├── dark_strong.jpg     (-100 brightness)
    └── high_contrast.jpg   (1.5× contrast)
```

---

## 4. Dataset Directory Structure

```
data/
├── README.md                  ← Describe each image (scene, device, conditions)
├── baseline/
│   ├── img_01.jpg
│   ├── img_02.jpg
│   └── img_03.jpg
├── rotation/
│   ├── ref.jpg
│   ├── rot_015.jpg
│   ├── rot_030.jpg
│   ├── rot_045.jpg
│   ├── rot_060.jpg
│   └── rot_090.jpg
├── scale/
│   ├── ref.jpg
│   └── ...
├── viewpoint/
│   ├── ref.jpg
│   └── ...
└── illumination/
    ├── ref.jpg
    └── ...
```

---

## 5. Dataset Documentation Requirements

A `data/README.md` file must record:

- **Scene description**: What does the scene depict? Where was it captured?
- **Capture device**: Camera model, focal length if known
- **Capture date and time**: For each image or group
- **Approximate overlap**: Estimated percentage between adjacent pairs
- **Conditions**: Lighting, weather (for outdoor), any special conditions
- **Synthetic images**: If any images were generated programmatically, document the script and parameters used

---

## 6. Image Preprocessing Before Storage

Before saving images for the experiments:

1. Verify images load correctly with `cv2.imread()` — check for `None` return
2. Verify resolution is adequate (warn if < 400 pixels in any dimension)
3. Verify images are not corrupted (check file size, pixel statistics)
4. Resize very large images to a manageable resolution (e.g., cap at 1920 × 1080) to avoid excessive processing time during development
5. Keep original high-resolution copies separately

---

## 7. What NOT to Do with the Dataset

- **Do not** use images from the internet without documenting the source and verifying licence
- **Do not** fabricate "captured images" that were actually AI-generated
- **Do not** use images so blurry that no features can be detected — this tests camera quality, not algorithm quality
- **Do not** claim images were captured under specific conditions if they were not
- **Do not** mix dataset images between experiments without clear documentation

---

## 8. Fallback: Publicly Available Benchmark Images

If real captured images are unavailable, the following publicly available sources are acceptable (document the source in `data/README.md`):

- Oxford Buildings Dataset (VGG, University of Oxford) — features viewpoint and illumination variants
- DTU Robot Image Dataset — controlled viewpoint changes
- Brown's panorama dataset — multiple overlapping images designed for stitching
- Adobe panorama images — publicly released for research

**Always cite the source if using benchmark data.**

---

*This document governs all image acquisition decisions. Any deviation must be documented and justified in the report.*
