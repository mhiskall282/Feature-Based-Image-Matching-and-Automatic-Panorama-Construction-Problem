# Dataset — Image Acquisition Guide

## Overview

This directory holds images for the panorama pipeline.

```
data/
├── raw/           ← Place your 3+ real images here
├── processed/     ← Auto-generated preprocessed versions
└── transformed/   ← Auto-generated synthetic experiment images
```

## How to Acquire Suitable Images

### Requirements for Real Images

You need **3 or more** overlapping photographs of the same scene:

| Property | Requirement |
|---|---|
| Overlap | 30–60% overlap between adjacent images |
| Features | Scene must have distinctive visual texture (not plain walls/sky) |
| Viewpoint | Captured from slightly different positions (pan or step sideways) |
| Blur | Minimal motion blur |
| Format | JPG or PNG |
| Size | Any (pipeline auto-resizes to ≤1280px if needed) |

### Recommended Scenes

- Outdoor building facades or street scenes
- Indoor rooms with furniture/books
- Landscapes with clear foreground features
- Architecture or murals

### Naming Convention

Name files so they sort in left-to-right order:
```
scene_01.jpg
scene_02.jpg
scene_03.jpg
```

### Synthetic Images (for Experiments)

Rotation, scale, viewpoint, and illumination experiment images are
generated automatically by:
```
python scripts/prepare_dataset.py --input data/raw --output data/transformed
```

All files in `data/transformed/` are SYNTHETIC — programmatically derived
from the raw images for controlled experiments. They are not real-world acquisitions.

## Image Checklist

- [ ] At least 3 images placed in `data/raw/`
- [ ] Images are named in order
- [ ] Scene has distinctive visual features (not blank walls)
- [ ] Images overlap by approximately 30–60%
- [ ] Images are not severely blurred
