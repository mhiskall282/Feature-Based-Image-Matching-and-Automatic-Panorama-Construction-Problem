# CSCD608 ADVANCED COMPUTER VISION — COMPLETE PROJECT IMPLEMENTATION

You are now responsible for implementing the complete postgraduate computer vision examination project in this repository:

https://github.com/mhiskall282/Feature-Based-Image-Matching-and-Automatic-Panorama-Construction-Problem

The project is:

# Feature-Based Image Matching and Automatic Panorama Construction

This is for:

**MPhil/MSc Computer Science**
**Second Semester Examinations 2025/2026**
**CSCD608: ADVANCED COMPUTER VISION (3 CREDITS)**

TIME ALLOWED: ONE WEEK

You have already initialized the `/ai` project intelligence directory.

Before writing implementation code:

1. Inspect the repository.
2. Read `/ai/README.md`.
3. Read all relevant files under `/ai/context`.
4. Read all relevant files under `/ai/skills`.
5. Read all relevant files under `/ai/rules`.
6. Build an implementation plan.
7. Map every examination requirement to an implementation component and experiment.

Then implement the complete project.

---

# 1. CORE REQUIREMENT

Build a working Python/OpenCV computer vision system that takes at least three overlapping images of the same scene and automatically constructs a panorama using classical feature-based image matching.

The system must explicitly implement and demonstrate:

```text
Input Images
     ↓
Image Preparation
     ↓
Feature Detection
     ↓
Feature Description
     ↓
Descriptor Matching
     ↓
Initial Correspondences
     ↓
RANSAC
     ↓
Homography Estimation
     ↓
Image Warping
     ↓
Image Alignment
     ↓
Panorama Stitching
     ↓
Quantitative Evaluation
     ↓
Visual Results
```

This pipeline is the central focus of the project.

---

# 2. DO NOT CHEAT THE COMPUTER VISION REQUIREMENT

Do NOT simply call:

```python
cv2.Stitcher_create()
```

as the primary solution.

Do NOT build the project around pretrained recognition models.

Do NOT hide the feature matching/homography pipeline behind a high-level API.

The examiner must be able to inspect the source code and clearly see:

* feature detection
* descriptor computation
* descriptor matching
* correspondence filtering
* RANSAC
* homography estimation
* image transformation
* image blending/stitching
* evaluation

OpenCV is encouraged, but the important computer vision concepts must remain visible.

---

# 3. PROJECT ARCHITECTURE

Create a professional project structure similar to:

```text
.
├── ai/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── transformed/
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── matching.py
│   ├── homography.py
│   ├── warping.py
│   ├── stitching.py
│   ├── evaluation.py
│   ├── visualization.py
│   └── pipeline.py
│
├── experiments/
│   ├── run_baseline.py
│   ├── run_rotation.py
│   ├── run_scale.py
│   ├── run_viewpoint.py
│   ├── run_illumination.py
│   └── run_all.py
│
├── scripts/
│   ├── prepare_dataset.py
│   ├── run_pipeline.py
│   └── generate_report_tables.py
│
├── notebooks/
│   └── analysis.ipynb
│
├── results/
│   ├── keypoints/
│   ├── raw_matches/
│   ├── ransac_matches/
│   ├── homographies/
│   ├── warped/
│   ├── panoramas/
│   ├── plots/
│   ├── tables/
│   └── logs/
│
├── tests/
│   ├── test_features.py
│   ├── test_matching.py
│   ├── test_homography.py
│   ├── test_warping.py
│   └── test_pipeline.py
│
└── report/
    ├── figures/
    ├── tables/
    └── report.md
```

You may improve this structure if there is a technically better organization, but preserve clear separation of concerns.

---

# 4. DATASET / IMAGE ACQUISITION

The examination explicitly requires:

> Acquire at least three overlapping images of the same scene from different viewpoints.

Design the project so the user can place three or more images into:

```text
data/raw/
```

The README must explain exactly how to acquire suitable images.

Images should:

* represent the same physical scene
* overlap significantly
* contain distinctive visual features
* be captured from slightly different viewpoints
* avoid excessive motion
* avoid extremely repetitive textures where possible

If suitable sample images are available legally, you may include or download them only when appropriate.

Do NOT silently fabricate a dataset.

The pipeline must support arbitrary user-provided images.

If sample images are generated programmatically for controlled experiments, clearly label them as transformed experimental images rather than real-world acquisitions.

---

# 5. IMAGE PREPROCESSING

Implement necessary preprocessing.

Potential operations may include:

* image validation
* resizing while preserving aspect ratio
* grayscale conversion for feature detection
* optional contrast normalization
* optional denoising
* handling image dimensions

Do not preprocess unnecessarily.

The project must explain why each preprocessing operation is used.

Save representative preprocessing visualizations.

---

# 6. FEATURE ALGORITHM COMPARISON

The examination requires comparison of at least TWO feature detection/description approaches studied in the course.

Implement at minimum:

## Method 1: SIFT

Use OpenCV's SIFT implementation where available.

Record:

* keypoints
* descriptors
* descriptor dimension
* extraction time

## Method 2: ORB

Use OpenCV ORB.

Record:

* keypoints
* descriptors
* descriptor dimension
* extraction time

The architecture must make it easy to add another detector/descriptor later.

Do not hard-code the entire pipeline specifically around SIFT.

Create a common interface such as:

```text
FeatureDetector
FeatureExtractor
```

or an equivalent clean abstraction.

---

# 7. FEATURE DETECTION

For every image and every algorithm:

Detect distinctive keypoints.

Generate visualizations showing:

* original image
* detected keypoints
* keypoint distribution

Record:

```text
image
algorithm
number_of_keypoints
processing_time
```

Save results.

---

# 8. FEATURE DESCRIPTION

Compute descriptors associated with detected keypoints.

Document the differences between:

* SIFT descriptors
* ORB descriptors

Ensure matching uses the correct distance metric for the descriptor type.

For example, do not blindly use the same matcher configuration for floating-point and binary descriptors.

---

# 9. FEATURE MATCHING

Implement descriptor matching between overlapping image pairs.

Support appropriate strategies such as:

### SIFT

Use an appropriate floating-point descriptor matcher, such as:

* BFMatcher with L2 distance
* KNN matching
* Lowe-style ratio filtering

### ORB

Use an appropriate binary descriptor matcher, such as:

* BFMatcher with Hamming distance
* KNN matching
* ratio filtering where appropriate

Record:

* number of descriptors
* raw matches
* filtered/good matches
* matching time

---

# 10. DISPLAY INITIAL FEATURE CORRESPONDENCES

This is explicitly required.

Before RANSAC, generate images showing initial feature correspondences.

The visualization must clearly distinguish:

```text
Image A                    Image B
   ● ------------------------ ●
   ● -------------------- ●
   ● ----------- ●
```

Use OpenCV drawing functions appropriately.

Save:

```text
results/raw_matches/
```

These images must be included in the report.

---

# 11. RANSAC

Implement robust homography estimation using RANSAC.

Use:

```python
cv2.findHomography(
    src_points,
    dst_points,
    cv2.RANSAC,
    reprojection_threshold
)
```

or an equivalent OpenCV implementation.

Clearly document:

* why RANSAC is required
* what an inlier is
* what an outlier is
* reprojection error
* threshold selection
* confidence
* iteration behavior

Record:

```text
initial_matches
ransac_inliers
inlier_ratio
```

Calculate:

```text
inlier_ratio = ransac_inliers / initial_matches
```

Handle cases where homography estimation fails.

Never allow the pipeline to crash silently.

---

# 12. BEFORE VS AFTER RANSAC

Create direct visual comparisons:

### Before RANSAC

Show raw/filtered descriptor correspondences.

### After RANSAC

Show only geometrically consistent inliers.

Create side-by-side comparison figures.

Also create quantitative comparison tables.

At minimum:

| Algorithm | Pair | Initial Matches | RANSAC Inliers | Inlier Ratio |
| --------- | ---: | --------------: | -------------: | -----------: |

---

# 13. HOMOGRAPHY

Estimate the 3×3 homography matrix between corresponding images.

Store every successful matrix.

The implementation must explain:

```text
x' ~ Hx
```

and why homogeneous coordinates are used.

Save matrices as:

* JSON
* CSV
* NumPy files

where appropriate.

Include homography diagnostics.

If the homography is degenerate or unreliable, detect and report it rather than blindly using it.

---

# 14. IMAGE WARPING

Use the estimated homography to transform one image into the coordinate system of another.

Implement the geometry explicitly using OpenCV warping operations.

Account for:

* transformed corners
* canvas size
* image translation
* negative coordinates
* overlap
* valid masks

Do not simply crop the result arbitrarily.

The resulting warped images must be saved.

---

# 15. PANORAMA CONSTRUCTION

Construct the panorama from at least three images.

The pipeline should support:

```text
Image 1 + Image 2
       ↓
Intermediate panorama
       ↓
+ Image 3
       ↓
Final panorama
```

If technically preferable, use a reference/central image and transform other images into a common coordinate system.

Implement sensible image compositing/blending.

At minimum address:

* overlap
* seams
* canvas boundaries
* black borders
* duplicate pixels
* exposure differences where possible

Avoid unnecessarily complex blending algorithms unless they improve the result meaningfully.

---

# 16. PANORAMA QUALITY

The examination asks for:

> Quality of the final panorama.

Do not rely only on subjective statements.

Implement quantitative metrics where meaningful.

Possible metrics include:

* structural similarity where a valid reference exists
* seam/discontinuity measurements
* reprojection error
* overlap consistency
* valid panorama area
* crop/border ratio

Clearly distinguish metrics that are objectively measurable from qualitative visual inspection.

Do not invent ground truth where none exists.

Also create a qualitative evaluation framework addressing:

* alignment
* visible seams
* ghosting
* distortion
* missing regions
* sharpness
* completeness

---

# 17. REQUIRED TRANSFORMATION EXPERIMENTS

Create controlled experiments for:

## A. Rotation

Investigate matching when one image is rotated.

Use multiple reasonable rotation levels.

Measure:

* keypoints
* matches
* RANSAC inliers
* inlier ratio
* processing time
* panorama quality

Compare SIFT vs ORB.

---

## B. SCALE CHANGES

Investigate scale changes.

Create controlled scale variations.

Evaluate both algorithms.

Report all required metrics.

---

## C. VIEWPOINT CHANGES

Investigate changes in viewpoint.

Use real images from different viewpoints where possible.

If controlled synthetic perspective transformations are used, clearly label them as synthetic viewpoint experiments.

Evaluate robustness.

---

## D. ILLUMINATION CHANGES

Investigate changes in brightness/contrast/illumination.

Use controlled image transformations or suitable real images.

Evaluate matching and panorama construction.

---

# 18. EXPERIMENT MATRIX

Create an automated experiment runner.

At minimum:

```text
Baseline
Rotation
Scale
Viewpoint
Illumination
```

For each:

```text
SIFT
ORB
```

For every experiment record:

```text
experiment
algorithm
image_pair
keypoints_image_1
keypoints_image_2
initial_matches
good_matches
ransac_inliers
inlier_ratio
feature_extraction_time
matching_time
homography_time
warping_time
total_time
panorama_width
panorama_height
panorama_quality_metrics
status
failure_reason
```

Store results in CSV.

Also generate JSON for machine-readable experiment output.

---

# 19. PROCESSING TIME

The examination explicitly requires processing time.

Use Python timing utilities appropriately.

Measure meaningful stages separately:

```text
feature detection/description
matching
homography/RANSAC
warping
stitching
total pipeline
```

Do not report made-up timing values.

Run actual experiments.

Clearly state hardware/software environment.

---

# 20. RESULTS TABLES

Automatically generate tables such as:

### Feature Comparison

| Method | Keypoints | Initial Matches | RANSAC Inliers | Inlier Ratio | Processing Time | Panorama Quality |
| ------ | --------: | --------------: | -------------: | -----------: | --------------: | ---------------: |

### Transformation Robustness

| Transformation | Method | Matches | Inliers | Inlier Ratio | Time | Result |
| -------------- | ------ | ------: | ------: | -----------: | ---: | ------ |

### Before/After RANSAC

| Method | Initial Matches | Inliers | Rejected | Inlier Ratio |
| ------ | --------------: | ------: | -------: | -----------: |

Generate these automatically from experiment outputs.

---

# 21. VISUALIZATIONS

Automatically generate professional figures for the report.

At minimum:

1. original images
2. preprocessed images
3. SIFT keypoints
4. ORB keypoints
5. raw matches
6. RANSAC inlier matches
7. before/after RANSAC
8. warped images
9. intermediate panorama
10. final panorama
11. rotation comparison
12. scale comparison
13. viewpoint comparison
14. illumination comparison
15. algorithm performance plots
16. inlier-ratio comparison
17. processing-time comparison
18. failure examples

Use Matplotlib/OpenCV.

Make figures publication/report friendly.

Include:

* titles
* axis labels where appropriate
* legends
* captions
* readable dimensions
* consistent naming

---

# 22. FAILURE CASES

The system must deliberately detect and document failure cases.

Examples:

* insufficient overlap
* too few keypoints
* too few good matches
* insufficient RANSAC inliers
* homography failure
* degenerate homography
* severe viewpoint change
* repetitive textures
* large illumination differences
* motion blur
* excessive scale differences

Do not hide failed experiments.

Create:

```text
results/logs/failures.csv
```

with:

```text
experiment
algorithm
image_pair
failure_stage
failure_reason
```

Generate representative failure visualizations.

---

# 23. UNIT TESTING

Create tests for important components.

Test:

* image loading
* feature extraction
* matching
* insufficient matches
* homography estimation
* warping
* pipeline failure handling

Tests should not depend entirely on large external datasets.

Use small synthetic fixtures where appropriate.

---

# 24. COMMAND-LINE INTERFACE

Provide an easy CLI.

Examples:

```bash
python scripts/run_pipeline.py --algorithm sift
```

```bash
python scripts/run_pipeline.py --algorithm orb
```

```bash
python experiments/run_all.py
```

Provide useful options for:

* input directory
* output directory
* algorithm
* ratio threshold
* RANSAC threshold
* resize factor
* experiment selection

---

# 25. CONFIGURATION

Centralize configurable values.

Examples:

```text
feature algorithm
SIFT parameters
ORB parameters
ratio threshold
RANSAC reprojection threshold
RANSAC confidence
image resize
random seed
```

Do not scatter magic numbers throughout the code.

---

# 26. LOGGING

Use Python logging.

The system should clearly communicate:

* image loading
* keypoint counts
* match counts
* RANSAC results
* homography status
* panorama status
* experiment status
* failures

---

# 27. REPRODUCIBILITY

Create:

```text
requirements.txt
```

and ensure the project can be installed with:

```bash
pip install -r requirements.txt
```

Document Python version.

Record:

* OpenCV version
* NumPy version
* Matplotlib version
* operating system
* CPU/GPU information where relevant

Do not require a GPU unless absolutely necessary.

This project should run on a normal computer.

---

# 28. README

Write a comprehensive root README containing:

## Project title

## Problem statement

## Objectives

## Examination requirements

## Computer vision methodology

## Algorithms

Explain:

* SIFT
* ORB
* descriptor matching
* RANSAC
* homography
* image warping
* panorama construction

## Installation

## Dataset preparation

## Running the baseline

## Running experiments

## Generating results

## Testing

## Output structure

## Results interpretation

## Failure handling

## Limitations

## Reproducibility

## Academic note

Make clear that experimental values are generated by actually running the pipeline.

---

# 29. ACADEMIC REPORT

Create a detailed report template in:

```text
report/report.md
```

The report must contain:

# Abstract

# 1. Introduction

# 2. Problem Definition

# 3. Objectives

# 4. Dataset and Image Acquisition

# 5. Image Preparation

# 6. Methodology

## 6.1 Feature Detection

## 6.2 Feature Description

## 6.3 Feature Matching

## 6.4 RANSAC

## 6.5 Homography Estimation

## 6.6 Image Warping

## 6.7 Panorama Construction

# 7. Experimental Design

# 8. Results

# 9. SIFT vs ORB Comparison

# 10. Rotation Experiment

# 11. Scale Experiment

# 12. Viewpoint Experiment

# 13. Illumination Experiment

# 14. Before vs After RANSAC

# 15. Failure Cases

# 16. Limitations

# 17. Discussion

# 18. Conclusion

# 19. Future Improvements

# References

Do NOT fabricate results.

Use placeholders until actual experiments have been executed.

Once experiments are run, automatically insert actual measured values where practical.

---

# 30. REQUIREMENT TRACEABILITY TABLE

Create a final table mapping:

| Examination Requirement | Implementation | Evidence | Result |
| ----------------------- | -------------- | -------- | ------ |

Every one of the 12 examination tasks must be covered.

Also map the experimental requirements.

---

# 31. FINAL VALIDATION

Before declaring the project complete, run:

```text
unit tests
baseline experiment
SIFT experiment
ORB experiment
rotation experiment
scale experiment
viewpoint experiment
illumination experiment
result generation
visualization generation
```

Check that:

* files exist
* output images can be opened
* CSV files contain actual measurements
* figures are generated
* panoramas are generated
* failure cases are recorded
* README commands work
* tests pass

---

# 32. QUALITY CONTROL

Do not stop after writing code.

Actually execute the system.

If errors occur:

1. diagnose them,
2. fix them,
3. rerun the affected experiment,
4. verify outputs.

Do not claim success merely because code compiles.

Do not fabricate experimental values.

---

# 33. IMPORTANT PRACTICAL REQUIREMENT

If the repository does not contain appropriate input images, do NOT pretend the requirement has been fulfilled.

Instead:

1. create the expected `data/raw/` structure,
2. provide clear instructions for adding the required images,
3. create controlled synthetic transformation utilities for rotation, scale, viewpoint, and illumination experiments,
4. make the pipeline ready to run immediately once valid images are supplied.

If publicly available sample images can be legally retrieved and included, document their source and licensing appropriately.

---

# 34. FINAL DELIVERABLE

The final repository must look like a serious postgraduate computer vision research/engineering project rather than a simple classroom script.

It must demonstrate:

```text
Understanding
     ↓
Implementation
     ↓
Experimentation
     ↓
Measurement
     ↓
Visualization
     ↓
Analysis
     ↓
Conclusion
```

The most important principle is:

**The examiner should be able to follow exactly how two or more images go from raw pixels to feature correspondences, through RANSAC and homography estimation, into an aligned multi-image panorama, and then see quantitative evidence comparing SIFT and ORB under different transformations.**

Do not over-engineer irrelevant features.

Prioritize correctness, explainability, experimental evidence, and completeness.

---

# FINAL RESPONSE AFTER IMPLEMENTATION

When finished, provide:

1. project structure
2. files created
3. algorithms implemented
4. experiments implemented
5. tests executed and their status
6. actual experimental results generated
7. visualizations generated
8. panorama outputs generated
9. any remaining manual steps
10. any limitations encountered

Do not claim that an experiment was completed unless it was actually executed.

Start by inspecting `/ai` and the repository, then implement the project end-to-end.

