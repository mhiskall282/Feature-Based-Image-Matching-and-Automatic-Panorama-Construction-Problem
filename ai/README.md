# AI Workspace — CSCD608 Feature-Based Image Matching & Panorama Construction

## What This Directory Is

This `/ai` directory is the **internal engineering and research specification** for the CSCD608 Advanced Computer Vision examination project. It is NOT source code. It is a structured set of instructions, context documents, skills, and rules that govern how any future agent (or human implementer) should approach, implement, evaluate, and report on this project.

This workspace was designed for:
- **Rigorous AI-assisted implementation** — every decision is traceable to an examination requirement
- **Academic integrity** — no fabricated results, no plagiarism, no black-box shortcuts
- **Reproducible experiments** — deterministic pipelines, recorded seeds, documented configurations
- **MPhil/MSc-level quality** — methodology, justification, quantitative evaluation, failure analysis

---

## How Future Agents Must Use This Workspace

Before writing any code or running any experiment, a future agent MUST:

1. **Read this README** completely.
2. **Read `/ai/context/project-overview.md`** to understand the full project scope.
3. **Read `/ai/context/examination-requirements.md`** for the complete requirement traceability matrix.
4. **Read all applicable `/ai/skills/*.md`** files relevant to the task at hand.
5. **Obey all rules in `/ai/rules/*.md`** — these are binding constraints, not suggestions.
6. **Never modify the `/ai` directory** unless the project specification itself needs to change (and only with documented justification).

### Agent Workflow Protocol

```
START
  │
  ├── 1. Read /ai/README.md                        ← You are here
  ├── 2. Read /ai/context/project-overview.md
  ├── 3. Read /ai/context/examination-requirements.md
  ├── 4. Identify task (feature detection? stitching? evaluation?)
  ├── 5. Read relevant /ai/skills/*.md files
  ├── 6. Check /ai/rules/*.md for all binding constraints
  ├── 7. Plan implementation with requirement traceability
  ├── 8. Implement modularly (see /ai/rules/general.md)
  ├── 9. Run experiments (see /ai/context/evaluation-framework.md)
  ├── 10. Record results (CSV/JSON — see /ai/rules/reproducibility.md)
  ├── 11. Generate visualizations (see /ai/skills/visualization.md)
  ├── 12. Analyze failures (see /ai/skills/failure-analysis.md)
  └── 13. Support report writing (see /ai/context/expected-deliverables.md)
```

---

## Project Objective

Implement a **complete, classical feature-based image matching and automatic panorama construction pipeline** using Python and OpenCV, demonstrating mastery of the following stages:

```
Image Acquisition
      ↓
Image Preprocessing
      ↓
Feature Detection  (SIFT, ORB, or equivalent classical detector)
      ↓
Feature Description  (descriptor computation)
      ↓
Feature Matching  (BFMatcher, FLANN, ratio test)
      ↓
RANSAC  (outlier rejection)
      ↓
Homography Estimation  (perspective transform matrix H)
      ↓
Image Warping  (cv2.warpPerspective)
      ↓
Image Alignment & Blending
      ↓
Panorama Construction  (multi-image stitching)
      ↓
Quantitative Evaluation  (metrics, comparison table)
      ↓
Academic Report
```

**Every stage must be explicitly implemented and visualized.** High-level wrappers that hide the pipeline (e.g., `cv2.Stitcher_create()` used as a black box) are forbidden unless the internals are also explicitly demonstrated.

---

## Examination Context

- **Course**: CSCD608 Advanced Computer Vision (3 Credits)
- **Programme**: MPhil/MSc Computer Science
- **Semester**: Second Semester Examinations 2025/2026
- **Question**: 1 — Feature-Based Image Matching and Automatic Panorama Construction
- **Nature**: Practical implementation + experimental comparison + academic report

---

## Implementation Philosophy

1. **Classical computer vision first.** The core pipeline must use classical methods — SIFT, ORB, Harris, FAST, BRIEF, etc. — as studied in CSCD608. Deep learning feature extractors are not appropriate for this examination.

2. **No black boxes.** The implementation must be explainable at an oral examination. Every homography matrix, every RANSAC threshold, every distance metric must be justified.

3. **Comparison is mandatory.** At least two feature detection/description approaches must be compared on the same dataset with identical experimental conditions.

4. **Honesty about failure.** When an algorithm fails (e.g., poor matching under extreme viewpoint change), this must be documented, visualized, and discussed — not hidden.

5. **Reproducibility is non-negotiable.** All experiments must use fixed random seeds, recorded configurations, and deterministic outputs wherever possible.

---

## Experiment Philosophy

- Start from a **baseline** (three overlapping images, standard conditions).
- Then systematically **stress-test** with controlled transformations: rotation, scale, viewpoint, illumination.
- Each experiment must be **independently runnable** from the command line with documented parameters.
- Results must be **saved** (CSV, JSON, PNG) — never computed in an ad hoc manner.

---

## Evaluation Philosophy

- Metrics must be **measured** from the actual output of the pipeline — never estimated or fabricated.
- A clear distinction must be maintained between:
  - **Measured values** (keypoint count, inlier count, time)
  - **Derived metrics** (inlier ratio, match rate)
  - **Qualitative observations** (visual quality, artifact assessment)
- Ground truth for homography can be computed analytically for synthetic transformations (rotation, scale). For real images, visual inspection + RANSAC inlier ratio serves as a proxy.

---

## Expected Workflow (Implementation Phase)

When the implementation phase begins:

1. **Set up the project structure** (see `/ai/context/expected-deliverables.md` for directory layout).
2. **Acquire/prepare the dataset** (see `/ai/context/dataset-and-image-requirements.md`).
3. **Implement core modules** one at a time, in pipeline order.
4. **Run the baseline experiment** first.
5. **Run stress experiments** after baseline is confirmed working.
6. **Generate all visualizations** as specified in `/ai/skills/visualization.md`.
7. **Compile the results table** and comparison analysis.
8. **Write the academic report** supported by the context in `/ai/context/`.

---

## Directory Map

```
/ai
├── README.md                        ← This file. Read first.
│
├── context/
│   ├── project-overview.md          ← Full project scope and background
│   ├── examination-requirements.md  ← Requirement traceability matrix
│   ├── computer-vision-concepts.md  ← Core CV concepts required for this project
│   ├── dataset-and-image-requirements.md  ← Image acquisition and prep guidelines
│   ├── evaluation-framework.md      ← Quantitative metrics and evaluation design
│   └── expected-deliverables.md     ← Final output structure and report outline
│
├── skills/
│   ├── feature-detection.md         ← SIFT, ORB, Harris — how to implement & compare
│   ├── feature-description.md       ← Descriptor computation and properties
│   ├── feature-matching.md          ← BFMatcher, FLANN, ratio test, cross-check
│   ├── ransac-homography.md         ← RANSAC algorithm, homography estimation
│   ├── image-warping.md             ← Perspective warping, coordinate transforms
│   ├── panorama-stitching.md        ← Multi-image alignment, blending strategies
│   ├── experimental-evaluation.md   ← How to design and run the experiments
│   ├── visualization.md             ← Output specification, filenames, formats
│   ├── failure-analysis.md          ← How to identify, document, and discuss failures
│   └── academic-reporting.md        ← Report structure and academic writing guidance
│
└── rules/
    ├── general.md                   ← Binding general rules for the project
    ├── python-opencv.md             ← Python/OpenCV coding standards
    ├── research.md                  ← Academic integrity and research conduct
    ├── experiments.md               ← Experimental design rules
    ├── evaluation.md                ← Evaluation rules (no fabrication)
    ├── visualization.md             ← Visualization output rules
    └── reproducibility.md           ← Reproducibility requirements
```

---

## Critical Prohibitions (Summary)

- ❌ Do NOT use `cv2.Stitcher_create()` as a black box panorama solution
- ❌ Do NOT use pretrained deep learning models for feature extraction
- ❌ Do NOT fabricate keypoint counts, match ratios, or processing times
- ❌ Do NOT fabricate citations or reference papers you have not read
- ❌ Do NOT claim experimental results you have not measured
- ❌ Do NOT put the entire system in a single Python file
- ❌ Do NOT skip the RANSAC step and call results "matched correspondences"
- ❌ Do NOT use placeholder images or synthetic dummy data as the primary dataset

---

*This workspace is version 1.0 — initialized for the CSCD608 2025/2026 examination project.*
*Last updated: 2026-08-28*
