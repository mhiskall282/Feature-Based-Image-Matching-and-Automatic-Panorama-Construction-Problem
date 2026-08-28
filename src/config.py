"""
src/config.py
=============
Centralised configuration for the panorama pipeline.

All tunable parameters are defined here. Experiment scripts override specific
values using dict unpacking: {**DEFAULT_CONFIG, 'key': new_value}.

# RULE-PY09: All tunable parameters come from config — no magic numbers in code.
"""

import cv2

# ─────────────────────────────────────────────
# Random seed
# ─────────────────────────────────────────────
RANDOM_SEED = 42

# ─────────────────────────────────────────────
# Image loading & preprocessing
# ─────────────────────────────────────────────
MAX_IMAGE_DIMENSION = 1280          # Resize images if larger (preserves aspect ratio)
RESIZE_INTERPOLATION = cv2.INTER_LINEAR

# ─────────────────────────────────────────────
# SIFT parameters
# REQ-03, REQ-04: Feature detection and description
# ─────────────────────────────────────────────
SIFT_PARAMS = {
    "nfeatures":          0,     # 0 = unlimited keypoints
    "nOctaveLayers":      3,     # Layers per DoG octave (default 3)
    "contrastThreshold":  0.04,  # Filter low-contrast keypoints
    "edgeThreshold":      10,    # Filter edge-like responses
    "sigma":              1.6,   # Initial Gaussian blur
}

# ─────────────────────────────────────────────
# ORB parameters
# REQ-03, REQ-04: Feature detection and description
# ─────────────────────────────────────────────
ORB_PARAMS = {
    "nfeatures":      1000,   # Max keypoints to detect
    "scaleFactor":    1.2,    # Pyramid scale factor
    "nlevels":        8,      # Number of pyramid levels
    "edgeThreshold":  31,     # Border excluded from detection
    "patchSize":      31,     # Patch size for rBRIEF descriptor
    "fastThreshold":  20,     # FAST corner threshold
}

# ─────────────────────────────────────────────
# Matching parameters
# REQ-05: Descriptor matching
# ─────────────────────────────────────────────
RATIO_THRESHOLD   = 0.75   # Lowe's ratio test threshold for SIFT
CROSS_CHECK       = True   # Cross-check for ORB BFMatcher

# ─────────────────────────────────────────────
# RANSAC parameters
# REQ-07, REQ-08: RANSAC and homography estimation
# ─────────────────────────────────────────────
RANSAC_REPROJ_THRESHOLD = 5.0    # Reprojection error threshold (pixels)
RANSAC_CONFIDENCE       = 0.995  # Desired probability of correct result
RANSAC_MAX_ITERS        = 2000   # Maximum RANSAC iterations
MIN_RANSAC_INLIERS      = 10     # Minimum inliers to accept homography

# ─────────────────────────────────────────────
# Warping / Stitching
# REQ-09, REQ-10: Image warping and panorama construction
# ─────────────────────────────────────────────
WARP_INTERPOLATION = cv2.INTER_LINEAR
BLEND_ALPHA        = True    # Use alpha blending in overlap region

# ─────────────────────────────────────────────
# Experiment transformation ranges
# REQ-12: Robustness experiments
# ─────────────────────────────────────────────
ROTATION_ANGLES   = [0, 15, 30, 45, 60, 90, 120, 180]   # degrees
SCALE_FACTORS     = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
BRIGHTNESS_DELTAS = [-100, -50, 50, 100]                   # cv2.convertScaleAbs beta
CONTRAST_FACTORS  = [0.5, 0.75, 1.25, 1.5]                # cv2.convertScaleAbs alpha

# ─────────────────────────────────────────────
# Output directories
# ─────────────────────────────────────────────
RESULTS_DIR   = "results"
OUTPUTS_DIR   = "outputs"
DATA_RAW_DIR  = "data/raw"
DATA_PROC_DIR = "data/processed"
DATA_XFRM_DIR = "data/transformed"

# ─────────────────────────────────────────────
# Supported methods
# ─────────────────────────────────────────────
SUPPORTED_METHODS = ["SIFT", "ORB"]

# ─────────────────────────────────────────────
# Norm type per method (for descriptor matching)
# REQ-05, RULE-PY06: Correct distance metric per descriptor type
# ─────────────────────────────────────────────
METHOD_NORMS = {
    "SIFT": cv2.NORM_L2,
    "ORB":  cv2.NORM_HAMMING,
}

# ─────────────────────────────────────────────
# Descriptor type label (for documentation)
# ─────────────────────────────────────────────
DESCRIPTOR_TYPES = {
    "SIFT": "float32 (128-dim L2)",
    "ORB":  "binary uint8 (256-bit Hamming)",
}
