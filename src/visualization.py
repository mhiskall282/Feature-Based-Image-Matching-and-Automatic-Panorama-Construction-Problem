"""
src/visualization.py
====================
All visualization functions for the pipeline.

Generates publication-quality figures for every pipeline stage.
Output filenames and conventions follow /ai/rules/visualization.md.

RULE-VIZ conventions:
  - Every figure has a descriptive title
  - BGR → RGB before any matplotlib imshow
  - plt.close() after every savefig (memory management)
  - 150 dpi minimum
  - Fixed colour scheme: SIFT=blue, ORB=orange
"""

import logging
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend (safe in scripts and Colab)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

logger = logging.getLogger(__name__)

# ── Colour conventions (RULE-VIZ05) ─────────────────────────
COLOURS = {
    "SIFT": "#2196F3",       # Blue
    "ORB":  "#FF9800",       # Orange
    "raw_match":   (0,   0, 255),   # Red  (BGR for cv2 drawing)
    "inlier":      (0, 200,   0),   # Green
    "outlier":     (0,   0, 255),   # Red
}

MAX_DISPLAY_MATCHES = 150


def _save(fig: plt.Figure, path: str | Path, dpi: int = 150) -> None:
    """Save figure, ensure directory exists, then close."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.debug(f"Saved figure: {path}")


def _bgr2rgb(img: np.ndarray) -> np.ndarray:
    """Convert BGR to RGB for matplotlib display (RULE-VIZ04)."""
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# ─────────────────────────────────────────────────────────────
# VIZ-01: Input images
# ─────────────────────────────────────────────────────────────

def save_input_images(images: list, names: list, out_path: str | Path) -> None:
    """Show all input images side by side."""
    n   = len(images)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5))
    if n == 1:
        axes = [axes]
    for ax, img, name in zip(axes, images, names):
        ax.imshow(_bgr2rgb(img))
        ax.set_title(f"{name}\n{img.shape[1]}×{img.shape[0]} px", fontsize=11)
        ax.axis("off")
    fig.suptitle("Input Images", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, out_path)


# ─────────────────────────────────────────────────────────────
# VIZ-02: Preprocessed images
# ─────────────────────────────────────────────────────────────

def save_preprocessed(img_color: np.ndarray, img_gray: np.ndarray,
                      name: str, out_path: str | Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(_bgr2rgb(img_color)); axes[0].set_title(f"{name} — Colour"); axes[0].axis("off")
    axes[1].imshow(img_gray, cmap="gray"); axes[1].set_title(f"{name} — Grayscale (for detection)"); axes[1].axis("off")
    fig.suptitle("Preprocessed Images", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, out_path)


# ─────────────────────────────────────────────────────────────
# VIZ-03: Detected keypoints
# REQ-03 visualization
# ─────────────────────────────────────────────────────────────

def save_keypoints(img_gray: np.ndarray, keypoints: list,
                   method: str, img_name: str,
                   out_path: str | Path, max_kp: int = 500) -> None:
    """Draw rich keypoints (scale + orientation circles) on image."""
    display_kp = keypoints[:max_kp]
    img_kp = cv2.drawKeypoints(
        img_gray, display_kp, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    colour = COLOURS.get(method.upper(), "#888888")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(_bgr2rgb(img_kp))
    ax.set_title(
        f"{method} Keypoints — {img_name}\n"
        f"{len(keypoints)} detected (showing {len(display_kp)})",
        fontsize=12, color=colour, fontweight="bold"
    )
    ax.axis("off")
    plt.tight_layout()
    _save(fig, out_path)


# ─────────────────────────────────────────────────────────────
# VIZ-05: Raw matches (initial correspondences)
# REQ-06: Display initial feature correspondences
# ─────────────────────────────────────────────────────────────

def save_raw_matches(img1_gray: np.ndarray, kp1: list,
                     img2_gray: np.ndarray, kp2: list,
                     matches: list, method: str,
                     pair_name: str, out_path: str | Path) -> None:
    """
    Visualise initial feature correspondences (before RANSAC).
    # REQ-06: Display initial feature correspondences
    """
    display = matches[:MAX_DISPLAY_MATCHES]
    img_m   = cv2.drawMatches(
        img1_gray, kp1, img2_gray, kp2, display, None,
        matchColor=COLOURS["raw_match"],
        singlePointColor=(180, 180, 180),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.imshow(_bgr2rgb(img_m))
    ax.set_title(
        f"{method} — Initial Correspondences: {pair_name}\n"
        f"{len(matches)} matches (showing {len(display)})",
        fontsize=12
    )
    ax.axis("off")
    plt.tight_layout()
    _save(fig, out_path)


# ─────────────────────────────────────────────────────────────
# VIZ-06: Before vs After RANSAC
# REQ-11: Compare feature matching before and after RANSAC
# ─────────────────────────────────────────────────────────────

def save_before_after_ransac(img1_gray: np.ndarray, kp1: list,
                              img2_gray: np.ndarray, kp2: list,
                              all_matches: list,
                              inlier_matches: list,
                              method: str, pair_name: str,
                              out_path: str | Path) -> None:
    """
    Side-by-side: all matches (red) vs RANSAC inliers (green).
    # REQ-11: Compare feature matching before and after RANSAC
    """
    fig, axes = plt.subplots(1, 2, figsize=(22, 7))

    before_img = cv2.drawMatches(
        img1_gray, kp1, img2_gray, kp2,
        all_matches[:MAX_DISPLAY_MATCHES], None,
        matchColor=COLOURS["raw_match"],
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    axes[0].imshow(_bgr2rgb(before_img))
    axes[0].set_title(f"Before RANSAC: {len(all_matches)} matches", fontsize=12)
    axes[0].axis("off")

    after_img = cv2.drawMatches(
        img1_gray, kp1, img2_gray, kp2,
        inlier_matches, None,
        matchColor=COLOURS["inlier"],
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    ratio = len(inlier_matches) / len(all_matches) if all_matches else 0.0
    axes[1].imshow(_bgr2rgb(after_img))
    axes[1].set_title(
        f"After RANSAC: {len(inlier_matches)} inliers ({ratio:.1%})", fontsize=12
    )
    axes[1].axis("off")

    fig.suptitle(f"{method} — Before vs After RANSAC: {pair_name}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, out_path)


# ─────────────────────────────────────────────────────────────
# VIZ-07: Warped image
# ─────────────────────────────────────────────────────────────

def save_warped(img_ref: np.ndarray, img_warped: np.ndarray,
                method: str, out_path: str | Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].imshow(_bgr2rgb(img_ref));    axes[0].set_title("Reference Image");             axes[0].axis("off")
    axes[1].imshow(_bgr2rgb(img_warped)); axes[1].set_title(f"Warped Source ({method})"); axes[1].axis("off")
    fig.suptitle("Image Warping Result", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, out_path)


# ─────────────────────────────────────────────────────────────
# VIZ-08: Final panorama
# ─────────────────────────────────────────────────────────────

def save_panorama(panorama: np.ndarray, method: str,
                  experiment: str, out_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(20, 7))
    ax.imshow(_bgr2rgb(panorama))
    ax.set_title(
        f"Final Panorama — {method}  |  Experiment: {experiment}\n"
        f"Dimensions: {panorama.shape[1]}×{panorama.shape[0]} px",
        fontsize=12
    )
    ax.axis("off")
    plt.tight_layout()
    _save(fig, out_path)


# ─────────────────────────────────────────────────────────────
# VIZ-09: Method comparison bar chart
# ─────────────────────────────────────────────────────────────

def save_comparison_bars(df, metrics: list, title: str,
                         out_path: str | Path) -> None:
    """Grouped bar chart comparing SIFT vs ORB across key metrics."""
    methods = df["method"].unique()
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 6))
    if n == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        for j, method in enumerate(methods):
            vals = df[df["method"] == method][metric].dropna()
            colour = COLOURS.get(method.upper(), "#888888")
            ax.bar(j, vals.mean(), color=colour, label=method, alpha=0.85, edgecolor="black")
        ax.set_title(metric.replace("_", "\n"), fontsize=9)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, fontsize=10)
        ax.set_ylabel(metric, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    handles = [plt.Rectangle((0,0),1,1, color=COLOURS.get(m.upper(),"#888")) for m in methods]
    fig.legend(handles, methods, loc="upper right", fontsize=10)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, out_path)


# ─────────────────────────────────────────────────────────────
# VIZ-10: Experiment line plots
# ─────────────────────────────────────────────────────────────

def save_experiment_trend(df, x_col: str, y_col: str,
                          title: str, xlabel: str, ylabel: str,
                          out_path: str | Path) -> None:
    """Line plot: y_col vs x_col, one line per method."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for method in df["method"].unique():
        sub = df[df["method"] == method].sort_values(x_col)
        colour = COLOURS.get(method.upper(), "#888")
        ax.plot(sub[x_col], sub[y_col], marker="o", label=method,
                color=colour, linewidth=2, markersize=7)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, out_path)


# ─────────────────────────────────────────────────────────────
# Failure visualization
# ─────────────────────────────────────────────────────────────

def save_failure_case(img1: np.ndarray, img2: np.ndarray,
                      method: str, failure_code: str,
                      details: str, out_path: str | Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(_bgr2rgb(img1)); axes[0].set_title("Image 1"); axes[0].axis("off")
    axes[1].imshow(_bgr2rgb(img2)); axes[1].set_title("Image 2"); axes[1].axis("off")
    fig.suptitle(
        f"FAILURE CASE — {method}\nCode: {failure_code}\nDetails: {details}",
        fontsize=11, color="red", fontweight="bold"
    )
    plt.tight_layout()
    _save(fig, out_path)
