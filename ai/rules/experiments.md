# Rules: Experiment Design

## Status
These rules govern how experiments are designed, configured, executed, and recorded. They are **mandatory** for all experiments in `experiments/`.

---

## RULE-E01: One Variable Per Experiment

Each stress experiment (rotation, scale, viewpoint, illumination) must change **only one variable** while holding all others constant.

**Correct**:
- Rotation experiment: change rotation angle; keep detector parameters, matching parameters, RANSAC threshold constant
- Illumination experiment: change brightness/contrast; keep all other parameters constant

**Incorrect**:
- Changing the RANSAC threshold between rotation experiment runs to "help" the algorithm succeed
- Using different `nfeatures` values for SIFT and ORB to "level the playing field" without documenting this

If you change any parameter between conditions, document it explicitly in `experiment_config.json`.

---

## RULE-E02: Run Both Methods on Every Experiment

Every experiment must be run with **both SIFT and ORB** under identical conditions. The comparison is only meaningful if the conditions are exactly the same.

---

## RULE-E03: Baseline Must Come First

The baseline experiment (three overlapping images, standard conditions) must be completed and verified before any stress experiments are run. Stress experiment results are meaningless without an established baseline to compare against.

---

## RULE-E04: Fixed Random Seed

Every experiment run must set:
```python
np.random.seed(42)
```
before any stochastic operation. This includes RANSAC (note: OpenCV's RANSAC has internal randomness not controlled by NumPy — document this).

Record the seed in `experiment_config.json`:
```json
{"random_seed": 42}
```

---

## RULE-E05: Save Config Before Running

Before executing any experiment, save the configuration to `experiment_config.json`:

```python
config = {
    'experiment': 'rotation',
    'condition': 'rot_045',
    'method': 'SIFT',
    'rotation_angle_deg': 45,
    'sift_nfeatures': 0,
    'ratio_test_threshold': 0.75,
    'ransac_threshold': 5.0,
    'random_seed': 42,
    'opencv_version': cv2.__version__,
    'numpy_version': np.__version__,
    'timestamp': datetime.utcnow().isoformat(),
}
save_config(config, output_dir / 'experiment_config.json')
```

This must happen before results are generated, not after.

---

## RULE-E06: Save Results Immediately After Each Trial

Do not accumulate all results in memory and save at the end. After each individual trial (one method, one condition, one image pair):
```python
metrics_df = pd.DataFrame([metrics_dict])
metrics_df.to_csv(results_path, mode='a', header=not results_path.exists(), index=False)
```

Using append mode ensures results are preserved even if a later trial crashes.

---

## RULE-E07: Synthetic Image Generation Must Be Reproducible

For rotation, scale, and viewpoint experiments using synthetic (programmatically generated) images:
- The generation function must be deterministic given the same inputs
- The generation code must be committed to the repository
- The generated images must be saved to `data/` alongside the generation script
- The `data/README.md` must document which images are synthetic and what the generation parameters were

---

## RULE-E08: Minimum Experiment Coverage

The experiment suite is considered complete only when:

| Experiment | Minimum conditions | Both methods |
|---|---|---|
| Baseline | 1 (standard overlap, ≥3 images) | ✓ |
| Rotation | ≥4 angles (e.g., 0°, 30°, 60°, 90°) | ✓ |
| Scale | ≥4 factors (e.g., 0.5×, 0.75×, 1.25×, 1.5×) | ✓ |
| Viewpoint | ≥3 levels (mild, moderate, extreme) | ✓ |
| Illumination | ≥4 conditions (2 brightness, 2 contrast or 1 each direction) | ✓ |

Fewer than these minimum conditions produces insufficient data for meaningful trend analysis.

---

## RULE-E09: Failure Trials Are Valid Results

If a trial produces a failure (RANSAC returns None, H is None, insufficient keypoints):
- Record the failure in `results/*.csv` with a clear flag: `homography_success=False`
- Record the failure code in `failures.json`
- Generate a failure visualization
- **Do not** re-run the trial with tweaked parameters to hide the failure

A failure trial is a real data point that informs the failure analysis section of the report.

---

## RULE-E10: Independent Trial Execution

Each experiment trial must be completely self-contained — no state shared between trials from the previous run:
```python
def run_single_trial(img1, img2, method, config, output_dir, trial_id):
    np.random.seed(config['random_seed'])  # Reset seed for each trial
    detector = create_detector(method, config)  # Create fresh detector
    # ... full pipeline ...
```

Do not reuse detector or matcher objects between trials.

---

## RULE-E11: Document Image Sources in Experiments

For each experiment, the `experiment_config.json` must record:
```json
{
  "img1": {
    "filename": "img_01.jpg",
    "source": "captured / synthetic",
    "condition": "reference"
  },
  "img2": {
    "filename": "rot_045.jpg",
    "source": "synthetic",
    "generation": "cv2.getRotationMatrix2D at 45 degrees from img_01.jpg"
  }
}
```

---

## RULE-E12: Controlled Variable Must Be Precisely Stated

For each stress experiment, the precise numerical value of the controlled variable must be recorded:
- Rotation: exact angle in degrees
- Scale: exact scale factor (e.g., 0.500, not "roughly half")
- Viewpoint: exact perspective warp offset in pixels (or exact camera position change)
- Illumination: exact `beta` value or `alpha` value used in `cv2.convertScaleAbs()`

"Roughly", "approximately", and "around" are not acceptable in the data records.

---

## RULE-E13: Plots Must Be Generated From CSV Files

All plots and charts in the report must be generated by code that reads from `results/*.csv` files:

```python
# Correct: load from saved results
df = pd.read_csv('results/rotation_results.csv')
plot_metric_vs_condition(df, 'inlier_ratio', 'rotation_angle_deg', ...)

# Forbidden: hardcode values in a plot script
# plt.plot([0, 30, 60, 90], [0.71, 0.68, 0.55, 0.21])
```

This ensures the plots are always consistent with the actual data.
