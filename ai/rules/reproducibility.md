# Rules: Reproducibility

## Status
These rules ensure that every experiment can be reproduced exactly by any person with access to the repository. They are **mandatory**.

---

## RULE-REP01: Fixed Random Seeds Are Mandatory

Before any stochastic operation, set a fixed seed:

```python
import numpy as np
np.random.seed(42)
```

Set this at the start of **every experiment script** and at the start of **every individual trial function**.

**OpenCV caveat**: `cv2.findHomography()` with `cv2.RANSAC` uses OpenCV's internal random number generator, which is not controlled by `np.random.seed()`. This means RANSAC results may vary slightly between runs even with the same NumPy seed.

**Required response to this caveat**:
1. Document this in the experiment methodology
2. Run each RANSAC-based trial multiple times and report mean ± std
3. For the standard comparison, use a fixed trial count (e.g., 5 runs per condition)

---

## RULE-REP02: Record Software Versions

Every experiment config JSON must include:

```json
{
  "python_version": "3.10.12",
  "opencv_version": "4.8.1",
  "numpy_version": "1.24.3",
  "matplotlib_version": "3.7.1",
  "pandas_version": "2.0.3",
  "scikit_image_version": "0.21.0",
  "platform": "Windows/Linux/macOS",
  "cpu": "optional: processor model"
}
```

Generate this programmatically:

```python
import sys
import cv2
import numpy as np
import matplotlib
import pandas as pd
import skimage

def get_versions() -> dict:
    return {
        'python_version': sys.version,
        'opencv_version': cv2.__version__,
        'numpy_version': np.__version__,
        'matplotlib_version': matplotlib.__version__,
        'pandas_version': pd.__version__,
        'skimage_version': skimage.__version__,
    }
```

---

## RULE-REP03: All Experiment Parameters Must Be in Config Files

No hardcoded parameters may exist in experiment scripts. All parameters must come from `experiments/config.py` or a trial-specific config dict that is saved to `experiment_config.json`.

This means:
- Anyone running the experiment uses the same parameters
- Changing a parameter is done in one place and automatically applies everywhere

---

## RULE-REP04: Data Must Be Version-Controlled or Documented

**Option A** (preferred): Commit all images to the repository. This ensures exact reproducibility.

**Option B** (if images are too large): Commit a `data/README.md` that documents exactly:
- Where to obtain each image
- What transformations were applied (e.g., resize to X×Y, rotation by Z°)
- What filename to save it as
- Checksum (MD5 or SHA256) to verify the correct file was obtained

Without one of these options, someone else cannot reproduce the experiments.

---

## RULE-REP05: Synthetic Image Generation Scripts Must Be Committed

If any experiment images were generated programmatically (rotation, scale, viewpoint, illumination variants), the generation script must be committed to the repository:

```
experiments/generate_rotation_images.py
experiments/generate_scale_images.py
experiments/generate_viewpoint_images.py
experiments/generate_illumination_images.py
```

These scripts must be deterministic given the same input images.

---

## RULE-REP06: Results Must Be Committed to the Repository

The following must be committed to git (not .gitignored):
- `results/baseline_results.csv`
- `results/rotation_results.csv`
- `results/scale_results.csv`
- `results/viewpoint_results.csv`
- `results/illumination_results.csv`
- `results/comparison_table.csv`

This allows anyone to verify results without re-running all experiments, and allows examiners to inspect the raw numbers.

---

## RULE-REP07: Key Visualizations Must Be Committed

A representative set of visualizations must be committed to git:
- `outputs/baseline/SIFT/panorama_SIFT.png`
- `outputs/baseline/ORB/panorama_ORB.png`
- `outputs/comparison/method_comparison_bar.png`
- One rotation experiment visualization per method
- At least one failure case visualization

The full `outputs/` directory may be large; use `.gitignore` selectively to exclude intermediate files while keeping key outputs.

---

## RULE-REP08: README Must Contain Exact Run Commands

The main `README.md` must include copy-pasteable commands:

```markdown
## Running the Experiments

### Setup
pip install -r requirements.txt

### Baseline experiment
python experiments/run_baseline.py

### Rotation experiment
python experiments/run_rotation.py

### All experiments (sequential)
python experiments/run_all.py

### Generate comparison table and plots
python experiments/compare_methods.py
```

**Prohibited**: "Run the main script to see results" without specifying what the script is.

---

## RULE-REP09: .gitignore Must Be Properly Configured

```gitignore
# Python
__pycache__/
*.pyc
*.pyo
.venv/
venv/
*.egg-info/

# Experiment intermediate outputs (large files)
outputs/*/SIFT/descriptors/
outputs/*/ORB/descriptors/
outputs/*/SIFT/warped_uncropped.png
outputs/*/ORB/warped_uncropped.png

# Temporary files
*.tmp
.DS_Store
Thumbs.db

# Jupyter checkpoints
.ipynb_checkpoints/

# Keep these (tracked):
# results/*.csv
# outputs/baseline/*/panorama_*.png
# outputs/comparison/*.png
```

---

## RULE-REP10: No Absolute Paths in Code

**Prohibited**:
```python
img = cv2.imread('C:/Users/YourName/Desktop/project/data/img_01.jpg')
```

**Required**:
```python
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent  # Works regardless of where code is run from
img_path = PROJECT_ROOT / 'data' / 'baseline' / 'img_01.jpg'
img = cv2.imread(str(img_path))
```

Using absolute paths makes the project non-reproducible on any machine other than the developer's.

---

## RULE-REP11: requirements.txt Must Be Up-to-Date

Before submission, generate a fresh `requirements.txt`:

```bash
pip freeze > requirements.txt
```

Or use a curated version with only project dependencies (not the full environment). Either way, `requirements.txt` must be correct and complete so that:
```bash
pip install -r requirements.txt
```
produces a working environment.

---

## RULE-REP12: Timestamp All Experiment Runs

Every experiment run must record its start time in UTC:
```python
from datetime import datetime, timezone
timestamp = datetime.now(timezone.utc).isoformat()
```

This enables identifying which CSV rows correspond to which run if an experiment is re-run after the codebase has changed.
