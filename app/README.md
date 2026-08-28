# CSCD608 Interactive Panorama Web Dashboard

A web dashboard allowing examiners, researchers, and students to interactively test, visualize, and benchmark the CSCD608 Feature-Based Image Matching & Automatic Panorama Construction System.

---

## Features

1. **Interactive Multi-Image Upload & Preset Scenes**:
   - Upload 2 to 6 overlapping photos or test instantly with built-in panoramic scenes.
2. **Dynamic Algorithm Switching & Hyperparameter Tuning**:
   - SIFT (128-float), ORB (256-binary), or Dual Side-by-Side Benchmark.
   - Real-time sliders for Lowe's Ratio Threshold ($0.50$–$0.95$) and RANSAC Reprojection Threshold ($1.0$–$15.0\text{ px}$).
3. **Step-by-Step Visualizer**:
   - Preprocessed grayscale views.
   - Rich Keypoints (DoG scale/orientation circles vs. FAST corners).
   - Filtered Feature Match lines.
   - Before vs. After RANSAC side-by-side outlier rejection with inlier ratio badge.
   - $3\times3$ Homography matrix values, determinant, and condition number.
   - Final stitched panorama with distance-weighted alpha blending and full-resolution download.
4. **Live Transformation Stress Playground**:
   - Sliders for in-plane rotation ($-180^\circ$ to $+180^\circ$), scale ($0.5\times$ to $2.0\times$), perspective shear, and illumination ($\Delta\beta, \alpha$).

---

## Launching the Web App

```bash
# Launch the dashboard
python run_app.py

# Or launch directly from the app folder:
python app/server.py --port 5000 --host 127.0.0.1
```

Once running, open [http://127.0.0.1:5000](http://127.0.0.1:5000) in any web browser.

---

## Cloud & Online Deployment (Hugging Face Spaces / Render / AWS)

### Deploying to Hugging Face Spaces:
1. Create a new Space on [Hugging Face](https://huggingface.co/spaces) with SDK: `Docker` or `Gradio/FastAPI/Flask`.
2. Push this repository.
3. Set the entrypoint to `python run_app.py --port 7860 --host 0.0.0.0`.

### Deploying to Render / Railway / Heroku:
- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn -w 2 -b 0.0.0.0:$PORT app.server:app`
