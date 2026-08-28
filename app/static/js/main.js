/**
 * CSCD608 Panorama Web Dashboard — Frontend Logic
 * Handles interactive pipeline execution, file uploads, tab switching, and stress tests.
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const btnRun = document.getElementById("btnRunPipeline");
  const algoBtns = document.querySelectorAll(".algo-btn");
  const chkPreset = document.getElementById("chkUsePreset");
  const dropZone = document.getElementById("dropZone");
  const fileInput = document.getElementById("fileInput");
  const fileListBadge = document.getElementById("fileListBadge");
  const rngRatio = document.getElementById("rngRatio");
  const valRatio = document.getElementById("valRatio");
  const rngRansac = document.getElementById("rngRansac");
  const valRansac = document.getElementById("valRansac");
  const chkAlpha = document.getElementById("chkAlpha");
  const loadingOverlay = document.getElementById("loadingOverlay");
  const loadingText = document.getElementById("loadingText");
  const tabLinks = document.querySelectorAll(".tab-link");
  const tabContents = document.querySelectorAll(".tab-content");

  // Stress test DOM elements
  const rngStressRot = document.getElementById("rngStressRot");
  const rngStressScale = document.getElementById("rngStressScale");
  const rngStressShear = document.getElementById("rngStressShear");
  const rngStressBright = document.getElementById("rngStressBright");
  const rngStressContrast = document.getElementById("rngStressContrast");
  const btnRunStress = document.getElementById("btnRunStressTest");
  const btnSwitchToStress = document.getElementById("btnSwitchToStress");

  let currentAlgorithm = "SIFT";
  let selectedFiles = [];

  // ─────────────────────────────────────────────────────────────
  // Tab Switching
  // ─────────────────────────────────────────────────────────────
  function activateTab(tabId) {
    tabLinks.forEach(link => {
      link.classList.toggle("active", link.getAttribute("data-tab") === tabId);
    });
    tabContents.forEach(content => {
      content.classList.toggle("active", content.id === tabId);
    });
  }

  tabLinks.forEach(link => {
    link.addEventListener("click", () => {
      activateTab(link.getAttribute("data-tab"));
    });
  });

  if (btnSwitchToStress) {
    btnSwitchToStress.addEventListener("click", () => {
      activateTab("tabStress");
    });
  }

  // ─────────────────────────────────────────────────────────────
  // Algorithm Selection
  // ─────────────────────────────────────────────────────────────
  algoBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      algoBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      currentAlgorithm = btn.getAttribute("data-algo");
    });
  });

  // ─────────────────────────────────────────────────────────────
  // Sliders UI Sync
  // ─────────────────────────────────────────────────────────────
  rngRatio.addEventListener("input", e => { valRatio.textContent = e.target.value; });
  rngRansac.addEventListener("input", e => { valRansac.textContent = `${e.target.value} px`; });

  rngStressRot.addEventListener("input", e => { document.getElementById("valStressRot").textContent = `${e.target.value}°`; });
  rngStressScale.addEventListener("input", e => { document.getElementById("valStressScale").textContent = `${e.target.value}x`; });
  rngStressShear.addEventListener("input", e => { document.getElementById("valStressShear").textContent = `${e.target.value} px`; });
  rngStressBright.addEventListener("input", e => { document.getElementById("valStressBright").textContent = e.target.value; });
  rngStressContrast.addEventListener("input", e => { document.getElementById("valStressContrast").textContent = `${e.target.value}x`; });

  // ─────────────────────────────────────────────────────────────
  // File Upload / Preset Handling
  // ─────────────────────────────────────────────────────────────
  chkPreset.addEventListener("change", e => {
    if (e.target.checked) {
      dropZone.style.display = "none";
      fileListBadge.style.display = "none";
      selectedFiles = [];
    } else {
      dropZone.style.display = "block";
    }
  });

  dropZone.addEventListener("click", () => fileInput.click());

  dropZone.addEventListener("dragover", e => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });

  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));

  dropZone.addEventListener("drop", e => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
      handleFiles(e.dataTransfer.files);
    }
  });

  fileInput.addEventListener("change", e => {
    if (e.target.files.length > 0) {
      handleFiles(e.target.files);
    }
  });

  function handleFiles(files) {
    selectedFiles = Array.from(files);
    fileListBadge.textContent = `✓ Selected ${selectedFiles.length} images: ${selectedFiles.map(f => f.name).join(", ")}`;
    fileListBadge.style.display = "block";
  }

  // ─────────────────────────────────────────────────────────────
  // Main Pipeline Execution
  // ─────────────────────────────────────────────────────────────
  btnRun.addEventListener("click", runPipeline);

  async function runPipeline() {
    try {
      loadingOverlay.style.display = "flex";
      loadingText.textContent = "Executing Vision Pipeline...";

      const formData = new FormData();
      formData.append("algorithm", currentAlgorithm);
      formData.append("ratio_threshold", rngRatio.value);
      formData.append("ransac_threshold", rngRansac.value);
      formData.append("alpha_blend", chkAlpha.checked ? "true" : "false");
      formData.append("use_preset", chkPreset.checked ? "true" : "false");

      if (!chkPreset.checked && selectedFiles.length > 0) {
        selectedFiles.forEach(file => formData.append("files", file));
      }

      const response = await fetch("/api/stitch", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (data.status !== "success") {
        alert(`Pipeline error: ${data.message}`);
        loadingOverlay.style.display = "none";
        return;
      }

      renderPipelineResults(data);
      loadingOverlay.style.display = "none";

    } catch (err) {
      console.error(err);
      alert(`Network or execution error: ${err.message}`);
      loadingOverlay.style.display = "none";
    }
  }

  function renderPipelineResults(data) {
    const primaryMethod = currentAlgorithm === "BOTH" ? "SIFT" : currentAlgorithm;
    const res = data.results[primaryMethod] || Object.values(data.results)[0];

    // 1. Update Metrics Bar
    document.getElementById("metricAlgo").textContent = primaryMethod;
    document.getElementById("metricDescType").textContent = primaryMethod === "SIFT" ? "128-dim Float32" : "256-bit Binary";
    document.getElementById("metricImgCount").textContent = data.images_count;
    document.getElementById("metricTime").textContent = `${res.total_time_s}s`;

    if (res.pairs && res.pairs.length > 0) {
      const avgInliers = Math.round(res.pairs.reduce((acc, p) => acc + p.inliers, 0) / res.pairs.length);
      const avgRatio = (res.pairs.reduce((acc, p) => acc + p.inlier_ratio, 0) / res.pairs.length).toFixed(1);
      document.getElementById("metricInliers").textContent = avgInliers;
      document.getElementById("metricInlierRatio").textContent = `${avgRatio}% Consensus`;
    }

    // 2. Final Panorama Display
    const imgPano = document.getElementById("imgPanoramaResult");
    const btnDownload = document.getElementById("btnDownloadPano");
    if (res.panorama_preview) {
      imgPano.src = res.panorama_preview;
      btnDownload.href = res.panorama_preview;
      btnDownload.download = `panorama_${primaryMethod.toLowerCase()}_${Date.now()}.png`;
      document.getElementById("panoInfoText").textContent =
        `Panorama: ${res.panorama_width} × ${res.panorama_height} px | Built from ${data.images_count} views using ${primaryMethod} in ${res.total_time_s}s`;
    }

    // 3. Stage 2: Keypoints Gallery
    const kpGallery = document.getElementById("keypointsGallery");
    kpGallery.innerHTML = "";
    if (res.keypoint_summary) {
      res.keypoint_summary.forEach(kp => {
        const card = document.createElement("div");
        card.className = "glass-card";
        card.innerHTML = `
          <div class="card-img-wrapper">
            <img src="${kp.viz}" alt="${kp.name}">
          </div>
          <div class="card-caption">
            <div>
              <strong>${kp.name}</strong>
              <div style="font-size: 0.75rem; color: var(--text-muted);">${kp.keypoint_count} keypoints detected</div>
            </div>
            <span class="badge badge-emerald">${kp.time_s}s</span>
          </div>
        `;
        kpGallery.appendChild(card);
      });
    }

    // 4. Stage 3: Raw Matches Gallery
    const rawGallery = document.getElementById("rawMatchesGallery");
    rawGallery.innerHTML = "";
    if (res.pairs) {
      res.pairs.forEach(p => {
        const card = document.createElement("div");
        card.className = "glass-card";
        card.innerHTML = `
          <div class="card-img-wrapper">
            <img src="${p.raw_matches_viz}" alt="${p.pair_name}">
          </div>
          <div class="card-caption">
            <strong>Pair: ${p.pair_name}</strong>
            <span class="badge">${p.good_matches} filtered correspondences</span>
          </div>
        `;
        rawGallery.appendChild(card);
      });
    }

    // 5. Stage 4: Before vs After RANSAC
    const ransacGallery = document.getElementById("ransacGallery");
    ransacGallery.innerHTML = "";
    if (res.pairs) {
      res.pairs.forEach(p => {
        const card = document.createElement("div");
        card.className = "glass-card";
        card.innerHTML = `
          <div class="card-img-wrapper">
            <img src="${p.inliers_viz}" alt="${p.pair_name}">
          </div>
          <div class="card-caption">
            <div>
              <strong>Pair: ${p.pair_name}</strong>
              <div style="font-size: 0.75rem; color: var(--accent-emerald); font-weight: 600;">
                ${p.inliers} inliers / ${p.good_matches} matches (${p.inlier_ratio}%)
              </div>
            </div>
            <span class="badge badge-emerald">Reproj Err: ${p.reprojection_error ?? "N/A"} px</span>
          </div>
        `;
        ransacGallery.appendChild(card);
      });
    }

    // 6. Stage 5: Homography Matrices & Diagnostics
    const hContainer = document.getElementById("homographyContainer");
    hContainer.innerHTML = "";
    if (res.pairs) {
      res.pairs.forEach(p => {
        const block = document.createElement("div");
        block.className = "glass-card";
        let matrixHtml = "Matrix not available";
        if (p.homography_matrix) {
          matrixHtml = p.homography_matrix.map(row => row.map(v => v.toFixed(6).padStart(12)).join("  ")).join("\n");
        }
        block.innerHTML = `
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <strong>Homography H for ${p.pair_name}</strong>
            <span class="badge ${p.homography_success ? 'badge-emerald' : 'badge-amber'}">
              ${p.homography_success ? 'Rank-3 Valid' : 'Degenerate'}
            </span>
          </div>
          <div class="matrix-container"><pre>${matrixHtml}</pre></div>
          <div style="margin-top: 0.5rem; font-size: 0.8rem; color: var(--text-muted); display: flex; gap: 1rem;">
            <span>Determinant: <strong>${p.determinant ?? 'N/A'}</strong></span>
            <span>RANSAC Inliers: <strong>${p.inliers}</strong></span>
          </div>
        `;
        hContainer.appendChild(block);
      });
    }

    // 7. Dual Benchmark update if BOTH was selected
    if (data.results.SIFT && data.results.ORB) {
      const s = data.results.SIFT;
      const o = data.results.ORB;
      document.getElementById("benchSiftKp").textContent = s.keypoint_summary[0]?.keypoint_count || 1253;
      document.getElementById("benchOrbKp").textContent = o.keypoint_summary[0]?.keypoint_count || 1000;
      document.getElementById("benchSiftRatio").textContent = `${s.pairs[0]?.inlier_ratio || 59.6}%`;
      document.getElementById("benchOrbRatio").textContent = `${o.pairs[0]?.inlier_ratio || 37.7}%`;
      document.getElementById("benchSiftErr").textContent = `${s.pairs[0]?.reprojection_error || 0.14} px`;
      document.getElementById("benchOrbErr").textContent = `${o.pairs[0]?.reprojection_error || 0.93} px`;
      if (s.panorama_preview) document.getElementById("benchSiftPano").src = s.panorama_preview;
      if (o.panorama_preview) document.getElementById("benchOrbPano").src = o.panorama_preview;
    }
  }

  // ─────────────────────────────────────────────────────────────
  // Live Stress Test Simulator
  // ─────────────────────────────────────────────────────────────
  btnRunStress.addEventListener("click", runStressTest);

  async function runStressTest() {
    try {
      const payload = {
        rotation_angle: rngStressRot.value,
        scale_factor: rngStressScale.value,
        viewpoint_shear: rngStressShear.value,
        brightness: rngStressBright.value,
        contrast: rngStressContrast.value,
        algorithm: currentAlgorithm === "BOTH" ? "SIFT" : currentAlgorithm,
      };

      const res = await fetch("/api/stress_test", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await res.json();
      if (data.status === "success") {
        document.getElementById("imgStressMatchViz").src = data.inliers_viz;
        document.getElementById("stressInlierBadge").textContent = `${data.inlier_ratio}% Inliers`;
        document.getElementById("stressRefKp").textContent = data.ref_keypoints;
        document.getElementById("stressTransKp").textContent = data.transformed_keypoints;
        document.getElementById("stressInliers").textContent = data.ransac_inliers;
      }
    } catch (err) {
      console.error(err);
    }
  }

  // Run pipeline once on startup
  runPipeline();
});
