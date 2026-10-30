# MathVision — Deployment Readiness Audit (Phase 5)

**Document ID:** `MATHVISION-DEPLOYMENT-AUDIT-PHASE5`  
**Date:** October 30, 2026  
**Auditor:** Antigravity Engineering (AI Assistant)  
**Target Platform:** Render (Web Service)  
**Public Repository:** `https://github.com/jainamja/CV`  
**Audit Scope:** Inspection-Only (Zero Code/Model Modifications Applied)  

---

## 1. Executive Summary

A comprehensive pre-deployment readiness audit was conducted on the MathVision project to evaluate its suitability for public cloud hosting on **Render**.

### Key Audit Findings:
1. **Model & Heuristic Integrity (Verified):** The local production CNN checkpoint is confirmed to be **Model V4** (SHA-256: `5d2559c57477cc0646f696a0bff781145bd728a1be5bd630478faaf33e9b3d56`), and the **Stage 3.5 Topological Disambiguation Heuristic** is verified active in `src/pipeline.py`.
2. **Major Deployment Blocker — Local vs GitHub Divergence:** The remote GitHub `main` branch (`origin/main`) is currently frozen at commit `8b57b0c` (the old baseline model). **None of the Phase 2, 3, 4, or 5 enhancements, bugfixes, or Model V4 checkpoints are pushed to GitHub.** Deploying to Render directly from GitHub today would deploy the legacy failing system.
3. **Major Dependency Blocker — Headless Linux OpenCV:** `requirements.txt` specifies `opencv-python>=4.8.0`. On standard Render Python Linux environments (which lack X11 and OpenGL libraries), importing `opencv-python` fails immediately with `ImportError: libGL.so.1: cannot open shared object file`. It must be changed to `opencv-python-headless>=4.8.0`.
4. **Memory Constraint on Render Free Tier (512 MB RAM):** Running Streamlit, TensorFlow 2.x, Keras 3, OpenCV, and SymPy simultaneously requires approximately **380 MB – 480 MB RAM**. On Render's 512 MB Free Tier, memory spikes during model inference or multi-request usage risk triggering the Linux Out-Of-Memory (OOM) killer (`Exit code 137`).
5. **No Credentials Leaked, but Local Path Leakage Present:** No API keys or passwords exist in the codebase. However, multiple markdown documentation files (including `README.md`) contain absolute Windows developer paths (`C:\Users\Jainam\...`).

---

## 2. Current Git State & Local vs Remote Comparison

### 2.1 Git Status & Commit Inspection
- **Current Branch:** `main`
- **HEAD Commit:** `8b57b0c03af4de227af21b347ed068a726ccb764`  
  *Commit Message:* `docs: finalize viva preparation guide, demo walkthrough, test report, and README for college submission`
- **Remote URL:** `https://github.com/jainamja/CV.git`
- **Tracking Status:** Local branch is up to date with `origin/main` at commit `8b57b0c`, but **working tree has extensive unstaged modifications**.

### 2.2 Working Tree Modifications (13 Tracked Files Changed)
The local working directory has **3,468 insertions and 332 deletions** compared to `origin/main`:
```
 app.py                      | 285 +++++++++---------
 models/cnn_mathvision.keras | Bin 3579586 -> 3579586 bytes (Promoted from Baseline to V4)
 requirements.txt            |   1 -
 src/parser.py               | 372 +++++++++++++++++++++--
 src/pipeline.py             | 694 ++++++++++++++++++++++++++++++++++++++++--
 src/preprocess.py           | 436 +++++++++++++++++++++++++--
 src/segment.py              | 529 +++++++++++++++++++++++++++++---
 src/solver.py               | 720 +++++++++++++++++++++++++++++++++++++++-----
 tests/test_parser.py        | 138 +++++++++
 tests/test_preprocess.py    | 117 +++++++
 tests/test_segment.py       | 342 +++++++++++++++++++++
 tests/test_solver.py        | 138 +++++++++
 ui/styles.css               |  28 ++
```

### 2.3 Untracked Files Requiring Categorization Before Git Staging
The repository contains 85 untracked files across multiple categories:

| Category | Files | Action Required Before Deployment |
| :--- | :--- | :--- |
| **Critical Tests** | `tests/test_pipeline.py`, `tests/test_case_e_slash.py`, `tests/test_phase3_1_topological.py`, etc. | **Must be committed** to ensure CI/CD and regression protection. |
| **New Phase Docs** | `docs/phase*.md`, `docs/deployment_readiness_audit.md` | **Commit** for architectural documentation and auditing. |
| **Temporary / Backup Files** | `*.bak`, `*.bak_*`, `*.pre_phase5_backup` | **Add to `.gitignore`**; do NOT commit to public repository. |
| **Heavy Experimental Data** | `data/processed/split_data_v2/v3/v4.npz` (~96 MB total) | **Add to `.gitignore`**; would severely bloat git repository. |
| **Intermediate Zip Archives** | `outputs/mathvision_results_*.zip` (~40 MB total) | **Add to `.gitignore`**; ephemeral test bundles. |
| **Personal Academic Reports** | `MathVision_Project_Report.pdf`, `MathVision_Project_Report.md` | **Review for privacy**; exclude from git if personal details are present. |

---

## 3. Production Model & Heuristic Verification

### 3.1 Model Checkpoint Cryptographic Hash
- **Target File:** `models/cnn_mathvision.keras`
- **Expected SHA-256:** `5d2559c57477cc0646f696a0bff781145bd728a1be5bd630478faaf33e9b3d56`
- **Actual SHA-256:** `5d2559c57477cc0646f696a0bff781145bd728a1be5bd630478faaf33e9b3d56`
- **Verification Status:** **100% MATCH (Model V4 CNN Active)**

### 3.2 Production Heuristic Verification
- **Target File:** `src/pipeline.py` (lines 617–644)
- **Heuristic:** Stage 3.5 Topological Disambiguation for Orthogonal Division (`div` / `÷`) vs Plus (`+`).
- **Implementation Status:** Verified present with two-layer defense:
  - **Layer 1 Neural Gating:** Confirms low CNN plus confidence ($P(+) < 0.65$) with competing division ($P(\text{div}) \ge 0.15$).
  - **Layer 2 Pixel Morphology:** `check_division_topology()` verifies prominent central horizontal bar ($\ge 45\%$ width, $\ge 12\text{px}$) and detached upper/lower dot strokes separated by empty whitespace rows ($\ge 2\text{px}$).

---

## 4. Required Runtime Files & Packaging Audit

For MathVision to launch and operate cleanly on Render, the following runtime assets are strictly required:

```
├── app.py                               <-- Application entry point
├── .streamlit/
│   └── config.toml                      <-- Streamlit server & theme configuration
├── ui/
│   ├── components.py                    <-- UI presentation widgets & hero headers
│   └── styles.css                       <-- Ed-tech modern light stylesheet
├── models/
│   ├── cnn_mathvision.keras             <-- Promoted Model V4 CNN (3.58 MB)
│   └── svm_baseline.joblib              <-- Optional HOG+SVM baseline model (30.4 MB)
├── src/
│   ├── pipeline.py                      <-- 7-stage CV orchestration & arbitration
│   ├── preprocess.py                    <-- Classical OpenCV contrast & morphology
│   ├── segment.py                       <-- Contour retrieval & multi-stroke heuristics
│   ├── model.py                         <-- HOG feature extraction & CNN loader
│   ├── dataset.py                       <-- 18 canonical class definitions
│   ├── parser.py                        <-- Bayesian grammar rescoring & SymPy syntax
│   └── solver.py                        <-- Symbolic equation solving
├── data/
│   └── samples/                         <-- Preloaded equation samples (#1 to #10)
│       └── sample_*.png
└── outputs/                             <-- Static charts for Tab 4 (Technical Insights)
    ├── confusion_matrix_cnn.png
    ├── confusion_matrix_svm.png
    ├── training_curves.png
    ├── results.md
    └── failure_cases/
```

### File Availability Verification:
- All required runtime files are present in the local workspace.
- `app.py` includes safe fallback guards (`if os.path.exists(...)`) for Tab 4 static graphics, preventing runtime crashes if any auxiliary chart is absent.
- `create_results_zip()` operates entirely in-memory using Python's `io.BytesIO()`, eliminating local disk write requirements for student downloads.

---

## 5. Dependency & Linux Compatibility Findings

### 5.1 Critical Blocker: `opencv-python` vs `opencv-python-headless`
- **Issue:** `requirements.txt` currently specifies `opencv-python>=4.8.0`.
- **Cause:** Standard `opencv-python` links against X11, OpenGL (`libGL.so.1`), and desktop windowing libraries. Render's standard Linux containers do not include GUI/display packages.
- **Consequence:** At container boot, `import cv2` will crash with:
  ```
  ImportError: libGL.so.1: cannot open shared object file: No such file or directory
  ```
- **Remedy:** Replace `opencv-python` with `opencv-python-headless` in `requirements.txt`. The headless package provides 100% of the image processing functions used by MathVision (`cv2.cvtColor`, `cv2.threshold`, `cv2.findContours`, `cv2.warpAffine`, etc.) without GUI dependencies.

### 5.2 `streamlit-drawable-canvas` Status
- **Investigation:** An audit for `streamlit-drawable-canvas` confirmed it is **not imported or used** anywhere in `app.py`, `ui/`, or `src/`.
- **Finding:** The application receives inputs via `st.file_uploader`, `st.camera_input`, and preloaded sample images. There is no dependency on `streamlit-drawable-canvas`.

### 5.3 Missing Explicit Dependency: `joblib`
- **Issue:** `src/pipeline.py:35` and `src/model.py` import `joblib` directly to load the optional SVM baseline.
- **Finding:** `joblib` is currently installed as a transitive dependency of `scikit-learn`. However, best practice for cloud deployment is to declare `joblib>=1.3.0` explicitly in `requirements.txt`.

### 5.4 Proposed Production `requirements.txt`
```
opencv-python-headless>=4.8.0
numpy>=1.24.0,<2.0.0
tensorflow>=2.15.0
keras>=3.0.0
sympy>=1.12
streamlit>=1.30.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
scikit-image>=0.22.0
Pillow>=9.5.0
joblib>=1.3.0
```
*(Note: Pinning `numpy<2.0.0` is strongly recommended for TensorFlow 2.15/2.16 stability on Linux).*

---

## 6. Storage, Memory & Hosting Considerations (Render)

### 6.1 Repository Size
- **Total Local Workspace:** ~442 MB (dominated by `data/raw/` containing >200,000 Kaggle symbol images).
- **Core Repository (excluding `data/raw/` and `.git`):** ~107 MB.
- **Runtime Model Size:**
  - `cnn_mathvision.keras`: **3.58 MB** (Extremely lightweight).
  - `svm_baseline.joblib`: **30.4 MB** (Moderate).
- **GitHub Compatibility:** All individual files are well below GitHub's 100 MB hard limit.

### 6.2 Render Memory (RAM) Analysis
- **Render Free Tier Limit:** **512 MB RAM**.
- **MathVision Memory Profile (Linux container):**
  - Python runtime + Streamlit base: ~100 MB.
  - TensorFlow + Keras runtime: ~280 MB.
  - Model V4 CNN loaded in memory: ~15 MB.
  - OpenCV image processing buffers: ~20 MB.
  - SymPy symbolic engine: ~15 MB.
  - **Estimated Baseline RSS:** **~430 MB – 460 MB**.
- **Risk Assessment:** While 430 MB fits within 512 MB, heavy concurrent requests or loading `svm_baseline.joblib` (an additional ~35 MB RAM) leaves less than a 50 MB buffer. If RAM exceeds 512 MB, Render will send `SIGKILL` (status 137).
- **Mitigation Strategies:**
  1. Set environment variable `TF_ENABLE_ONEDNN_OPTS=0` and `TF_CPP_MIN_LOG_LEVEL=2` to reduce TensorFlow allocation overhead.
  2. Lazy-load `svm_baseline.joblib` only if the user explicitly switches the model dropdown from CNN to SVM.
  3. If multi-user traffic is expected, upgrade Render service from Free (512 MB) to **Starter (1 GB RAM, \$7/mo)**.

---

## 7. OS & Environment Portability Audit

### 7.1 Path Separators
- Path operations in `app.py` and `src/` use `os.path.join` or forward slashes (`/`), which are fully portable between Windows and Linux.
- No Windows-only backslash path bugs were identified in runtime code.

### 7.2 Windows Font Fallbacks
- `src/dataset.py:86` and `src/generate_samples.py:25` contain references to `C:\Windows\Fonts\...`.
- **Finding:** Both scripts include automated fallbacks (`if os.path.exists(...)` and `ImageFont.load_default()`), and neither script is executed during live web application inference. Therefore, this will not cause runtime errors on Linux.

---

## 8. Security & Privacy Audit

1. **Credentials & Secrets:** Zero API keys, passwords, tokens, or cloud credentials were found.
2. **Local Path Privacy:**
   - `README.md:381` contains: `cd "c:\Users\Jainam\Downloads\CV Project"`. This should be sanitized to `cd MathVision` before making the GitHub repository public.
   - Documentation files in `docs/` contain reference links to the local development environment.
3. **Sensitive Artifacts to Exclude:**
   - Backup files (`*.bak`, `*.bak_*`, `*.pre_phase5_backup`) must be ignored.
   - Any private college evaluation rubrics or personal student PDFs in `MathVision_Project_Report.pdf` should be reviewed before making the GitHub repository public.

---

## 9. Recommended Render Configuration

To deploy MathVision on Render as a Web Service:

### 9.1 Render Web Service Settings

| Setting | Value |
| :--- | :--- |
| **Service Type** | Web Service |
| **Runtime** | Python 3 |
| **Branch** | `main` |
| **Root Directory** | `.` (blank / root) |
| **Build Command** | `pip install --upgrade pip && pip install -r requirements.txt` |
| **Start Command** | `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true` |
| **Plan** | Free (512 MB) for demo / Starter (1 GB) for production reliability |

### 9.2 Required Environment Variables on Render

| Variable Key | Recommended Value | Purpose |
| :--- | :--- | :--- |
| `PYTHON_VERSION` | `3.11.9` | Locks a highly stable, pre-built wheel Linux Python runtime. |
| `TF_ENABLE_ONEDNN_OPTS` | `0` | Disables TensorFlow floating-point round-off variances and reduces memory overhead. |
| `TF_CPP_MIN_LOG_LEVEL` | `2` | Silences TensorFlow C++ informational logging, keeping Render build logs clean. |
| `STREAMLIT_SERVER_HEADLESS` | `true` | Ensures Streamlit runs headlessly without expecting an interactive terminal. |
| `STREAMLIT_BROWSER_GATHER_USAGE_STATS` | `false` | Disables telemetry to optimize startup speed and network calls. |

---

## 10. Prioritized Deployment Checklist (For Future Execution)

When authorization is granted to proceed with deployment, follow these prioritized steps:

### Phase A: Dependency & Hygiene Preparation
- [ ] 1. Update `requirements.txt`: switch `opencv-python` $\to$ `opencv-python-headless>=4.8.0`, add `joblib>=1.3.0`, and pin `numpy<2.0.0`.
- [ ] 2. Update `.gitignore`: add `*.bak*`, `*.pre_phase5_backup`, `data/processed/split_data_v*.npz`, and `outputs/*.zip`.
- [ ] 3. Sanitize `README.md`: replace `cd "c:\Users\Jainam\Downloads\CV Project"` with standard portable instructions (`cd MathVision`).

### Phase B: Git Commit & Remote Synchronization
- [ ] 4. Stage production code (`app.py`, `src/`, `ui/`, `models/cnn_mathvision.keras`, `requirements.txt`).
- [ ] 5. Stage automated test suite (`tests/`).
- [ ] 6. Create clean deployment commit: `feat: deploy MathVision with Model V4 CNN and Stage 3.5 topological disambiguation`.
- [ ] 7. Push commit to `https://github.com/jainamja/CV.git` on branch `main`.

### Phase C: Render Service Provisioning
- [ ] 8. Connect GitHub repository `jainamja/CV` in Render Dashboard.
- [ ] 9. Configure build and start commands as specified in Section 9.1.
- [ ] 10. Configure environment variables (`PYTHON_VERSION=3.11.9`, `TF_ENABLE_ONEDNN_OPTS=0`).
- [ ] 11. Trigger initial deployment and monitor live build logs.
- [ ] 12. Perform live smoke test on the Render `.onrender.com` URL (uploading sample equations, verifying Model V4 badges, checking solve latencies).

---

> [!NOTE]
> **Audit Status:** Inspection-Only Phase Completed. No files were modified, no dependencies were altered, and no Git actions were taken. Awaiting your review and explicit authorization before proceeding to any deployment preparation steps.
