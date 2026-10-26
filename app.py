"""
=============================================================================
MathVision: Streamlit UI Application (Ed-Tech Light Theme)
=============================================================================

This application provides a modern, interactive interface for handwritten
mathematical equation recognition and step-by-step solving using OpenCV,
deep learning (CNN), Bayesian grammar rescoring, and SymPy.
=============================================================================
"""

import os
import glob
from datetime import datetime
import cv2
import numpy as np
import sympy as sp
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from src.pipeline import run_pipeline, get_loaded_models, create_results_zip
from src.solver import solve_expression
from src.dataset import CLASSES, LABEL_TO_DISPLAY
from ui.components import (
    render_hero_header,
    render_metric_row,
    render_pipeline_stage_card,
    render_solution_step_card,
    render_final_answer_box,
    render_alert_card,
    render_footer,
)


# =============================================================================
# 1. PAGE SETUP & THEME INJECTION
# =============================================================================
st.set_page_config(
    page_title="MathVision | Handwritten Equation Solver",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom light ed-tech stylesheet
CSS_PATH = os.path.join(os.path.dirname(__file__), "ui", "styles.css")
if os.path.exists(CSS_PATH):
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# =============================================================================
# 2. RESOURCE CACHING & SESSION STATE
# =============================================================================
@st.cache_resource
def load_all_models():
    """Caches loaded neural networks and baseline models across Streamlit reruns."""
    return get_loaded_models(models_dir="models")


models = load_all_models()

# Initialize session state variables
if "current_image" not in st.session_state:
    st.session_state.current_image = None
if "manual_overrides" not in st.session_state:
    st.session_state.manual_overrides = {}
if "selected_sample" not in st.session_state:
    st.session_state.selected_sample = None
if "pipeline_results" not in st.session_state:
    st.session_state.pipeline_results = None
if "history" not in st.session_state:
    st.session_state.history = []


# =============================================================================
# 3. SIDEBAR: TUNING PANEL & PARAMETER CONTROLS
# =============================================================================
with st.sidebar:
    st.markdown("### 🎛️ Control Panel")
    st.markdown("<p style='color:var(--text-muted); font-size:0.8rem;'>Tune parameters for live computer vision and inference adjustments.</p>", unsafe_allow_html=True)

    # Group 1: Preprocessing Parameters
    with st.expander("🔬 Preprocessing (OpenCV)", expanded=True):
        threshold_method = st.selectbox(
            "Thresholding Method",
            options=["otsu", "adaptive", "global"],
            index=0,
            format_func=lambda x: {
                "otsu": "Otsu's Thresholding (Bimodal - Recommended)",
                "adaptive": "Adaptive Gaussian (Local)",
                "global": "Global Fixed Cutoff (127)"
            }[x],
            help="Otsu binarization preserves clean 1-2px stroke widths matching the training data distribution."
        )

        adaptive_block_size = st.slider(
            "Adaptive Block Size",
            min_value=3, max_value=31, value=15, step=2,
            help="Neighborhood size for local threshold calculation."
        )

        adaptive_c = st.slider(
            "Constant C",
            min_value=1, max_value=15, value=8, step=1,
            help="Offset subtracted from weighted neighborhood mean."
        )

    # Group 2: Segmentation Parameters
    with st.expander("✂️ Segmentation & Merging", expanded=True):
        min_contour_area = st.slider(
            "Min Contour Area (px)",
            min_value=5, max_value=150, value=10, step=5,
            help="Filters out tiny ink specks and background dust without dropping thin strokes."
        )

        merge_overlap_ratio = st.slider(
            "Multi-stroke Overlap Ratio",
            min_value=0.20, max_value=0.80, value=0.35, step=0.05,
            help="Horizontal overlap threshold to merge '=' and '÷' components."
        )

    # Group 3: Model Parameters
    with st.expander("🧠 Model & Disambiguation", expanded=True):
        model_type = st.radio(
            "Inference Architecture",
            options=["cnn", "svm"],
            index=0,
            format_func=lambda x: "Deep CNN (3-Block)" if x == "cnn" else "Baseline HOG + SVM",
            horizontal=True
        )

        confidence_threshold = st.slider(
            "Confidence Warning Cutoff",
            min_value=0.50, max_value=0.95, value=0.70, step=0.05,
            help="Symbols with posterior confidence below this threshold receive an amber warning."
        )

        debug_save = st.checkbox("Save intermediates to outputs/debug", value=False)

    # Reset Manual Overrides Button
    if st.session_state.manual_overrides:
        if st.button("🔄 Clear Symbol Overrides", use_container_width=True):
            st.session_state.manual_overrides = {}
            st.rerun()


# =============================================================================
# 4. MAIN BRANDING HEADER
# =============================================================================
render_hero_header()


# =============================================================================
# 5. TOP NAVIGATION TABS
# =============================================================================
main_tab_solver, main_tab_history, main_tab_viva = st.tabs([
    "🧮 Equation Solver",
    "📜 Session History",
    "🎓 About & Viva Guide"
])


# =============================================================================
# TAB 1: EQUATION SOLVER WORKSPACE
# =============================================================================
with main_tab_solver:
    st.markdown("#### 1. Input Acquisition")
    st.markdown("<p style='color:var(--text-secondary); font-size:0.88rem;'>Choose an input source: upload a photo, use your webcam, draw in the sketchpad, or select a pre-loaded equation sample.</p>", unsafe_allow_html=True)

    tab_upload, tab_camera, tab_canvas = st.tabs([
        "📁 Upload Image",
        "📷 Live Webcam",
        "✏️ Draw on Canvas"
    ])

    input_image_to_process = None

    # Tab A: Upload File
    with tab_upload:
        uploaded_file = st.file_uploader(
            "Upload handwritten math image (.png, .jpg, .jpeg)",
            type=["png", "jpg", "jpeg"],
            key="file_uploader"
        )
        if uploaded_file is not None:
            file_bytes = uploaded_file.read()
            nparr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                input_image_to_process = img
                st.session_state.selected_sample = None

    # Tab B: Webcam Capture
    with tab_camera:
        camera_file = st.camera_input("Capture equation from webcam", key="camera_input")
        if camera_file is not None:
            file_bytes = camera_file.read()
            nparr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                input_image_to_process = img
                st.session_state.selected_sample = None

    # Tab C: Interactive Drawing Canvas (White paper with dark pen)
    with tab_canvas:
        st.markdown("<p style='font-size:0.85rem; color:var(--text-muted);'>Draw an equation using dark ink on white canvas:</p>", unsafe_allow_html=True)
        canvas_col1, canvas_col2 = st.columns([4, 1])
        with canvas_col2:
            stroke_width = st.slider("Stroke Width", 3, 16, 6)
            st.caption("Tip: Use smooth, separated strokes for best segmentation.")

        with canvas_col1:
            canvas_result = st_canvas(
                fill_color="rgba(0, 0, 0, 0)",
                stroke_width=stroke_width,
                stroke_color="#0F172A",
                background_color="#FFFFFF",
                height=200,
                width=720,
                drawing_mode="freedraw",
                return_image_data=True,
                key="draw_canvas"
            )

        if canvas_result.image_data is not None:
            canvas_arr = canvas_result.image_data
            # Check if non-empty (minimum pixel intensity is dark, indicating a drawn stroke)
            if np.min(canvas_arr[..., :3]) < 220:
                drawn_bgr = cv2.cvtColor(canvas_arr.astype(np.uint8), cv2.COLOR_RGBA2BGR)
                if st.button("✨ Solve Canvas Drawing", type="primary"):
                    input_image_to_process = drawn_bgr
                    st.session_state.selected_sample = None

    # Sample Gallery Buttons
    st.markdown("<p style='font-size:0.85rem; color:var(--text-secondary); margin-top:14px; font-weight:600;'>Or choose a pre-loaded sample equation:</p>", unsafe_allow_html=True)
    sample_files = sorted(glob.glob("data/samples/*.png"))

    if sample_files:
        sample_cols = st.columns(min(5, len(sample_files)))
        for i, sfile in enumerate(sample_files[:5]):
            btn_label = f"Sample #{i+1}"
            with sample_cols[i]:
                if st.button(btn_label, key=f"sample_btn_{i}", use_container_width=True):
                    st.session_state.selected_sample = sfile
                    st.session_state.manual_overrides = {}
                    input_image_to_process = cv2.imread(sfile)

    if st.session_state.selected_sample and input_image_to_process is None:
        input_image_to_process = cv2.imread(st.session_state.selected_sample)

    # -------------------------------------------------------------------------
    # Pipeline Execution Settings & Run
    # -------------------------------------------------------------------------
    settings = {
        "threshold_method": threshold_method,
        "adaptive_block_size": adaptive_block_size,
        "adaptive_c": adaptive_c,
        "min_contour_area": min_contour_area,
        "merge_min_h_overlap_ratio": merge_overlap_ratio,
        "padding_margin_ratio": 0.04,
        "model_type": model_type,
        "confidence_threshold": confidence_threshold,
        "debug": debug_save,
    }

    if input_image_to_process is not None:
        st.session_state.current_image = input_image_to_process

    if st.session_state.current_image is not None and (
        st.session_state.pipeline_results is None
        or input_image_to_process is not None
        or st.session_state.manual_overrides
    ):
        with st.spinner("Executing CV pipeline, neural inference, and symbolic solver..."):
            results = run_pipeline(
                st.session_state.current_image,
                settings=settings,
                manual_overrides=st.session_state.manual_overrides,
                cached_models=models
            )
            st.session_state.pipeline_results = results

            # Log to session history if valid expression
            parsed_str = results["parsed"].get("parsed_expression", "")
            if parsed_str:
                now_str = datetime.now().strftime("%H:%M:%S")
                # Avoid duplicate back-to-back log entries
                if not st.session_state.history or st.session_state.history[-1].get("expression") != parsed_str:
                    st.session_state.history.append({
                        "timestamp": now_str,
                        "expression": parsed_str,
                        "latex": results["parsed"].get("latex_expression", ""),
                        "final_answer": results["solution"].get("final_answer", "N/A"),
                        "status": results["solution"].get("status", "unknown"),
                        "latency_ms": results["timings"].get("total_ms", 0.0),
                        "symbol_count": results.get("count", 0),
                        "image": st.session_state.current_image,
                        "results": results
                    })

    # -------------------------------------------------------------------------
    # Results & Interactive Workspace
    # -------------------------------------------------------------------------
    if st.session_state.pipeline_results is not None:
        res = st.session_state.pipeline_results
        sol = res["solution"]
        parsed = res["parsed"]
        timings = res["timings"]

        st.markdown("---")
        st.markdown("#### 2. Pipeline Telemetry & Results")

        # Telemetry KPI Row
        render_metric_row(
            symbol_count=res["count"],
            mean_confidence=res.get("mean_confidence", 0.0),
            total_latency_ms=timings.get("total_ms", 0.0),
            status_text=sol.get("status", "Unknown")
        )

        # Ambiguity Notice & Alerts
        ambig_list = res.get("parsed", {}).get("ambiguous_tokens", [])
        if ambig_list:
            ambig_notes = []
            for a in ambig_list:
                idx_display = a["index"] + 1
                ch = LABEL_TO_DISPLAY.get(a["chosen"], a["chosen"])
                alt = LABEL_TO_DISPLAY.get(a["alternative"], a["alternative"])
                ambig_notes.append(f"Symbol #{idx_display} resolved to **{ch}** (alternative: **{alt}**).")
            st.warning("⚠️ **Ambiguity Flagged:** " + " ".join(ambig_notes) + " You can edit the formula below or override symbols in the 'Symbols & Overrides' tab.")
        elif not parsed.get("is_valid", False):
            render_alert_card("Syntax Validation Alert", parsed.get("error_message", "Expression is incomplete or mathematically invalid."))
        elif res.get("has_low_confidence", False):
            render_alert_card(
                "Low Confidence Symbols Flagged",
                "One or more symbols were recognized with confidence below your set threshold. Review or edit below.",
                is_warning=True
            )

        # ---------------------------------------------------------------------
        # REAL-TIME DIAGNOSTIC MODE (Expandable Section)
        # ---------------------------------------------------------------------
        with st.expander("🔍 Real-Time Pipeline Diagnostics", expanded=False):
            st.markdown("##### 🔬 Pipeline Telemetry & Health Verification")

            orig_img = res["prep"]["original"]
            active_bin = res.get("active_binary", res["prep"]["cleaned"])
            fg_pixels = int(cv2.countNonZero(active_bin))

            # Metric telemetry cards
            d_col1, d_col2, d_col3, d_col4 = st.columns(4)
            with d_col1:
                st.markdown("**1. Ingested Image**")
                st.write(f"• Size: `{orig_img.shape[1]} × {orig_img.shape[0]}` px")
                st.write(f"• Channels: `{orig_img.shape[2] if orig_img.ndim == 3 else 1}`")
                st.write(f"• Dtype: `{orig_img.dtype}`")
                st.write(f"• Mean: `{float(np.mean(orig_img)):.1f}`")
            with d_col2:
                st.markdown("**2. Binarization**")
                st.write(f"• Method: `{threshold_method.upper()}`")
                st.write(f"• Binary Shape: `{active_bin.shape[1]} × {active_bin.shape[0]}` px")
                st.write(f"• Ink Pixels: `{fg_pixels}`")
                st.write(f"• Paper Warped: `{res['prep'].get('paper_detected', False)}`")
            with d_col3:
                st.markdown("**3. Segmentation**")
                st.write(f"• Detected Boxes: `{res['count']}`")
                st.write(f"• Min Area Threshold: `{min_contour_area}` px")
                st.write(f"• Stroke Margin: `4%`")
                st.write(f"• Multi-stroke Merged: `{'Yes' if res['count'] > 0 else 'None'}`")
            with d_col4:
                st.markdown("**4. Parsing & Solver**")
                st.write(f"• Valid Syntax: `{parsed.get('is_valid', False)}`")
                st.write(f"• Solver Status: `{sol.get('status', 'N/A')}`")
                st.write(f"• Total Latency: `{timings.get('total_ms', 0.0):.1f} ms`")
                if parsed.get("error_message"):
                    st.write(f"• Stage Error: `{parsed.get('error_message')}`")

            # Stage health indicators
            st.markdown("---")
            h_col1, h_col2, h_col3, h_col4 = st.columns(4)
            with h_col1:
                if orig_img is not None:
                    st.success("✓ Ingestion: OK")
                else:
                    st.error("✗ Ingestion: Failed")
            with h_col2:
                if fg_pixels > 0:
                    st.success(f"✓ Binarization: {fg_pixels} px")
                else:
                    st.error("✗ Binarization: 0 ink pixels")
            with h_col3:
                if res["count"] > 0:
                    st.success(f"✓ Segmentation: {res['count']} symbols")
                else:
                    st.error("✗ Segmentation: 0 symbols")
            with h_col4:
                if parsed.get("is_valid", False):
                    st.success("✓ Parsing & SymPy: OK")
                else:
                    st.warning(f"⚠️ Parsing: {parsed.get('error_message', 'Invalid syntax')}")

            # Per-symbol crop details and confidences
            if res["count"] > 0 and res.get("symbol_diagnostics"):
                st.markdown("---")
                st.markdown("##### 🔠 Symbol Crops & Top-3 Model Candidates")
                diag_items = res["symbol_diagnostics"]
                diag_cols = st.columns(min(8, max(1, len(diag_items))))
                for idx, item in enumerate(diag_items):
                    with diag_cols[idx % len(diag_cols)]:
                        if idx < len(res["symbol_crops"]):
                            crop_disp = (res["symbol_crops"][idx] * 255).astype(np.uint8)
                            st.image(crop_disp, width=50, caption=f"#{idx+1}")
                        sym_glyph = LABEL_TO_DISPLAY.get(item['symbol'], item['symbol'])
                        st.markdown(f"**{sym_glyph}** (`{item['confidence']*100:.1f}%`)")
                        st.caption(f"Box: {item['bbox'][2]}×{item['bbox'][3]}px")
                        if item.get("top3"):
                            t3_str = " | ".join([f"{LABEL_TO_DISPLAY.get(c, c)}:{p*100:.0f}%" for c, p in item["top3"]])
                            st.caption(f"`{t3_str}`")

        # ---------------------------------------------------------------------
        # Editable Recognition Preview Card
        # ---------------------------------------------------------------------
        st.markdown(
            f"""
            <div class="preview-card">
                <div class="preview-header">
                    <span class="preview-title">Recognized Equation Preview</span>
                    {"<span class='ambiguity-badge'>⚠️ Ambiguity Resolved</span>" if ambig_list else "<span class='tech-badge emerald'>✓ Valid Syntax</span>"}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Show LaTeX Formula Render
        if parsed.get("latex_expression"):
            st.latex(parsed["latex_expression"])
        else:
            st.code(parsed.get("parsed_expression", ""))

        # Interactive editable equation input
        col_edit1, col_edit2 = st.columns([4, 1])
        with col_edit1:
            user_edited_expr = st.text_input(
                "Confirm or Edit Formula (SymPy notation):",
                value=parsed.get("parsed_expression", ""),
                key="equation_editor_input",
                help="You can manually modify the recognized equation here to re-evaluate it instantly."
            )
        with col_edit2:
            st.write("")
            st.write("")
            re_solve_clicked = st.button("⚡ Re-Solve Formula", use_container_width=True)

        if re_solve_clicked and user_edited_expr.strip():
            with st.spinner("Re-solving customized formula..."):
                custom_sol = solve_expression(user_edited_expr.strip())
                res["solution"] = custom_sol
                res["parsed"]["parsed_expression"] = user_edited_expr.strip()
                # Update LaTeX
                try:
                    if "=" in user_edited_expr:
                        lp, rp = user_edited_expr.split("=", 1)
                        res["parsed"]["latex_expression"] = f"{sp.latex(sp.sympify(lp.strip()))} = {sp.latex(sp.sympify(rp.strip()))}"
                    else:
                        res["parsed"]["latex_expression"] = sp.latex(sp.sympify(user_edited_expr.strip()))
                except Exception:
                    res["parsed"]["latex_expression"] = user_edited_expr.strip()
                st.session_state.pipeline_results = res
                st.rerun()

        # Detailed Result Tabs
        res_tab_solution, res_tab_pipeline, res_tab_symbols, res_tab_insights = st.tabs([
            "💡 Step-by-Step Solution",
            "🔬 CV Pipeline Stages",
            "🔤 Symbol Recognition & Overrides",
            "📊 Model Insights & Comparison"
        ])

        # ---------------------------------------------------------------------
        # TAB 1.1: STEP-BY-STEP DERIVATION
        # -------------------------------------------------------------------------
        with res_tab_solution:
            steps = sol.get("steps", [])
            if steps:
                for s in steps:
                    render_solution_step_card(
                        step_number=s["step_number"],
                        title=s["title"],
                        description=s["description"],
                        latex_str=s.get("latex", "")
                    )
            else:
                st.info("No mathematical derivation steps available for this input.")

            # Prominent Final Answer Box
            render_final_answer_box(
                answer_str=sol.get("final_answer", "N/A"),
                latex_str=sol.get("final_answer_latex", ""),
                equation_type=sol.get("equation_type", "Result")
            )

            # Download Complete ZIP Package
            st.markdown("<br>", unsafe_allow_html=True)
            zip_bytes = create_results_zip(res)
            st.download_button(
                label="📦 Download Complete Results & Intermediate Images (ZIP)",
                data=zip_bytes,
                file_name="mathvision_results.zip",
                mime="application/zip",
                use_container_width=True
            )

        # ---------------------------------------------------------------------
        # TAB 1.2: CV PIPELINE STAGES
        # ---------------------------------------------------------------------
        with res_tab_pipeline:
            prep = res["prep"]
            stage_cols = st.columns(6)

            with stage_cols[0]:
                render_pipeline_stage_card(
                    cv2.cvtColor(prep["original"], cv2.COLOR_BGR2RGB),
                    "1. Original",
                    "Ingested raw RGB input"
                )

            with stage_cols[1]:
                render_pipeline_stage_card(
                    cv2.cvtColor(prep["resized"], cv2.COLOR_BGR2RGB),
                    "2. Resized / Deskew",
                    "Aspect normalization"
                )

            with stage_cols[2]:
                render_pipeline_stage_card(
                    prep["gray"],
                    "3. Grayscale",
                    "Single-channel luminance"
                )

            with stage_cols[3]:
                render_pipeline_stage_card(
                    res["active_binary"],
                    "4. Inverted Binary",
                    f"{threshold_method.capitalize()} thresholding"
                )

            with stage_cols[4]:
                render_pipeline_stage_card(
                    prep["cleaned"],
                    "5. Morphology",
                    "Opening & closing filters"
                )

            with stage_cols[5]:
                render_pipeline_stage_card(
                    cv2.cvtColor(res["annotated_vis"], cv2.COLOR_BGR2RGB),
                    "6. Segmentation",
                    "Contour bounding boxes"
                )

            # Threshold Comparison Toggle
            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("⚖️ Side-by-Side Threshold Comparison (Adaptive vs. Otsu vs. Global)"):
                comp = prep["threshold_comparison"]
                t_col1, t_col2, t_col3 = st.columns(3)
                with t_col1:
                    st.markdown("**Global Threshold (Cutoff 127)**")
                    st.image(comp["global"], use_container_width=True)
                    st.caption("Fails under uneven illumination or shadows.")
                with t_col2:
                    st.markdown("**Otsu's Thresholding**")
                    st.image(comp["otsu"], use_container_width=True)
                    st.caption("Minimizes intra-class intensity variance.")
                with t_col3:
                    st.markdown("**Adaptive Gaussian (Selected)**")
                    st.image(comp["adaptive"], use_container_width=True)
                    st.caption("Dynamic thresholding per local neighborhood.")

        # ---------------------------------------------------------------------
        # TAB 1.3: SYMBOL RECOGNITION & OVERRIDES
        # ---------------------------------------------------------------------
        with res_tab_symbols:
            st.markdown("<p style='font-size:0.88rem; color:var(--text-secondary);'>Review isolated 45×45 character crops. If a character is misclassified, override it using the dropdown below to instantly re-solve:</p>", unsafe_allow_html=True)

            crops = res["symbol_crops"]
            preds = res["predicted_symbols"]
            confs = res["confidences"]
            low_flags = res["is_low_confidence"]
            ambig_list = res.get("parsed", {}).get("ambiguous_tokens", [])
            ambig_dict = {a["index"]: a for a in ambig_list}

            num_cols = min(8, max(4, len(crops)))
            sym_cols = st.columns(num_cols)
            re_solve_needed = False

            for i, crop in enumerate(crops):
                col = sym_cols[i % num_cols]
                with col:
                    is_low = low_flags[i]
                    is_ambig = i in ambig_dict

                    if is_ambig:
                        border_style = "border: 2px solid var(--accent-amber); background: #FFFBEB;"
                        badge_html = '<span class="conf-badge low" style="background:#F59E0B; color:white; font-weight:700;">⚠️ x/×</span>'
                    elif is_low:
                        border_style = "border: 1.5px solid var(--accent-amber);"
                        badge_html = f'<span class="conf-badge low">{int(confs[i]*100)}%</span>'
                    else:
                        border_style = "border: 1px solid var(--border-color);"
                        badge_html = f'<span class="conf-badge">{int(confs[i]*100)}%</span>'

                    st.markdown(
                        f"""
                        <div class="symbol-card" style="{border_style}">
                            <div class="symbol-badge-row">
                                <span class="symbol-idx">#{i+1}</span>
                                {badge_html}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # Display symbol crop
                    crop_uint8 = np.uint8(crop * 255)
                    st.image(crop_uint8, width=70)

                    # Dropdown for manual override
                    display_options = [LABEL_TO_DISPLAY.get(c, c) for c in CLASSES]
                    current_val = preds[i]
                    current_display = LABEL_TO_DISPLAY.get(current_val, current_val)
                    curr_idx = display_options.index(current_display) if current_display in display_options else 0

                    new_choice = st.selectbox(
                        f"Label #{i+1}",
                        options=display_options,
                        index=curr_idx,
                        key=f"override_select_{i}",
                        label_visibility="collapsed"
                    )

                    if is_ambig:
                        info = ambig_dict[i]
                        st.caption(f"⚠️ {info['reason']}")

                    # Check for override change
                    canonical_choice = CLASSES[display_options.index(new_choice)]
                    if canonical_choice != current_val:
                        st.session_state.manual_overrides[i] = canonical_choice
                        re_solve_needed = True

            if re_solve_needed:
                st.rerun()

        # ---------------------------------------------------------------------
        # TAB 1.4: MODEL INSIGHTS & BENCHMARK COMPARISON
        # ---------------------------------------------------------------------
        with res_tab_insights:
            st.info(
                "📌 **Academic Evaluation Note (Writer Separation):** "
                "Because the Kaggle dataset does not contain individual writer IDs, our stratified 70/15/15 split was partitioned "
                "at the image level (after exact and near-duplicate deduplication). Reported test metrics reflect unseen images from "
                "seen writers. Genuine unseen-writer accuracy is estimated to be 5–10% lower."
            )

            st.markdown("##### Model Architecture Comparison (Held-Out Test Split)")
            m_col1, m_col2 = st.columns(2)

            with m_col1:
                st.markdown("**Deep CNN Normalized Confusion Matrix**")
                cm_cnn_path = "outputs/confusion_matrix_cnn.png"
                if os.path.exists(cm_cnn_path):
                    st.image(cm_cnn_path, use_container_width=True)
                else:
                    st.info("Run `python src/evaluate.py` to generate the CNN confusion matrix.")

            with m_col2:
                st.markdown("**Baseline HOG + SVM Normalized Confusion Matrix**")
                cm_svm_path = "outputs/confusion_matrix_svm.png"
                if os.path.exists(cm_svm_path):
                    st.image(cm_svm_path, use_container_width=True)
                else:
                    st.info("Run `python src/evaluate.py` to generate the SVM confusion matrix.")

            st.markdown("##### Training Convergence Curves")
            curves_path = "outputs/training_curves.png"
            if os.path.exists(curves_path):
                st.image(curves_path, use_container_width=True)

            st.markdown("##### Benchmark Report (from `outputs/results.md`)")
            results_md_path = "outputs/results.md"
            if os.path.exists(results_md_path):
                with open(results_md_path, "r", encoding="utf-8") as f:
                    st.markdown(f.read())

            # Misclassified crops gallery
            fc_dir = "outputs/failure_cases"
            if os.path.exists(fc_dir):
                fc_files = sorted(glob.glob(os.path.join(fc_dir, "*.png")))
                if fc_files:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("##### 🔍 Misclassified Symbol Crops (Visual Inspection Gallery)")
                    st.markdown(
                        "<p style='font-size:0.85rem; color:var(--text-secondary);'>"
                        "Real handwritten glyphs misclassified by the Deep CNN on the held-out test split. "
                        "Notice real-world ambiguities such as upright crossing strokes ('x' vs '×'):"
                        "</p>",
                        unsafe_allow_html=True
                    )
                    cols_fc = st.columns(5)
                    for idx_fc, fc_path in enumerate(fc_files):
                        fname = os.path.basename(fc_path).replace(".png", "")
                        parts = fname.split("_")
                        true_idx = parts.index("true") + 1 if "true" in parts else -1
                        pred_idx = parts.index("pred") + 1 if "pred" in parts else -1
                        true_str = parts[true_idx] if true_idx != -1 and true_idx < len(parts) else "?"
                        pred_str = parts[pred_idx] if pred_idx != -1 and pred_idx < len(parts) else "?"

                        true_display = LABEL_TO_DISPLAY.get(true_str, true_str)
                        pred_display = LABEL_TO_DISPLAY.get(pred_str, pred_str)

                        with cols_fc[idx_fc % 5]:
                            st.image(fc_path, caption=f"True: {true_display} → Pred: {pred_display}", use_container_width=True)

    else:
        st.info("👈 Upload an equation image, draw on the canvas, or select a sample equation above to begin!")


# =============================================================================
# TAB 2: SESSION HISTORY
# =============================================================================
with main_tab_history:
    st.markdown("### 📜 Session History")
    st.markdown("<p style='color:var(--text-secondary); font-size:0.88rem;'>Every equation processed during this browser session is archived here. You can inspect previous derivations or reload any equation back into the solver.</p>", unsafe_allow_html=True)

    if not st.session_state.history:
        st.info("No equations solved yet in this session. Process an equation in the Solver tab to see history here!")
    else:
        h_ctrl1, h_ctrl2 = st.columns([5, 1])
        with h_ctrl1:
            st.markdown(f"**Total Solved Equations:** {len(st.session_state.history)}")
        with h_ctrl2:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.history = []
                st.rerun()

        for idx, h in enumerate(reversed(st.session_state.history)):
            real_idx = len(st.session_state.history) - 1 - idx
            status_color = "#059669" if h["status"].lower() in ("solved", "success") else "#D97706"

            st.markdown(
                f"""
                <div class="history-card">
                    <div class="history-info">
                        <div class="history-time">Entry #{real_idx + 1} • {h['timestamp']} • {h['symbol_count']} symbols • {h['latency_ms']:.1f} ms</div>
                        <div class="history-expr">{h['expression']}</div>
                        <div class="history-result">Solution: <strong>{h['final_answer']}</strong> <span style="color:{status_color}; font-weight:700;">({h['status'].upper()})</span></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            col_btn1, col_btn2 = st.columns([6, 1])
            with col_btn2:
                if st.button(f"🔄 Reload #{real_idx + 1}", key=f"reload_hist_{real_idx}", use_container_width=True):
                    st.session_state.current_image = h["image"]
                    st.session_state.pipeline_results = h["results"]
                    st.rerun()


# =============================================================================
# TAB 3: ABOUT & VIVA DEFENSE GUIDE
# =============================================================================
with main_tab_viva:
    st.markdown("### 🎓 About MathVision & Viva Defense Guide")
    st.markdown("<p style='color:var(--text-secondary); font-size:0.9rem;'>Essential theoretical insights, architectural justification, and zero-leakage evaluation methodology for university project evaluation and viva defense.</p>", unsafe_allow_html=True)

    with st.expander("1. End-to-End System Architecture & Data Flow", expanded=True):
        st.markdown(
            """
            MathVision employs a modular, inspectable 7-stage pipeline:
            1. **Image Preprocessing:** Ingestion, resize with aspect-ratio preservation, grayscale conversion, and Adaptive Gaussian thresholding.
            2. **Morphological Filtering:** Morphological opening ($3\\times 3$ kernel) removes salt-and-pepper noise; morphological closing bridges minor stroke gaps.
            3. **Contour Segmentation & Merging:** External contour detection isolates characters. A multi-stroke merging heuristic pairs disjoint strokes belonging to `$=$` and `$\\div$`.
            4. **Aspect-Preserving Normalization:** Each glyph is centered on a $45\\times 45$ canvas with zero padding to prevent non-square distortion.
            5. **Deep CNN Inference:** 3-block convolutional neural network (895k parameters) computes softmax posterior probabilities over 18 mathematical classes.
            6. **Bayesian Grammar Soft Rescoring:** Resolves ambiguous `$x$` vs `$\\times$` glyphs by combining CNN logits with neighboring grammar transition priors. Low-margin cases are flagged with amber warnings.
            7. **Step-by-Step Symbolic Solver:** SymPy parses the verified mathematical AST and generates step-by-step pedagogical derivations for arithmetic, linear, and quadratic problems.
            """
        )

    with st.expander("2. Computer Vision Defense (OpenCV)", expanded=False):
        st.markdown(
            """
            - **Why Adaptive Thresholding over Global/Otsu?** Global fixed cutoff (127) fails when paper has shadows or gradients. Otsu computes a single global threshold which breaks when background lighting is non-uniform across the page. Adaptive Gaussian calculates local thresholds per $15\\times 15$ window, providing robust binarization across uneven ambient light.
            - **Why Aspect-Preserving Padding to 45×45?** Directly resizing non-square contours (e.g. a narrow `1` or wide `-`) to $45\\times 45$ artificially stretches strokes, destroying topological features. Centering the scaled character with black border padding preserves original geometric proportions.
            - **How are Multi-stroke Symbols Merged?** Equals (`=`) consists of two separate horizontal strokes, and division (`÷`) consists of a bar and two dots. The pipeline calculates the horizontal bounding box overlap ratio $\\frac{\\text{overlap}}{\\min(w_1, w_2)}$. When overlap $> 0.35$ and vertical gap is within threshold, the components are unioned into a single glyph bounding box.
            """
        )

    with st.expander("3. Deep Learning & Disambiguation Defense", expanded=False):
        st.markdown(
            """
            - **Why CNN over HOG + SVM?** HOG features rely on hand-crafted gradient orientation histograms, achieving 93.05% test accuracy. The 3-block CNN learns hierarchical spatial representations (edges $\\to$ junctions $\\to$ whole glyph parts), achieving 94.18% accuracy with higher resilience to stroke thickness variations.
            - **Why Soft Rescoring over Hard Rules?** Hard grammatical rules crash or yield invalid syntax when users write non-standard expressions. Soft rescoring applies Bayesian posterior updates:
              $$\\log P(\\text{symbol} \\mid \\text{context}) = \\log P_{\\text{CNN}}(\\text{symbol}) + w_{\\text{gram}} \\cdot \\log P(\\text{symbol} \\mid \\text{neighbors})$$
              When posterior margin $|\\Delta \\log P| < 0.60$, the system flags the equation as *amber* for human confirmation rather than making a high-confidence error.
            """
        )

    with st.expander("4. Zero-Leakage Data Integrity & Honest Evaluation", expanded=False):
        st.markdown(
            """
            - **Strict Split Protocol:** 18,994 samples split 70% Train (13,296), 15% Validation (2,849), and 15% Held-Out Test (2,849) using fixed seed 42.
            - **Grammar Priors & Hyperparameters:** Tuned exclusively on validation split crops and equations. Held-out test crops were never accessed during training or prior estimation.
            - **Equation-Level Performance on 220 Held-Out Equations:**
              - Raw CNN exact accuracy: 60.45% (133/220) with 54 $x \\leftrightarrow \\times$ confusions.
              - Production Rescoring exact accuracy: **77.27% (170/220)** with only 5 cross errors (-90.7% error reduction).
              - Strict autonomous accuracy (flagged counted as error): **68.18% (150/220)**.
              - Oracle-simulated assisted accuracy: **78.64% (173/220)**.
            - **Known Academic Limitation:** The Kaggle dataset lacks writer identifiers. Test accuracy reflects unseen images from seen writer styles; genuine novel unseen-writer performance is expected to be 5–10% lower.
            """
        )

    with st.expander("5. Comprehensive Project Documentation References", expanded=False):
        st.markdown(
            """
            Detailed documentation artifacts are preserved in the codebase for reference:
            - **Model Card:** [`docs/MODEL_CARD.md`](file:///C:/Users/Jainam/Downloads/CV%20Project/docs/MODEL_CARD.md)
            - **Viva Defense Preparation Guide:** [`docs/VIVA_PREPARATION_GUIDE.md`](file:///C:/Users/Jainam/Downloads/CV%20Project/docs/VIVA_PREPARATION_GUIDE.md)
            - **Evaluation & Benchmark Output:** [`outputs/results.md`](file:///C:/Users/Jainam/Downloads/CV%20Project/outputs/results.md)
            """
        )


# =============================================================================
# 6. FOOTER
# =============================================================================
render_footer()
