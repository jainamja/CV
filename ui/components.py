"""
=============================================================================
MathVision: Reusable Modern UI Components (Ed-Tech Light Theme)
=============================================================================

This module contains presentation helpers styled according to the clean
light ed-tech theme in ui/styles.css.
=============================================================================
"""

import streamlit as st
import numpy as np
from typing import List, Dict, Any, Optional


def render_hero_header() -> None:
    """Renders the top branding hero card with technology badges."""
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-title-row">
                <div>
                    <h1 class="hero-title">📐 MathVision</h1>
                    <p class="hero-tagline">Handwritten Mathematical Equation Recognition & Step-by-Step Solver</p>
                </div>
                <div class="hero-badges">
                    <span class="tech-badge blue">⚡ OpenCV Classical CV</span>
                    <span class="tech-badge blue">🧠 Deep CNN (895k params)</span>
                    <span class="tech-badge emerald">📐 SymPy Symbolic Engine</span>
                    <span class="tech-badge">🎯 Bayesian Rescoring</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_metric_row(
    symbol_count: int,
    mean_confidence: float,
    total_latency_ms: float,
    status_text: str = "Solved"
) -> None:
    """Renders the top KPI telemetry summary cards."""
    status_color = "var(--accent-emerald)" if status_text.lower() in ("solved", "success") else "var(--accent-amber)"
    conf_pct = int(round(mean_confidence * 100))

    st.markdown(
        f"""
        <div class="metric-row">
            <div class="metric-card">
                <div class="metric-label">Symbols Detected</div>
                <div class="metric-value">{symbol_count}</div>
                <div class="metric-subtext">Isolated & segmented</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Mean Confidence</div>
                <div class="metric-value">{conf_pct}%</div>
                <div class="metric-subtext">Across detected glyphs</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Pipeline Latency</div>
                <div class="metric-value">{total_latency_ms:.1f} <span style="font-size:0.85rem; font-weight:500; color:var(--text-muted);">ms</span></div>
                <div class="metric-subtext">End-to-end CPU runtime</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Solver Status</div>
                <div class="metric-value" style="color: {status_color}; font-size:1.35rem;">{status_text.upper()}</div>
                <div class="metric-subtext">Symbolic verification</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_pipeline_stage_card(
    image: np.ndarray,
    title: str,
    caption: str,
    use_container_width: bool = True
) -> None:
    """Renders an individual CV stage card with image and explanation."""
    st.markdown(
        f"""
        <div class="stage-title">{title}</div>
        """,
        unsafe_allow_html=True
    )
    st.image(image, use_container_width=use_container_width)
    st.markdown(
        f"""
        <div class="stage-caption">{caption}</div>
        """,
        unsafe_allow_html=True
    )


def render_solution_step_card(
    step_number: int,
    title: str,
    description: str,
    latex_str: str
) -> None:
    """Renders a single step-by-step mathematical working card."""
    st.markdown(
        f"""
        <div class="solution-step-card">
            <div class="step-header">
                <span class="step-num-badge">STEP {step_number}</span>
                <span class="step-title">{title}</span>
            </div>
            <div class="step-description">{description}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    if latex_str:
        st.latex(latex_str)


def render_final_answer_box(
    answer_str: str,
    latex_str: str = "",
    equation_type: str = "Equation"
) -> None:
    """Renders the prominent high-impact final answer card."""
    type_label = f"FINAL RESULT ({equation_type.upper()})"
    st.markdown(
        f"""
        <div class="final-answer-box">
            <div class="final-answer-label">{type_label}</div>
            <div class="final-answer-val">{answer_str}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    if latex_str and "\\text" not in latex_str:
        st.latex(latex_str)


def render_alert_card(title: str, message: str, is_warning: bool = False) -> None:
    """Renders diagnostic error or warning cards without ugly stack traces."""
    card_class = "alert-card warning" if is_warning else "alert-card"
    icon = "⚠️" if is_warning else "❌"
    st.markdown(
        f"""
        <div class="{card_class}">
            <div style="font-size: 1.25rem;">{icon}</div>
            <div>
                <div class="alert-title">{title}</div>
                <div class="alert-text">{message}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_footer() -> None:
    """Renders academic project attribution footer."""
    st.markdown(
        """
        <div class="app-footer">
            <p><strong>MathVision</strong> — Computer Vision Laboratory Project | OpenCV + Deep CNN + SymPy Engine</p>
            <p style="color: var(--text-muted); font-size: 0.75rem;">Structured for University Engineering Evaluation & Viva Defense</p>
        </div>
        """,
        unsafe_allow_html=True
    )
