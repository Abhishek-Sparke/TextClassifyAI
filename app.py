"""
TextClassify - AI-Powered Text Document Classification System (4 Classes)
Production-grade Document Intelligence Interface with Automated Inference,
Unknown / Out-of-Domain Detection, and Multi-Topic Ambiguity Analysis.
"""

import os
import io
import time
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from src.preprocessing import preprocess_document
from src.features import get_top_tfidf_terms_for_document
from src.models import get_prediction_probabilities
from src.inference_engine import (
    classify_document,
    InferenceConfig,
    DEFAULT_INFERENCE_CONFIG,
    TARGET_4_CLASSES,
    CATEGORY_DISPLAY_NAMES_4,
    CATEGORY_ICONS_4
)

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="TextClassify | Document Intelligence Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------------------------------------------------------
# Premium Dark-First AI SaaS Theme
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --bg-dark: #07090e;
        --bg-card: rgba(14, 20, 31, 0.75);
        --bg-card-hover: rgba(22, 32, 50, 0.9);
        --border-subtle: rgba(255, 255, 255, 0.08);
        --border-glow: rgba(99, 102, 241, 0.35);
        --accent-primary: #6366f1;
        --accent-secondary: #8b5cf6;
        --accent-cyan: #06b6d4;
        --accent-emerald: #10b981;
        --accent-amber: #f59e0b;
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --font-sans: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        --font-mono: 'JetBrains Mono', monospace;
    }

    [data-testid="stSidebar"], section[data-testid="stSidebar"], [data-testid="collapsedControl"] {
        display: none !important;
        visibility: hidden !important;
        width: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    #MainMenu {visibility: hidden; display: none !important;}
    header {visibility: hidden; display: none !important;}
    footer {visibility: hidden; display: none !important;}
    .stDeployButton {display: none !important;}
    [data-testid="stToolbar"] {display: none !important;}
    div[data-testid="stDecoration"] {display: none !important;}

    .stApp {
        background: radial-gradient(circle at 50% 0%, #111827 0%, #07090e 75%, #030508 100%) !important;
        font-family: var(--font-sans) !important;
        color: var(--text-primary) !important;
    }

    .main .block-container {
        padding-top: 0.8rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 1280px !important;
    }

    .tc-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        text-decoration: none;
    }
    .tc-logo-icon {
        width: 34px;
        height: 34px;
        border-radius: 10px;
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 17px;
        box-shadow: 0 0 16px rgba(99, 102, 241, 0.4);
    }
    .tc-brand-title {
        font-size: 17px;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #ffffff 40%, #a5b4fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .tc-brand-pill {
        font-size: 9px;
        font-family: var(--font-mono);
        font-weight: 700;
        text-transform: uppercase;
        padding: 2px 7px;
        background: rgba(99, 102, 241, 0.18);
        border: 1px solid rgba(99, 102, 241, 0.35);
        color: #a5b4fc;
        border-radius: 999px;
    }

    .tc-card {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 18px;
        padding: 24px;
        backdrop-filter: blur(14px);
        margin-bottom: 20px;
        position: relative;
    }
    .tc-card-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 18px;
    }
    .tc-card-title {
        font-size: 16px;
        font-weight: 750;
        color: #ffffff;
        letter-spacing: -0.01em;
    }
    .tc-card-sub {
        font-size: 12.5px;
        color: var(--text-muted);
        margin-top: 2px;
    }

    /* Prediction Banners */
    .tc-pred-banner-normal {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(14, 20, 31, 0.85) 100%);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 18px;
    }
    .tc-pred-banner-ambiguous {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(14, 20, 31, 0.85) 100%);
        border: 1px solid rgba(245, 158, 11, 0.35);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 18px;
    }
    .tc-pred-banner-unknown {
        background: linear-gradient(135deg, rgba(100, 116, 139, 0.15) 0%, rgba(14, 20, 31, 0.85) 100%);
        border: 1px solid rgba(100, 116, 139, 0.3);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 18px;
    }

    .tc-pred-kicker {
        font-family: var(--font-mono);
        font-size: 10.5px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        margin-bottom: 6px;
    }
    .tc-pred-cat {
        font-size: 22px;
        font-weight: 800;
        color: #ffffff;
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 10px;
    }

    /* Badges */
    .tc-badge-normal {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        background: rgba(16, 185, 129, 0.16);
        border: 1px solid rgba(16, 185, 129, 0.4);
        color: #6ee7b7;
        font-size: 11px;
        font-weight: 700;
        border-radius: 999px;
    }
    .tc-badge-ambiguous {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        background: rgba(245, 158, 11, 0.16);
        border: 1px solid rgba(245, 158, 11, 0.4);
        color: #fcd34d;
        font-size: 11px;
        font-weight: 700;
        border-radius: 999px;
    }
    .tc-badge-unknown {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        background: rgba(100, 116, 139, 0.2);
        border: 1px solid rgba(100, 116, 139, 0.4);
        color: #cbd5e1;
        font-size: 11px;
        font-weight: 700;
        border-radius: 999px;
    }

    /* Callout Note */
    .tc-callout-note {
        padding: 12px 16px;
        background: rgba(99, 102, 241, 0.08);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 12px;
        font-size: 12px;
        color: #c7d2fe;
        margin: 14px 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* Probability Bar */
    .tc-conf-row {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 7px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    }
    .tc-conf-label {
        width: 140px;
        font-size: 12px;
        font-weight: 600;
        color: #e2e8f0;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .tc-conf-track {
        flex: 1;
        height: 8px;
        background: rgba(255, 255, 255, 0.06);
        border-radius: 999px;
        overflow: hidden;
    }
    .tc-conf-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #6366f1 0%, #06b6d4 100%);
    }
    .tc-conf-fill-highlight {
        background: linear-gradient(90deg, #f59e0b 0%, #ef4444 100%) !important;
    }
    .tc-conf-pct {
        width: 52px;
        text-align: right;
        font-family: var(--font-mono);
        font-size: 11px;
        color: #94a3b8;
    }

    /* Keywords */
    .tc-tag-wrap {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }
    .tc-keyword-tag {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 11px;
        background: rgba(99, 102, 241, 0.09);
        border: 1px solid rgba(99, 102, 241, 0.22);
        border-radius: 8px;
        font-size: 12px;
        font-weight: 600;
        color: #c7d2fe;
    }
    .tc-keyword-wt {
        font-size: 10px;
        font-family: var(--font-mono);
        color: #818cf8;
        background: rgba(99, 102, 241, 0.2);
        padding: 1px 5px;
        border-radius: 4px;
    }

    /* Stat Cards Grid */
    .tc-stat-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
        margin-bottom: 22px;
    }
    .tc-stat-card {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 14px;
        padding: 16px;
    }
    .tc-stat-label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--text-muted);
        margin-bottom: 6px;
    }
    .tc-stat-val {
        font-size: 24px;
        font-weight: 800;
        color: #ffffff;
    }
    .tc-stat-sub {
        font-size: 11px;
        color: #818cf8;
        margin-top: 4px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Curated Test Examples
# -----------------------------------------------------------------------------
TEST_EXAMPLES = {
    "🚀 Pure Space": "NASA launched a spacecraft into orbit using advanced cryogenic rocket propulsion.",
    "⚾ Pure Baseball": "The baseball team scored five runs in the bottom of the ninth inning to win the game.",
    "🏛️ Pure Politics": "The government passed a new law regarding federal taxation and civil liberties.",
    "🎨 Pure Graphics": "The computer generated a 3D image using polygon mesh shading and ray tracing.",
    "⚡ Short Space": "NASA launch",
    "⚡ Short Baseball": "baseball game",
    "⚡ Short Politics": "new government law",
    "⚡ Short Graphics": "3D rendering software",
    "🔀 Mixed: Govt & Mars": "The government announced a new Mars mission.",
    "🔀 Mixed: President & Mars": "The president discussed NASA's Mars program.",
    "🍕 Out of Domain: Pizza": "I ate pizza today.",
    "🔋 Out of Domain: Battery": "My laptop battery is low.",
    "🛍️ Out of Domain: Shopping": "I went shopping yesterday.",
    "⚠️ Ambiguous Benchmark": "The Chief Minister bought a Royal Enfield bike and went to Mars to see Jesus and play football."
}


# -----------------------------------------------------------------------------
# Artifact Loader with Caching
# -----------------------------------------------------------------------------
@st.cache_resource
def load_project_artifacts():
    if not os.path.exists('models/best_model.joblib') or not os.path.exists('models/tfidf_vectorizer.joblib'):
        return None, None, None, None

    vectorizer = joblib.load('models/tfidf_vectorizer.joblib')
    best_model = joblib.load('models/best_model.joblib')
    all_models = joblib.load('models/all_models.joblib')

    with open('models/model_metadata.json', 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    return vectorizer, best_model, all_models, metadata


# -----------------------------------------------------------------------------
# Main Application
# -----------------------------------------------------------------------------
def main():
    if 'active_nav' not in st.session_state:
        st.session_state.active_nav = 'Classify'
    if 'doc_input' not in st.session_state:
        st.session_state.doc_input = 'The Chief Minister bought a Royal Enfield bike and went to Mars to see Jesus and play football.'
    if 'last_prediction' not in st.session_state:
        st.session_state.last_prediction = None
    if 'batch_df' not in st.session_state:
        st.session_state.batch_df = None

    vectorizer, best_model, all_models, metadata = load_project_artifacts()

    if vectorizer is None:
        st.error("⚠️ Model artifacts not found in `models/`. Run `python train.py` to train and save models first.")
        st.stop()

    categories = metadata.get('categories', TARGET_4_CLASSES)
    cat_display = metadata.get('category_display_names', CATEGORY_DISPLAY_NAMES_4)
    metrics_table = pd.DataFrame(metadata.get('metrics_table', []))
    stats = metadata.get('dataset_statistics', {})
    best_model_name = metadata.get('best_model_name', 'Multinomial Naive Bayes')

    # Reconstruct inference config
    thresh_meta = metadata.get('confidence_thresholds', {})
    inference_config = InferenceConfig(
        confidence_threshold=thresh_meta.get('confidence_threshold', DEFAULT_INFERENCE_CONFIG.confidence_threshold),
        min_active_tfidf=thresh_meta.get('min_active_tfidf', DEFAULT_INFERENCE_CONFIG.min_active_tfidf),
        ambiguity_margin=thresh_meta.get('ambiguity_margin', DEFAULT_INFERENCE_CONFIG.ambiguity_margin),
        min_ambiguity_sum=thresh_meta.get('min_ambiguity_sum', DEFAULT_INFERENCE_CONFIG.min_ambiguity_sum),
        topic_detection_threshold=thresh_meta.get('topic_detection_threshold', DEFAULT_INFERENCE_CONFIG.topic_detection_threshold)
    )

    # -------------------------------------------------------------------------
    # STICKY TOP NAVIGATION BAR
    # -------------------------------------------------------------------------
    col_nav_brand, col_nav_tabs, col_nav_status = st.columns([2.5, 5.5, 2.0], vertical_alignment="center")

    with col_nav_brand:
        st.markdown('''
        <div class="tc-brand">
            <div class="tc-logo-icon">⚡</div>
            <div>
                <div class="tc-brand-title">TextClassify</div>
            </div>
            <div class="tc-brand-pill">4-CLASS ML</div>
        </div>
        ''', unsafe_allow_html=True)

    with col_nav_tabs:
        nav_options = ["Classify", "CSV Batch", "Analytics", "Dataset", "About"]
        current_nav = st.session_state.active_nav if st.session_state.active_nav in nav_options else "Classify"
        selected_nav = st.segmented_control(
            "Navigation",
            options=nav_options,
            default=current_nav,
            label_visibility="collapsed",
            key="top_navbar_segmented"
        )
        if selected_nav and selected_nav != st.session_state.active_nav:
            st.session_state.active_nav = selected_nav
            st.rerun()

    with col_nav_status:
        st.markdown(f'''
        <div style="display:flex;justify-content:flex-end;align-items:center;">
            <div style="font-family:var(--font-mono);font-size:11px;color:#a5b4fc;background:rgba(99,102,241,0.12);padding:4px 10px;border-radius:8px;border:1px solid rgba(99,102,241,0.25);">
                ● {best_model_name}
            </div>
        </div>
        ''', unsafe_allow_html=True)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # ROUTE 1: CLASSIFY (Single Document Inference with Normal/Ambiguous/Unknown)
    # -------------------------------------------------------------------------
    if st.session_state.active_nav == "Classify":
        col_input, col_output = st.columns([1.1, 1], gap="large")

        # LEFT COLUMN: INPUT
        with col_input:
            st.markdown('''
            <div class="tc-card">
                <div class="tc-card-header">
                    <div>
                        <div class="tc-card-title">📝 Document Input</div>
                        <div class="tc-card-sub">Test inputs across pure topics, short text, out-of-domain, or ambiguous cases.</div>
                    </div>
                </div>
            ''', unsafe_allow_html=True)

            # Test Examples Selector
            st.markdown("<div style='font-size:11px;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:8px;'>SELECT TEST EXAMPLE</div>", unsafe_allow_html=True)
            
            ex_cols1 = st.columns(4)
            ex_keys = list(TEST_EXAMPLES.keys())
            for i in range(4):
                with ex_cols1[i]:
                    lbl = ex_keys[i]
                    if st.button(lbl, key=f"ex_btn_{i}", use_container_width=True):
                        st.session_state.doc_input = TEST_EXAMPLES[lbl]
                        st.rerun()

            ex_cols2 = st.columns(4)
            for i in range(4, 8):
                with ex_cols2[i - 4]:
                    lbl = ex_keys[i]
                    if st.button(lbl, key=f"ex_btn_{i}", use_container_width=True):
                        st.session_state.doc_input = TEST_EXAMPLES[lbl]
                        st.rerun()

            ex_cols3 = st.columns(3)
            for i in range(8, 11):
                with ex_cols3[i - 8]:
                    lbl = ex_keys[i]
                    if st.button(lbl, key=f"ex_btn_{i}", use_container_width=True):
                        st.session_state.doc_input = TEST_EXAMPLES[lbl]
                        st.rerun()

            col_sp1, col_sp2 = st.columns([1, 1])
            with col_sp1:
                if st.button("🔋 Battery (OOD)", key="ex_btn_11", use_container_width=True):
                    st.session_state.doc_input = TEST_EXAMPLES["🔋 Out of Domain: Battery"]
                    st.rerun()
            with col_sp2:
                if st.button("⚠️ Ambiguous Benchmark", type="primary", key="ex_btn_13", use_container_width=True):
                    st.session_state.doc_input = TEST_EXAMPLES["⚠️ Ambiguous Benchmark"]
                    st.rerun()

            st.write("")

            # Input Text Area
            input_text = st.text_area(
                "Document Text Area",
                value=st.session_state.doc_input,
                height=180,
                placeholder="Paste the document text to classify...",
                label_visibility="collapsed"
            )
            st.session_state.doc_input = input_text

            words_cnt = len(input_text.split()) if input_text.strip() else 0
            chars_cnt = len(input_text)
            st.markdown(f'''
            <div style="display:flex;justify-content:space-between;font-size:11.5px;color:#64748b;font-family:var(--font-mono);padding:4px 0 10px;">
                <span>{words_cnt:,} words &bull; {chars_cnt:,} characters</span>
                <span style="color:#818cf8;">4 Target Classes + OOD Detection</span>
            </div>
            ''', unsafe_allow_html=True)

            c_btn1, c_btn2 = st.columns([1, 2])
            with c_btn1:
                if st.button("Clear Input", use_container_width=True):
                    st.session_state.doc_input = ""
                    st.session_state.last_prediction = None
                    st.rerun()
            with c_btn2:
                analyze_clicked = st.button("Analyze Document →", type="primary", use_container_width=True)

            st.markdown('</div>', unsafe_allow_html=True)

        # RIGHT COLUMN: AI PREDICTION
        with col_output:
            st.markdown('''
            <div class="tc-card">
                <div class="tc-card-header">
                    <div>
                        <div class="tc-card-title">🤖 AI Prediction & Confidence Analysis</div>
                        <div class="tc-card-sub">Real-time inference with ambiguity and out-of-domain detection</div>
                    </div>
                </div>
            ''', unsafe_allow_html=True)

            # Auto-run if analyzed or preloaded
            if analyze_clicked or (st.session_state.last_prediction is None and input_text.strip()):
                if input_text.strip():
                    with st.spinner("Analyzing document through TF-IDF and calibrated classifier..."):
                        t0 = time.perf_counter()
                        res = classify_document(
                            raw_text=input_text,
                            model=best_model,
                            vectorizer=vectorizer,
                            categories=categories,
                            config=inference_config
                        )
                        res["latency"] = round(time.perf_counter() - t0, 4)
                        st.session_state.last_prediction = res

            if st.session_state.last_prediction:
                res = st.session_state.last_prediction
                status = res.get("status", "normal")
                pred_label = res.get("prediction", "Unknown")
                conf_pct = res.get("confidence", 0.0) * 100
                det_topics = res.get("detected_topics", [])
                top_kws = res.get("top_keywords", [])
                probs = res.get("probabilities", {})
                reason = res.get("reason", "")

                # 1. Prediction Banner according to status
                if status == "normal":
                    icon = CATEGORY_ICONS_4.get(pred_label, "📄")
                    disp_name = CATEGORY_DISPLAY_NAMES_4.get(pred_label, pred_label)
                    banner_cls = "tc-pred-banner-normal"
                    badge_html = f"<span class='tc-badge-normal'>✓ Normal Single-Topic</span>"
                elif status == "ambiguous":
                    icon = CATEGORY_ICONS_4["ambiguous"]
                    disp_name = "Ambiguous / Multi-topic"
                    banner_cls = "tc-pred-banner-ambiguous"
                    badge_html = f"<span class='tc-badge-ambiguous'>⚠️ Ambiguous / Multi-topic</span>"
                else:
                    icon = CATEGORY_ICONS_4["unknown"]
                    disp_name = "Unknown / Out-of-Domain"
                    banner_cls = "tc-pred-banner-unknown"
                    badge_html = f"<span class='tc-badge-unknown'>❓ Unknown / Out-of-Domain</span>"

                st.markdown(f'''
                <div class="{banner_cls}">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                        <span class="tc-pred-kicker">CLASSIFIER DECISION</span>
                        {badge_html}
                    </div>
                    <div class="tc-pred-cat">
                        <span>{icon}</span>
                        <span>{disp_name}</span>
                    </div>
                    <div style="display:flex;align-items:center;gap:16px;font-size:12px;color:#cbd5e1;font-family:var(--font-mono);">
                        <span>Confidence: <b style="color:#ffffff;">{conf_pct:.1f}%</b></span>
                        <span>&bull;</span>
                        <span>Latency: <b>{res.get('latency', 0.002):.3f}s</b></span>
                    </div>
                    <div style="margin-top:8px;font-size:11.5px;color:#94a3b8;line-height:1.4;">
                        Reason: {reason}
                    </div>
                </div>
                ''', unsafe_allow_html=True)

                # 2. Detected Topics
                st.markdown("<div style='font-size:11px;font-weight:700;color:#cbd5e1;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;'>DETECTED TOPICS</div>", unsafe_allow_html=True)
                if det_topics:
                    topics_html = "".join([f"<span class='tc-keyword-tag' style='color:#a5b4fc;'>🏷️ {t}</span> " for t in det_topics])
                    st.markdown(f"<div style='margin-bottom:14px;'>{topics_html}</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div style='color:#64748b;font-size:12px;margin-bottom:14px;'>No dominant domain topics detected.</div>", unsafe_allow_html=True)

                # 3. Clear Explanation Callout (Requirement 11)
                st.markdown('''
                <div class="tc-callout-note">
                    <span>💡</span>
                    <span><b>Explanation:</b> Low-confidence predictions are marked as <b>Unknown</b> rather than being forced into a category. Inputs with signals across multiple classes are marked as <b>Ambiguous</b>.</span>
                </div>
                ''', unsafe_allow_html=True)

                # 4. Class Probability Distribution Chart
                st.markdown("<div style='font-size:11px;font-weight:700;color:#cbd5e1;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:8px;'>CLASS PROBABILITY DISTRIBUTION</div>", unsafe_allow_html=True)
                sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
                for cat_id, p_val in sorted_probs:
                    p_pct = p_val * 100
                    disp_c = CATEGORY_DISPLAY_NAMES_4.get(cat_id, cat_id)
                    c_ico = CATEGORY_ICONS_4.get(cat_id, "📄")
                    is_top_ambiguous = (status == "ambiguous" and p_pct >= 15.0)
                    fill_cls = "tc-conf-fill tc-conf-fill-highlight" if is_top_ambiguous else "tc-conf-fill"

                    st.markdown(f'''
                    <div class="tc-conf-row">
                        <div class="tc-conf-label">
                            <span>{c_ico}</span> {disp_c}
                        </div>
                        <div class="tc-conf-track">
                            <div class="{fill_cls}" style="width: {p_pct:.1f}%;"></div>
                        </div>
                        <div class="tc-conf-pct">{p_pct:.1f}%</div>
                    </div>
                    ''', unsafe_allow_html=True)

                st.write("")

                # 5. Important TF-IDF Features
                st.markdown("<div style='font-size:11px;font-weight:700;color:#cbd5e1;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:8px;'>IMPORTANT TF-IDF FEATURES</div>", unsafe_allow_html=True)
                if top_kws:
                    kws_html = "".join([
                        f"<span class='tc-keyword-tag'>{item['term']} <span class='tc-keyword-wt'>{item['weight']:.3f}</span></span>"
                        for item in top_kws
                    ])
                    st.markdown(f"<div class='tc-tag-wrap'>{kws_html}</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div style='color:#64748b;font-size:12px;'>No vocabulary terms matched the trained domain features.</div>", unsafe_allow_html=True)

            else:
                st.markdown('''
                <div style="text-align:center;padding:48px 20px;border:1px dashed rgba(255,255,255,0.1);border-radius:14px;color:#64748b;">
                    <div style="font-size:32px;margin-bottom:8px;">🤖</div>
                    <div style="font-size:14px;color:#e2e8f0;font-weight:600;">Ready to analyze</div>
                    <div style="font-size:12px;margin-top:4px;">Enter text or pick a test example on the left.</div>
                </div>
                ''', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # ROUTE 2: CSV BATCH CLASSIFICATION (Requirement 12)
    # -------------------------------------------------------------------------
    elif st.session_state.active_nav == "CSV Batch":
        st.markdown('''
        <div class="tc-card">
            <div class="tc-card-header">
                <div>
                    <div class="tc-card-title">⚡ CSV Batch Document Classification</div>
                    <div class="tc-card-sub">Upload a CSV dataset. Every row is detected and processed independently.</div>
                </div>
            </div>
        ''', unsafe_allow_html=True)

        uploaded_csv = st.file_uploader(
            "Upload CSV File",
            type=["csv"],
            key="csv_batch_uploader",
            label_visibility="collapsed"
        )

        # Quick CSV Download template
        sample_csv_data = (
            "id,text\n"
            "1,NASA launched a spacecraft into orbit.\n"
            "2,The starting pitcher struck out nine batters.\n"
            "3,The government passed a new law.\n"
            "4,The computer generated a 3D image.\n"
            "5,I ate pizza today.\n"
            "6,My laptop battery is low.\n"
            "7,The Chief Minister bought a Royal Enfield bike and went to Mars to see Jesus and play football.\n"
        )
        st.download_button(
            "📥 Download Sample Test CSV Template",
            data=sample_csv_data,
            file_name="sample_text_batch.csv",
            mime="text/csv"
        )

        if uploaded_csv is not None:
            try:
                raw_bytes = uploaded_csv.read()
                df_raw = pd.read_csv(io.BytesIO(raw_bytes))

                # Autodetect text column (Requirement 12)
                possible_cols = ['text', 'document', 'content', 'message', 'sentence', 'body', 'doc']
                detected_col = None
                for c in df_raw.columns:
                    if str(c).lower().strip() in possible_cols:
                        detected_col = c
                        break

                if detected_col is None:
                    # Look for first string column
                    for c in df_raw.columns:
                        if df_raw[c].dtype == object:
                            detected_col = c
                            break

                if detected_col is None:
                    detected_col = df_raw.columns[0]

                st.success(f"✅ Loaded `{uploaded_csv.name}` ({len(df_raw)} rows). Autodetected text column: **`{detected_col}`**")

                if st.button("🚀 Classify All Rows", type="primary"):
                    progress_bar = st.progress(0.0)
                    results = []

                    t_batch_start = time.perf_counter()
                    for idx, row in df_raw.iterrows():
                        row_val = str(row[detected_col]) if pd.notna(row[detected_col]) else ""
                        row_id = row.get("id", idx + 1)

                        res = classify_document(
                            raw_text=row_val,
                            model=best_model,
                            vectorizer=vectorizer,
                            categories=categories,
                            config=inference_config
                        )

                        results.append({
                            "id": row_id,
                            "text": row_val,
                            "predicted_category": res["prediction"],
                            "confidence": f"{res['confidence'] * 100:.1f}%",
                            "status": res["status"],
                            "detected_topics": ", ".join(res["detected_topics"])
                        })
                        progress_bar.progress((idx + 1) / len(df_raw))

                    elapsed_batch = time.perf_counter() - t_batch_start
                    st.session_state.batch_df = pd.DataFrame(results)
                    st.toast(f"Classified {len(results)} rows in {elapsed_batch:.2f}s!")

            except Exception as e:
                st.error(f"Error processing CSV: {e}")

        # Render Batch Results if available
        if st.session_state.batch_df is not None:
            df_res = st.session_state.batch_df
            total_r = len(df_res)
            succ_r = sum(df_res["status"] == "normal")
            ambig_r = sum(df_res["status"] == "ambiguous")
            unk_r = sum(df_res["status"] == "unknown")

            st.write("")
            st.markdown("<div style='font-size:13px;font-weight:750;color:#cbd5e1;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:12px;'>BATCH EXECUTION SUMMARY</div>", unsafe_allow_html=True)
            
            # Summary Metrics (Requirement 12)
            c_s1, c_s2, c_s3, c_s4 = st.columns(4)
            with c_s1:
                st.metric("Total Rows", f"{total_r:,}")
            with c_s2:
                st.metric("Normal Predictions", f"{succ_r:,}", f"{(succ_r/max(total_r,1))*100:.1f}%")
            with c_s3:
                st.metric("Ambiguous Predictions", f"{ambig_r:,}", f"{(ambig_r/max(total_r,1))*100:.1f}%")
            with c_s4:
                st.metric("Unknown Predictions", f"{unk_r:,}", f"{(unk_r/max(total_r,1))*100:.1f}%")

            st.write("")

            # Distribution chart
            col_tbl, col_chart = st.columns([1.3, 0.7], gap="large")
            with col_tbl:
                st.markdown("<div style='font-size:12px;font-weight:700;color:#cbd5e1;margin-bottom:8px;'>Classified Rows Table</div>", unsafe_allow_html=True)
                st.dataframe(df_res, use_container_width=True, hide_index=True)

            with col_chart:
                st.markdown("<div style='font-size:12px;font-weight:700;color:#cbd5e1;margin-bottom:8px;'>Class Distribution Chart</div>", unsafe_allow_html=True)
                pred_counts = df_res["predicted_category"].value_counts()
                st.bar_chart(pred_counts, color="#6366f1")

            # Download classified_results.csv (Requirement 12)
            csv_export = df_res.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download classified_results.csv",
                data=csv_export,
                file_name="classified_results.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True
            )

        st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # ROUTE 3: ANALYTICS (Model Comparison & Confusion Matrix)
    # -------------------------------------------------------------------------
    elif st.session_state.active_nav == "Analytics":
        st.markdown('''
        <div class="tc-card">
            <div class="tc-card-header">
                <div>
                    <div class="tc-card-title">📊 Multi-Model Performance Benchmarks (4 Classes)</div>
                    <div class="tc-card-sub">Evaluation across all 4 machine learning algorithms on identical stratified test partitions</div>
                </div>
            </div>
        ''', unsafe_allow_html=True)

        # 4 Model Benchmark Cards
        cols_models = st.columns(4)
        for i, row in metrics_table.iterrows():
            m_name = row['Model']
            m_acc = row['Accuracy'] * 100
            m_prec = row.get('Precision (Weighted)', 0) * 100
            m_f1 = row.get('F1-Score (Weighted)', 0) * 100
            m_lat = row.get('Latency (ms/doc)', 0.0)
            is_prod = (m_name == best_model_name)

            with cols_models[i % 4]:
                st.markdown(f'''
                <div class="tc-stat-card" style="border-color:{'rgba(99,102,241,0.5)' if is_prod else 'var(--border-subtle)'};background:{'linear-gradient(135deg,rgba(99,102,241,0.1) 0%,rgba(14,20,31,0.8) 100%)' if is_prod else 'var(--bg-card)'};">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                        <span style="font-size:13px;font-weight:750;color:#f8fafc;">{m_name}</span>
                        {'<span style="font-size:9px;font-weight:700;padding:2px 6px;border-radius:4px;background:#6366f1;color:#fff;">PROD</span>' if is_prod else ''}
                    </div>
                    <div style="font-size:24px;font-weight:800;color:#a5b4fc;">{m_acc:.1f}%</div>
                    <div style="font-size:10px;text-transform:uppercase;color:#64748b;margin-bottom:8px;">Test Accuracy</div>
                    <div style="display:flex;justify-content:space-between;font-size:10.5px;color:#94a3b8;border-top:1px solid rgba(255,255,255,0.05);padding-top:8px;">
                        <span>F1: <b>{m_f1:.1f}%</b></span>
                        <span>Latency: <b>{m_lat:.3f}ms</b></span>
                    </div>
                </div>
                ''', unsafe_allow_html=True)

        st.write("")

        col_ch1, col_ch2 = st.columns([1, 1], gap="large")
        with col_ch1:
            st.markdown("<div style='font-size:13px;font-weight:700;color:#cbd5e1;margin-bottom:8px;'>Accuracy Comparison Across Classifiers</div>", unsafe_allow_html=True)
            if os.path.exists("visualizations/model_comparison.png"):
                st.image("visualizations/model_comparison.png", use_container_width=True)

        with col_ch2:
            st.markdown("<div style='font-size:13px;font-weight:700;color:#cbd5e1;margin-bottom:8px;'>Confusion Matrix Diagnostics (4 Classes)</div>", unsafe_allow_html=True)
            if os.path.exists("visualizations/confusion_matrices.png"):
                st.image("visualizations/confusion_matrices.png", use_container_width=True)

        st.write("")
        st.markdown("<div style='font-size:13px;font-weight:700;color:#cbd5e1;margin-bottom:8px;'>Detailed Statistical Metrics Table</div>", unsafe_allow_html=True)
        st.dataframe(metrics_table, use_container_width=True, hide_index=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # ROUTE 4: DATASET
    # -------------------------------------------------------------------------
    elif st.session_state.active_nav == "Dataset":
        st.markdown('''
        <div class="tc-card">
            <div class="tc-card-header">
                <div>
                    <div class="tc-card-title">📁 4-Class Dataset & Feature Distribution</div>
                    <div class="tc-card-sub">Overview of the 4 core categories, augmentations, and TF-IDF feature space</div>
                </div>
            </div>
        ''', unsafe_allow_html=True)

        d_c1, d_c2, d_c3, d_c4 = st.columns(4)
        with d_c1:
            st.metric("Total Documents", f"{stats.get('total_documents', 2769):,}")
        with d_c2:
            st.metric("Training Samples", f"{int(stats.get('total_documents', 2769) * 0.8):,}")
        with d_c3:
            st.metric("Testing Samples", f"{int(stats.get('total_documents', 2769) * 0.2):,}")
        with d_c4:
            st.metric("TF-IDF Features", "5,000")

        st.write("")

        col_dt1, col_dt2 = st.columns([1, 1], gap="large")
        with col_dt1:
            st.markdown("<div style='font-size:13px;font-weight:700;color:#cbd5e1;margin-bottom:8px;'>Class Distribution Across 4 Categories</div>", unsafe_allow_html=True)
            if os.path.exists("visualizations/class_distribution.png"):
                st.image("visualizations/class_distribution.png", use_container_width=True)

        with col_dt2:
            st.markdown("<div style='font-size:13px;font-weight:700;color:#cbd5e1;margin-bottom:8px;'>Top Discriminative Keywords per Category</div>", unsafe_allow_html=True)
            if os.path.exists("visualizations/top_keywords.png"):
                st.image("visualizations/top_keywords.png", use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # ROUTE 5: ABOUT
    # -------------------------------------------------------------------------
    elif st.session_state.active_nav == "About":
        st.markdown('''
        <div class="tc-card">
            <div class="tc-card-header">
                <div>
                    <div class="tc-card-title">ℹ️ About the Upgraded TextClassify AI System</div>
                    <div class="tc-card-sub">Robust 4-class classification with out-of-domain and ambiguity intelligence</div>
                </div>
            </div>

            <div style="font-size:14px;color:#cbd5e1;line-height:1.7;margin-bottom:20px;">
                <b>TextClassify</b> is a robust NLP document classification system designed to handle real-world text complexity.
                Rather than assuming every input belongs perfectly to one of the four categories, the engine distinguishes between:
                <br><br>
                1. <b>In-Domain Known Classes:</b> Computer Graphics, Baseball, Space Science, Politics.<br>
                2. <b>Ambiguous / Multi-Topic:</b> Detects when an input contains strong competing signals across multiple classes.<br>
                3. <b>Unknown / Out-of-Domain:</b> Flags inputs with low confidence or no domain vocabulary overlap.
            </div>

            <div style="font-size:11px;font-weight:700;color:#818cf8;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:10px;">CORE TECHNOLOGIES</div>
            <div style="display:flex;flex-wrap:wrap;gap:8px;">
                <span class="tc-keyword-tag">Python 3.10+</span>
                <span class="tc-keyword-tag">Scikit-Learn</span>
                <span class="tc-keyword-tag">Multinomial Naive Bayes</span>
                <span class="tc-keyword-tag">Logistic Regression</span>
                <span class="tc-keyword-tag">Linear SVM (Calibrated)</span>
                <span class="tc-keyword-tag">Random Forest</span>
                <span class="tc-keyword-tag">Sublinear TF-IDF (1-2 ngrams)</span>
                <span class="tc-keyword-tag">FastAPI / Starlette REST API</span>
                <span class="tc-keyword-tag">Streamlit UI</span>
                <span class="tc-keyword-tag">CSV Batch Processing</span>
            </div>
        </div>
        ''', unsafe_allow_html=True)

    # Global Footer
    st.markdown('''
    <div style="text-align:center;padding:28px 0 10px;font-size:12px;color:#475569;">
        ⚡ TextClassify &bull; 4-Class NLP Classification Platform with Unknown & Ambiguity Detection
    </div>
    ''', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
