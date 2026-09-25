"""
TextClassify - AI-Powered Text Document Classification System (4 Classes)
Production-grade Document Intelligence Interface with Automated Inference,
Unknown / Out-of-Domain Detection, and Multi-Topic Ambiguity Analysis.

Provides 8 Professional Sections:
1. Single Document Classification
2. Batch CSV Classification
3. Model Information & Benchmarks
4. Confidence & Out-of-Domain (OOD) Analysis
5. Model Explainability & Feature Attribution
6. Evaluation Results & Cross-Validation
7. Forensic Error Analysis
8. Dataset Profiling & Validation Audit
"""

import os
import time
import pandas as pd
import streamlit as st

from src.inference_engine import (
    classify_document,
    InferenceConfig,
    DEFAULT_INFERENCE_CONFIG,
    TARGET_4_CLASSES,
    CATEGORY_DISPLAY_NAMES_4,
    CATEGORY_ICONS_4
)
from src.explainability import explain_prediction
from src.model_validator import validate_and_load_artifacts

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
    .tc-pred-banner-low_confidence {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.12) 0%, rgba(14, 20, 31, 0.85) 100%);
        border: 1px solid rgba(239, 68, 68, 0.3);
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
    .tc-badge-low_confidence {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        background: rgba(239, 68, 68, 0.16);
        border: 1px solid rgba(239, 68, 68, 0.4);
        color: #fca5a5;
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
        color: #c7d2fe;
        font-size: 11.5px;
        border-radius: 8px;
        font-family: var(--font-mono);
    }
    .tc-keyword-wt {
        font-size: 10px;
        color: #818cf8;
        font-weight: 700;
    }

    /* Stats Card */
    .tc-stat-card {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 14px;
        padding: 18px;
        backdrop-filter: blur(10px);
    }
    .tc-stat-lbl {
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
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Curated Test Presets
# -----------------------------------------------------------------------------
TEST_EXAMPLES = {
    "🚀 Pure Space": "NASA launched a spacecraft into orbit using advanced cryogenic rocket propulsion.",
    "⚾ Pure Baseball": "The baseball team scored five runs in the bottom of the ninth inning to win the game.",
    "🏛️ Pure Politics": "The government passed a new law regarding federal taxation and civil liberties.",
    "🎨 Pure Graphics": "The computer generated a 3D image using polygon mesh shading and ray tracing.",
    "🍕 OOD: Pizza": "Pizza is my favorite food.",
    "🔋 OOD: Battery": "My laptop battery is low.",
    "🏍️ OOD: Motorcycle": "I bought a motorcycle.",
    "🐍 OOD: Python": "Python is easy to learn.",
    "🎬 OOD: Movie": "The movie was excellent.",
    "🔀 Mixed: Mars & Govt": "The government announced a new Mars mission.",
    "🔀 Mixed: NASA & Baseball": "The baseball team visited NASA.",
    "🔀 Mixed: Football on Mars": "The astronaut played football on Mars.",
    "⚠️ Mixed Benchmark": "The Chief Minister bought a Royal Enfield bike and went to Mars to see Jesus and play football."
}


# -----------------------------------------------------------------------------
# Artifact Loader with Validation
# -----------------------------------------------------------------------------
@st.cache_resource
def load_project_artifacts():
    try:
        best_model, vectorizer, all_models, metadata = validate_and_load_artifacts("models")
        return vectorizer, best_model, all_models, metadata
    except Exception as e:
        st.error(f"⚠️ Model artifact loading error: {e}. Run 'python train.py' to rebuild models.")
        return None, None, None, None


# -----------------------------------------------------------------------------
# Main Application
# -----------------------------------------------------------------------------
def main():
    if 'active_nav' not in st.session_state:
        st.session_state.active_nav = 'Single Classify'
    if 'doc_input' not in st.session_state:
        st.session_state.doc_input = TEST_EXAMPLES["⚠️ Mixed Benchmark"]
    if 'last_prediction' not in st.session_state:
        st.session_state.last_prediction = None
    if 'batch_df' not in st.session_state:
        st.session_state.batch_df = None

    vectorizer, best_model, all_models, metadata = load_project_artifacts()

    if vectorizer is None:
        st.stop()

    categories = metadata.get('categories', TARGET_4_CLASSES)
    metrics_table = pd.DataFrame(metadata.get('metrics_table', []))
    cv_table = pd.DataFrame(metadata.get('cross_validation_results', []))
    stats = metadata.get('dataset_statistics', {})
    best_model_name = metadata.get('best_model_name', 'Logistic Regression')

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
    # STICKY TOP NAVIGATION BAR (8 SECTIONS)
    # -------------------------------------------------------------------------
    col_nav_brand, col_nav_tabs, col_nav_status = st.columns([2.0, 6.5, 1.5], vertical_alignment="center")

    with col_nav_brand:
        st.markdown('''
        <div class="tc-brand">
            <div class="tc-logo-icon">⚡</div>
            <div>
                <div class="tc-brand-title">TextClassify</div>
            </div>
            <div class="tc-brand-pill">v2.0 ML</div>
        </div>
        ''', unsafe_allow_html=True)

    with col_nav_tabs:
        nav_options = [
            "Single Classify",
            "Batch CSV",
            "Model Info",
            "Confidence & OOD",
            "Explainability",
            "Evaluation",
            "Error Analysis",
            "Dataset Stats"
        ]
        current_nav = st.session_state.active_nav if st.session_state.active_nav in nav_options else "Single Classify"
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

    # =========================================================================
    # SECTION 1: SINGLE DOCUMENT CLASSIFICATION
    # =========================================================================
    if st.session_state.active_nav == "Single Classify":
        col_input, col_output = st.columns([1.1, 1], gap="large")

        with col_input:
            st.markdown('''
            <div class="tc-card">
                <div class="tc-card-header">
                    <div>
                        <div class="tc-card-title">📝 Single Document Classification</div>
                        <div class="tc-card-sub">Test inputs across pure topics, short sentences, OOD text, or mixed multi-topic queries.</div>
                    </div>
                </div>
            ''', unsafe_allow_html=True)

            st.markdown("<div style='font-size:11px;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:8px;'>PRESET TEST CASES</div>", unsafe_allow_html=True)

            ex_cols1 = st.columns(4)
            keys_list = list(TEST_EXAMPLES.keys())
            for i in range(4):
                with ex_cols1[i]:
                    lbl = keys_list[i]
                    if st.button(lbl, key=f"ex_btn_{i}", use_container_width=True):
                        st.session_state.doc_input = TEST_EXAMPLES[lbl]
                        st.rerun()

            ex_cols2 = st.columns(5)
            for i in range(4, 9):
                with ex_cols2[i - 4]:
                    lbl = keys_list[i]
                    if st.button(lbl, key=f"ex_btn_{i}", use_container_width=True):
                        st.session_state.doc_input = TEST_EXAMPLES[lbl]
                        st.rerun()

            ex_cols3 = st.columns(4)
            for i in range(9, 13):
                with ex_cols3[i - 9]:
                    lbl = keys_list[i]
                    is_main = (i == 12)
                    if st.button(lbl, key=f"ex_btn_{i}", type="primary" if is_main else "secondary", use_container_width=True):
                        st.session_state.doc_input = TEST_EXAMPLES[lbl]
                        st.rerun()

            st.write("")

            # Select model
            model_options = list(all_models.keys()) if all_models else [best_model_name]
            chosen_model_name = st.selectbox("Classification Model", model_options, index=0)
            active_model = all_models.get(chosen_model_name, best_model)

            # Input Text Area
            input_text = st.text_area(
                "Document Text Area",
                value=st.session_state.doc_input,
                height=160,
                placeholder="Paste the document text to classify...",
                label_visibility="collapsed"
            )
            st.session_state.doc_input = input_text

            words_cnt = len(input_text.split()) if input_text.strip() else 0
            chars_cnt = len(input_text)
            st.markdown(f'''
            <div style="display:flex;justify-content:space-between;font-size:11.5px;color:#64748b;font-family:var(--font-mono);padding:4px 0 10px;">
                <span>{words_cnt:,} words &bull; {chars_cnt:,} characters</span>
                <span style="color:#818cf8;">4 Classes + OOD & Ambiguity Guardrails</span>
            </div>
            ''', unsafe_allow_html=True)

            c_btn1, c_btn2 = st.columns([1, 2])
            with c_btn1:
                if st.button("Clear Input", use_container_width=True):
                    st.session_state.doc_input = ""
                    st.session_state.last_prediction = None
                    st.rerun()
            with c_btn2:
                analyze_clicked = st.button("Classify Document →", type="primary", use_container_width=True)

            st.markdown('</div>', unsafe_allow_html=True)

        with col_output:
            st.markdown('''
            <div class="tc-card">
                <div class="tc-card-header">
                    <div>
                        <div class="tc-card-title">🤖 Prediction & Decision Analysis</div>
                        <div class="tc-card-sub">Inference output with confidence calibration and topical breakdown</div>
                    </div>
                </div>
            ''', unsafe_allow_html=True)

            if analyze_clicked or (st.session_state.last_prediction is None and input_text.strip()):
                if input_text.strip():
                    with st.spinner("Classifying text..."):
                        t0 = time.perf_counter()
                        res = classify_document(
                            raw_text=input_text,
                            model=active_model,
                            vectorizer=vectorizer,
                            categories=categories,
                            config=inference_config
                        )
                        res["latency"] = round(time.perf_counter() - t0, 4)
                        res["model_name"] = chosen_model_name
                        st.session_state.last_prediction = res

            if st.session_state.last_prediction:
                res = st.session_state.last_prediction
                status = res.get("status", "normal")
                pred_label = res.get("prediction", "Unknown")
                conf_pct = res.get("confidence", 0.0) * 100
                top_kws = res.get("top_keywords", [])
                probs = res.get("probabilities", {})
                reason = res.get("reason", "")

                if status == "normal":
                    icon = CATEGORY_ICONS_4.get(pred_label, "📄")
                    disp_name = CATEGORY_DISPLAY_NAMES_4.get(pred_label, pred_label)
                    banner_cls = "tc-pred-banner-normal"
                    badge_html = "<span class='tc-badge-normal'>✓ NORMAL</span>"
                elif status == "ambiguous":
                    icon = CATEGORY_ICONS_4["ambiguous"]
                    disp_name = "Ambiguous / Multi-topic"
                    banner_cls = "tc-pred-banner-ambiguous"
                    badge_html = "<span class='tc-badge-ambiguous'>⚠️ AMBIGUOUS</span>"
                elif status == "low_confidence":
                    icon = CATEGORY_ICONS_4.get(pred_label, "⚡")
                    disp_name = f"{CATEGORY_DISPLAY_NAMES_4.get(pred_label, pred_label)} (Uncertain)"
                    banner_cls = "tc-pred-banner-low_confidence"
                    badge_html = "<span class='tc-badge-low_confidence'>⚡ LOW CONFIDENCE</span>"
                else:
                    icon = CATEGORY_ICONS_4["unknown"]
                    disp_name = "Unknown / Out-of-Domain"
                    banner_cls = "tc-pred-banner-unknown"
                    badge_html = "<span class='tc-badge-unknown'>❓ UNKNOWN (OOD)</span>"

                st.markdown(f'''
                <div class="{banner_cls}">
                    <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                        <div>
                            <div class="tc-pred-kicker">PREDICTED STATUS</div>
                            <div class="tc-pred-cat">{icon} {disp_name}</div>
                        </div>
                        <div>{badge_html}</div>
                    </div>
                    <div style="display:flex;gap:20px;margin-top:8px;font-size:12px;font-family:var(--font-mono);color:#cbd5e1;">
                        <span>Confidence: <b style="color:#ffffff;">{conf_pct:.1f}%</b></span>
                        <span>Latency: <b>{res.get("latency", 0.001)*1000:.2f} ms</b></span>
                        <span>Model: <b>{res.get("model_name", best_model_name)}</b></span>
                    </div>
                </div>
                ''', unsafe_allow_html=True)

                if reason:
                    st.markdown(f"<div class='tc-callout-note'>💡 <b>Decision Logic:</b> {reason}</div>", unsafe_allow_html=True)

                st.markdown("<div style='font-size:11px;font-weight:700;color:#cbd5e1;text-transform:uppercase;letter-spacing:0.06em;margin:12px 0 8px;'>CLASS PROBABILITIES</div>", unsafe_allow_html=True)
                for cat_id in categories:
                    p_val = probs.get(cat_id, 0.0)
                    p_pct = p_val * 100
                    disp_c = CATEGORY_DISPLAY_NAMES_4.get(cat_id, cat_id)
                    c_ico = CATEGORY_ICONS_4.get(cat_id, "📄")
                    is_top_ambiguous = (status == "ambiguous" and p_pct >= 15.0)
                    fill_cls = "tc-conf-fill tc-conf-fill-highlight" if is_top_ambiguous else "tc-conf-fill"

                    st.markdown(f'''
                    <div class="tc-conf-row">
                        <div class="tc-conf-label"><span>{c_ico}</span> {disp_c}</div>
                        <div class="tc-conf-track"><div class="{fill_cls}" style="width: {p_pct:.1f}%;"></div></div>
                        <div class="tc-conf-pct">{p_pct:.1f}%</div>
                    </div>
                    ''', unsafe_allow_html=True)

                st.markdown("<div style='font-size:11px;font-weight:700;color:#cbd5e1;text-transform:uppercase;letter-spacing:0.06em;margin:14px 0 8px;'>ACTIVE TF-IDF FEATURES</div>", unsafe_allow_html=True)
                if top_kws:
                    kws_html = "".join([
                        f"<span class='tc-keyword-tag'>{item['term']} <span class='tc-keyword-wt'>{item['weight']:.3f}</span></span>"
                        for item in top_kws
                    ])
                    st.markdown(f"<div class='tc-tag-wrap'>{kws_html}</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div style='color:#64748b;font-size:12px;'>No domain vocabulary terms detected in input.</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='text-align:center;padding:48px 20px;color:#64748b;'>Enter text or click a test button to run classification.</div>", unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

    # =========================================================================
    # SECTION 2: BATCH CSV CLASSIFICATION
    # =========================================================================
    elif st.session_state.active_nav == "Batch CSV":
        st.markdown('''
        <div class="tc-card">
            <div class="tc-card-header">
                <div>
                    <div class="tc-card-title">⚡ Batch CSV Document Classification</div>
                    <div class="tc-card-sub">Upload a CSV file. Columns ('text', 'document', 'content', 'message', 'sentence') are auto-detected and classified row-by-row.</div>
                </div>
            </div>
        ''', unsafe_allow_html=True)

        sample_csv_data = (
            "id,text\n"
            "1,NASA launched a spacecraft into orbit.\n"
            "2,The starting pitcher struck out nine batters in the baseball game.\n"
            "3,The government passed a new law regarding federal taxation.\n"
            "4,The computer generated a 3D image using polygon ray tracing.\n"
            "5,Pizza is my favorite food.\n"
            "6,My laptop battery is low.\n"
            "7,The Chief Minister bought a Royal Enfield bike and went to Mars to see Jesus and play football.\n"
        )
        st.download_button(
            "📥 Download Sample Test CSV Template",
            data=sample_csv_data,
            file_name="sample_documents.csv",
            mime="text/csv"
        )

        uploaded_csv = st.file_uploader("Upload CSV File", type=["csv"], key="csv_batch_uploader", label_visibility="collapsed")

        if uploaded_csv is not None:
            try:
                df_raw = pd.read_csv(uploaded_csv)
                possible_cols = ['text', 'document', 'content', 'message', 'sentence', 'body', 'doc']
                detected_col = None
                for c in df_raw.columns:
                    if str(c).lower().strip() in possible_cols:
                        detected_col = c
                        break
                if detected_col is None:
                    for c in df_raw.columns:
                        if df_raw[c].dtype == object:
                            detected_col = c
                            break

                if detected_col is None:
                    st.error("Could not find a text column in CSV. Expected one of: 'text', 'document', 'content', 'message', 'sentence'.")
                else:
                    st.info(f"Auto-detected text column: **'{detected_col}'** across {len(df_raw)} rows.")
                    rows_out = []
                    prog_bar = st.progress(0)

                    for idx, row in df_raw.iterrows():
                        raw_val = str(row[detected_col]) if pd.notna(row[detected_col]) else ""
                        c_id = row.get("id", idx + 1)
                        res = classify_document(raw_val, best_model, vectorizer, categories, inference_config)
                        top_words = ", ".join([kw["term"] for kw in res["top_keywords"][:3]])
                        rows_out.append({
                            "id": c_id,
                            "text": raw_val[:100] + ("..." if len(raw_val) > 100 else ""),
                            "predicted_category": res["prediction"],
                            "confidence": round(res["confidence"] * 100, 1),
                            "status": res["status"],
                            "top_keywords": top_words
                        })
                        prog_bar.progress((idx + 1) / len(df_raw))

                    df_batch_res = pd.DataFrame(rows_out)
                    tot = len(df_batch_res)
                    norm_c = sum(df_batch_res['status'] == 'normal')
                    amb_c = sum(df_batch_res['status'] == 'ambiguous')
                    unk_c = sum(df_batch_res['status'] == 'unknown')
                    low_c = sum(df_batch_res['status'] == 'low_confidence')

                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric("Total Rows", f"{tot:,}")
                    c2.metric("Normal", f"{norm_c:,}", f"{norm_c/tot*100:.1f}%")
                    c3.metric("Ambiguous", f"{amb_c:,}", f"{amb_c/tot*100:.1f}%")
                    c4.metric("Low Conf", f"{low_c:,}", f"{low_c/tot*100:.1f}%")
                    c5.metric("Unknown (OOD)", f"{unk_c:,}", f"{unk_c/tot*100:.1f}%")

                    st.dataframe(df_batch_res, use_container_width=True, hide_index=True)

                    csv_exp = df_batch_res.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download classified_results.csv",
                        data=csv_exp,
                        file_name="classified_results.csv",
                        mime="text/csv",
                        type="primary"
                    )
            except Exception as e:
                st.error(f"Error processing CSV: {e}")

        st.markdown('</div>', unsafe_allow_html=True)

    # =========================================================================
    # SECTION 3: MODEL INFORMATION
    # =========================================================================
    elif st.session_state.active_nav == "Model Info":
        st.markdown('''
        <div class="tc-card">
            <div class="tc-card-header">
                <div>
                    <div class="tc-card-title">🏆 Trained Classifiers & Hyperparameter Configurations</div>
                    <div class="tc-card-sub">Comparison across 7 algorithms tuned via 5-Fold GridSearchCV on training data</div>
                </div>
            </div>
        ''', unsafe_allow_html=True)

        st.dataframe(metrics_table, use_container_width=True, hide_index=True)

        st.write("")
        st.markdown("<div style='font-size:13px;font-weight:700;color:#cbd5e1;margin-bottom:8px;'>DISCOVERED OPTIMAL HYPERPARAMETERS</div>", unsafe_allow_html=True)
        hparams = metadata.get("hyperparameters", {})
        h_records = [{"Model": m, "Best Parameters": str(p)} for m, p in hparams.items()]
        st.dataframe(pd.DataFrame(h_records), use_container_width=True, hide_index=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # =========================================================================
    # SECTION 4: CONFIDENCE & OOD ANALYSIS
    # =========================================================================
    elif st.session_state.active_nav == "Confidence & OOD":
        st.markdown('''
        <div class="tc-card">
            <div class="tc-card-header">
                <div>
                    <div class="tc-card-title">🎯 Prediction Confidence & Out-of-Domain Guardrails</div>
                    <div class="tc-card-sub">Mechanisms preventing uncalibrated predictions on out-of-distribution texts</div>
                </div>
            </div>
        ''', unsafe_allow_html=True)

        col_c1, col_c2 = st.columns([1, 1], gap="large")
        with col_c1:
            st.markdown("<div style='font-size:13px;font-weight:700;color:#cbd5e1;margin-bottom:8px;'>Confidence Thresholding Mechanism</div>", unsafe_allow_html=True)
            st.markdown(f"""
            - **Current Confidence Threshold:** `{inference_config.confidence_threshold:.2f}` (58%)
            - **Minimum Active TF-IDF Overlap:** `{inference_config.min_active_tfidf:.2f}`
            - **Ambiguity Margin:** `{inference_config.ambiguity_margin:.2f}`
            - **Status Hierarchy:**
              1. **UNKNOWN**: Triggered when active vocabulary weight is below `{inference_config.min_active_tfidf}` or input lacks domain keywords.
              2. **AMBIGUOUS**: Triggered when 2+ classes exhibit significant probability mass ($p_1, p_2 \\ge {inference_config.topic_detection_threshold:.2f}$, gap $\\le {inference_config.ambiguity_margin:.2f}$).
              3. **LOW_CONFIDENCE**: In-domain text with active words, but top probability is under `{inference_config.confidence_threshold:.2f}`.
              4. **NORMAL**: Confident identification of a single dominant topic.
            """)

            st.markdown("<div style='font-size:13px;font-weight:700;color:#ef4444;margin-top:14px;margin-bottom:6px;'>⚠️ Limitations of Confidence-Based OOD Detection</div>", unsafe_allow_html=True)
            st.markdown("""
            1. **Softmax Normalization Overconfidence:** Probabilities are constrained to sum to 1.0. Linear models can output spuriously high confidence on out-of-domain inputs if an unrelated document contains an in-domain n-gram by chance.
            2. **Vocabulary Overlap Blindness:** TF-IDF cannot detect semantic novelty for words not in the training vocabulary.
            3. **Closed-World Assumption:** The model is trained on 4 specific domains and cannot replace genuine zero-shot semantic foundation models.
            """)

        with col_c2:
            st.markdown("<div style='font-size:13px;font-weight:700;color:#cbd5e1;margin-bottom:8px;'>Confidence Distribution on Test Set</div>", unsafe_allow_html=True)
            if os.path.exists("visualizations/confidence_distribution.png"):
                st.image("visualizations/confidence_distribution.png", use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # =========================================================================
    # SECTION 5: MODEL EXPLAINABILITY
    # =========================================================================
    elif st.session_state.active_nav == "Explainability":
        st.markdown('''
        <div class="tc-card">
            <div class="tc-card-header">
                <div>
                    <div class="tc-card-title">🔍 Feature Attribution & Prediction Explainability</div>
                    <div class="tc-card-sub">Inspect how linear model coefficients and TF-IDF terms contribute to decisions</div>
                </div>
            </div>
        ''', unsafe_allow_html=True)

        exp_input = st.text_input("Document to Explain", value="The baseball pitcher recorded twelve strikeouts and allowed zero runs.")
        if st.button("Generate Explanation", type="primary"):
            exp_res = explain_prediction(exp_input, best_model, vectorizer, categories)

            e_col1, e_col2 = st.columns([1, 1], gap="large")
            with e_col1:
                st.markdown(f"**Predicted Category:** `{exp_res['predicted_class']}` ({exp_res['confidence']*100:.1f}%)")
                if exp_res.get('runner_up_class'):
                    st.markdown(f"**Runner-Up Category:** `{exp_res['runner_up_class']}`")

                st.markdown("<div style='font-size:12px;font-weight:700;color:#cbd5e1;margin:12px 0 6px;'>WORDS SUPPORTING PREDICTION</div>", unsafe_allow_html=True)
                for w_info in exp_res.get("top_contributing_words", []):
                    st.markdown(f"- **{w_info['word']}** (score: `{w_info['contribution_score']}`) -> *{w_info['direction']}*")

            with e_col2:
                st.markdown("<div style='font-size:12px;font-weight:700;color:#cbd5e1;margin-bottom:6px;'>ACTIVE TF-IDF TERMS</div>", unsafe_allow_html=True)
                for t_info in exp_res.get("active_terms", []):
                    st.markdown(f"- **{t_info['term']}**: weight = `{t_info['tfidf_weight']}`")

                st.markdown(f"<div class='tc-callout-note'>{exp_res['disclaimer']}</div>", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # =========================================================================
    # SECTION 6: EVALUATION RESULTS
    # =========================================================================
    elif st.session_state.active_nav == "Evaluation":
        st.markdown('''
        <div class="tc-card">
            <div class="tc-card-header">
                <div>
                    <div class="tc-card-title">📈 Cross-Validation & Untouched Test Set Evaluation</div>
                    <div class="tc-card-sub">Strict separation between training cross-validation and final untouched test set</div>
                </div>
            </div>
        ''', unsafe_allow_html=True)

        st.markdown("<div style='font-size:13px;font-weight:700;color:#cbd5e1;margin-bottom:8px;'>5-FOLD STRATIFIED CROSS-VALIDATION (TRAINING DATA ONLY)</div>", unsafe_allow_html=True)
        st.dataframe(cv_table, use_container_width=True, hide_index=True)

        st.write("")
        st.markdown("<div style='font-size:13px;font-weight:700;color:#cbd5e1;margin-bottom:8px;'>UNTOUCHED TEST SET METRICS (547 SAMPLES)</div>", unsafe_allow_html=True)
        st.dataframe(metrics_table, use_container_width=True, hide_index=True)

        col_ev1, col_ev2 = st.columns([1, 1], gap="large")
        with col_ev1:
            if os.path.exists("visualizations/model_comparison.png"):
                st.image("visualizations/model_comparison.png", use_container_width=True)
        with col_ev2:
            if os.path.exists("visualizations/per_class_f1.png"):
                st.image("visualizations/per_class_f1.png", use_container_width=True)

        if os.path.exists("visualizations/confusion_matrices.png"):
            st.image("visualizations/confusion_matrices.png", use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # =========================================================================
    # SECTION 7: ERROR ANALYSIS
    # =========================================================================
    elif st.session_state.active_nav == "Error Analysis":
        st.markdown('''
        <div class="tc-card">
            <div class="tc-card-header">
                <div>
                    <div class="tc-card-title">🔬 Forensic Error Analysis & Confused Classes</div>
                    <div class="tc-card-sub">Systematic breakdown of misclassifications, confusion pairs, and length-based failure modes</div>
                </div>
            </div>
        ''', unsafe_allow_html=True)

        err_summary = metadata.get("error_analysis_summary", {})
        e_c1, e_c2, e_c3 = st.columns(3)
        e_c1.metric("Total Test Samples", f"{err_summary.get('total_test_samples', 547):,}")
        e_c2.metric("Total Errors", f"{err_summary.get('total_errors', 75):,}")
        e_c3.metric("Overall Error Rate", f"{err_summary.get('overall_error_rate', 0.1371)*100:.2f}%")

        col_err1, col_err2 = st.columns([1, 1], gap="large")
        with col_err1:
            st.markdown("<div style='font-size:13px;font-weight:700;color:#cbd5e1;margin-bottom:8px;'>FREQUENTLY CONFUSED CLASS PAIRS</div>", unsafe_allow_html=True)
            conf_pairs = err_summary.get("confused_class_pairs", [])
            st.dataframe(pd.DataFrame(conf_pairs), use_container_width=True, hide_index=True)

        with col_err2:
            st.markdown("<div style='font-size:13px;font-weight:700;color:#cbd5e1;margin-bottom:8px;'>ERROR RATE BY DOCUMENT LENGTH SLICE</div>", unsafe_allow_html=True)
            length_slices = err_summary.get("length_slice_metrics", {})
            length_records = [
                {
                    "Slice": k,
                    "Total Docs": v["total_documents"],
                    "Errors": v["error_count"],
                    "Error Rate": f"{v['error_rate']*100:.1f}%"
                }
                for k, v in length_slices.items()
            ]
            st.dataframe(pd.DataFrame(length_records), use_container_width=True, hide_index=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # =========================================================================
    # SECTION 8: DATASET STATISTICS & AUDIT REPORT
    # =========================================================================
    elif st.session_state.active_nav == "Dataset Stats":
        st.markdown('''
        <div class="tc-card">
            <div class="tc-card-header">
                <div>
                    <div class="tc-card-title">📊 Dataset Statistics, Provenance & Validation Audit</div>
                    <div class="tc-card-sub">Complete data validation log, text profiling, and cryptographic fingerprinting</div>
                </div>
            </div>
        ''', unsafe_allow_html=True)

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Total Cleaned Documents", f"{stats.get('total_documents', 2744):,}")
        d2.metric("SHA-256 Fingerprint", stats.get('fingerprint', '15c0a10bf2fa2b17'))
        d3.metric("Vocabulary Size", f"{stats.get('vocabulary_size', 34076):,}")
        d4.metric("Mean Word Count", f"{stats.get('word_count_mean', 194.9)}")

        st.write("")
        st.markdown("<div style='font-size:13px;font-weight:700;color:#cbd5e1;margin-bottom:8px;'>DATA PIPELINE AUDIT REPORT (WHAT WAS REMOVED & WHY)</div>", unsafe_allow_html=True)
        audit_rep = metadata.get("data_audit_report", {})
        for item in audit_rep.get("audit_log", []):
            st.markdown(f"- 🧹 {item}")

        st.write("")
        col_ds1, col_ds2 = st.columns([1, 1], gap="large")
        with col_ds1:
            if os.path.exists("visualizations/class_distribution.png"):
                st.image("visualizations/class_distribution.png", use_container_width=True)
        with col_ds2:
            if os.path.exists("visualizations/top_keywords.png"):
                st.image("visualizations/top_keywords.png", use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # Global Footer
    st.markdown('''
    <div style="text-align:center;padding:28px 0 10px;font-size:12px;color:#475569;">
        ⚡ TextClassify &bull; 4-Class NLP Classification Platform with Unknown & Ambiguity Detection
    </div>
    ''', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
