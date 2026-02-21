"""
BotTrainer — Streamlit Web Application.
==========================================
Production-grade NLU demo with polished UI, dataset explorer,
model evaluation dashboard, and system settings.
"""

import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so `modules` can be imported
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.entity_extractor import EntityExtractor
from modules.evaluator import Evaluator
from modules.intent_classifier import IntentClassifier
from modules.llm_client import GeminiClient
from modules.utils import load_json_file, setup_logging

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="BotTrainer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Production-grade CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ── Import Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global ── */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    [data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        padding: 8px 12px;
        border-radius: 8px;
        transition: all 0.2s ease;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(255,255,255,0.08);
    }

    /* ── Hero / Page Headers ── */
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 4px;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #8b95a5;
        margin-bottom: 28px;
        font-weight: 400;
    }

    /* ── Glass Card ── */
    .glass-card {
        background: rgba(255,255,255,0.03);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 16px;
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(102,126,234,0.3);
        box-shadow: 0 8px 32px rgba(102,126,234,0.1);
    }

    /* ── Stat Cards ── */
    .stat-card {
        background: linear-gradient(135deg, rgba(102,126,234,0.08), rgba(118,75,162,0.08));
        border: 1px solid rgba(102,126,234,0.15);
        border-radius: 16px;
        padding: 24px 20px;
        text-align: center;
        transition: all 0.3s ease;
    }
    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(102,126,234,0.15);
        border-color: rgba(102,126,234,0.3);
    }
    .stat-value {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.2;
    }
    .stat-label {
        font-size: 0.82rem;
        color: #8b95a5;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 600;
        margin-top: 6px;
    }

    /* ── Intent Badge (large) ── */
    .intent-chip {
        display: inline-block;
        padding: 10px 24px;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1.15rem;
        color: #fff;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        box-shadow: 0 4px 15px rgba(102,126,234,0.35);
        letter-spacing: 0.3px;
    }

    /* ── Confidence Gauge ── */
    .confidence-ring {
        display: inline-flex;
        align-items: center;
        gap: 10px;
        margin-top: 12px;
    }
    .confidence-pct {
        font-size: 2rem;
        font-weight: 800;
        color: #667eea;
    }
    .confidence-label {
        font-size: 0.8rem;
        color: #8b95a5;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* ── Entity Table ── */
    .entity-row {
        display: flex;
        align-items: center;
        padding: 10px 16px;
        border-radius: 10px;
        margin-bottom: 6px;
        background: rgba(102,126,234,0.06);
        border-left: 3px solid #667eea;
        transition: all 0.2s ease;
    }
    .entity-row:hover {
        background: rgba(102,126,234,0.12);
    }
    .entity-key {
        font-weight: 600;
        color: #667eea;
        min-width: 120px;
        font-size: 0.9rem;
    }
    .entity-val {
        font-weight: 500;
        color: #e0e0e0;
        font-size: 0.95rem;
    }

    /* ── Reasoning Box ── */
    .reasoning-box {
        background: linear-gradient(135deg, rgba(46,204,113,0.06), rgba(39,174,96,0.06));
        border: 1px solid rgba(46,204,113,0.2);
        border-radius: 12px;
        padding: 16px 20px;
        font-size: 0.95rem;
        line-height: 1.6;
        color: #c8d6e5;
    }
    .reasoning-box .reasoning-icon {
        font-size: 1.2rem;
        margin-right: 8px;
    }

    /* ── Latency Badge ── */
    .latency-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    .latency-fast { background: rgba(46,204,113,0.12); color: #2ecc71; border: 1px solid rgba(46,204,113,0.25); }
    .latency-mid { background: rgba(243,156,18,0.12); color: #f39c12; border: 1px solid rgba(243,156,18,0.25); }
    .latency-slow { background: rgba(231,76,60,0.12); color: #e74c3c; border: 1px solid rgba(231,76,60,0.25); }

    /* ── History Cards ── */
    .history-card {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        transition: all 0.2s ease;
    }
    .history-card:hover {
        border-color: rgba(102,126,234,0.25);
        background: rgba(102,126,234,0.04);
    }
    .history-intent {
        font-weight: 600;
        color: #667eea;
        font-size: 0.9rem;
    }
    .history-msg {
        color: #8b95a5;
        font-size: 0.85rem;
        max-width: 400px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .history-meta {
        display: flex;
        gap: 12px;
        align-items: center;
        font-size: 0.8rem;
        color: #6b7280;
    }

    /* ── Progress Bar Override ── */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2) !important;
        border-radius: 10px;
    }

    /* ── Metric Overrides ── */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px;
        padding: 18px;
        border-left: 4px solid #667eea;
    }
    [data-testid="stMetric"] label {
        color: #8b95a5 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-weight: 800 !important;
        color: #667eea !important;
    }

    /* ── Button Overrides ── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 32px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        letter-spacing: 0.3px;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102,126,234,0.3) !important;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 6px 25px rgba(102,126,234,0.45) !important;
        transform: translateY(-1px);
    }

    /* ── Tab Overrides ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255,255,255,0.02);
        border-radius: 12px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 0.95rem;
    }

    /* ── Divider ── */
    .pro-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(102,126,234,0.3), transparent);
        margin: 32px 0;
        border: none;
    }

    /* ── Settings Card ── */
    .settings-section {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 20px;
    }
    .settings-title {
        font-weight: 700;
        font-size: 1.05rem;
        color: #e0e0e0;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 6px;
    }
    .status-ok { background: #2ecc71; box-shadow: 0 0 8px rgba(46,204,113,0.5); }
    .status-err { background: #e74c3c; box-shadow: 0 0 8px rgba(231,76,60,0.5); }

    /* ── Sidebar Branding ── */
    .sidebar-brand {
        text-align: center;
        padding: 20px 0 10px 0;
    }
    .sidebar-brand-title {
        font-size: 1.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .sidebar-brand-sub {
        font-size: 0.75rem;
        color: #8b95a5 !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-top: 2px;
    }
    .sidebar-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        margin: 16px 0;
    }
    .sidebar-footer {
        position: fixed;
        bottom: 20px;
        left: 16px;
        font-size: 0.72rem;
        color: #6b7280 !important;
        letter-spacing: 0.5px;
    }

    /* ── Hide Streamlit defaults ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
INTENTS_PATH: Path = PROJECT_ROOT / "data" / "intents.json"
EVAL_DATASET_PATH: Path = PROJECT_ROOT / "data" / "eval_dataset.json"


# ---------------------------------------------------------------------------
# Cached initialisation helpers
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="⚙️ Initializing Gemini client …")
def _get_llm_client() -> GeminiClient:
    return GeminiClient()


@st.cache_resource(show_spinner="⚙️ Building intent classifier …")
def _get_classifier(_llm: GeminiClient, n_few_shot: int = 5) -> IntentClassifier:
    return IntentClassifier(
        intents_json_path=str(INTENTS_PATH),
        llm_client=_llm,
        n_few_shot=n_few_shot,
    )


@st.cache_resource(show_spinner="⚙️ Loading entity extractor …")
def _get_extractor(_llm: GeminiClient) -> EntityExtractor:
    return EntityExtractor(llm_client=_llm)


@st.cache_data(show_spinner="Loading intents …")
def _load_intents() -> Optional[dict]:
    return load_json_file(INTENTS_PATH)


@st.cache_data(show_spinner="Loading eval dataset …")
def _load_eval_dataset() -> Optional[dict]:
    return load_json_file(EVAL_DATASET_PATH)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "predictions" not in st.session_state:
    st.session_state.predictions = []
if "n_few_shot" not in st.session_state:
    st.session_state.n_few_shot = 5
if "eval_results" not in st.session_state:
    st.session_state.eval_results = None


# ---------------------------------------------------------------------------
# Helper: Plotly theme
# ---------------------------------------------------------------------------
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#c8d6e5"),
    margin=dict(t=40, b=40, l=40, r=20),
    xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
)


# =========================================================================
# SIDEBAR
# =========================================================================
with st.sidebar:
    st.markdown(
        '<div class="sidebar-brand">'
        '<div class="sidebar-brand-title">🤖 BotTrainer</div>'
        '<div class="sidebar-brand-sub">NLU Engine</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        options=[
            "🏠 Live Demo",
            "📊 Dataset Explorer",
            "🧪 Evaluation",
            "⚙️ Settings",
        ],
        label_visibility="collapsed",
    )

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # Model selector (always visible)
    model_choice = st.selectbox(
        "🧠 Model",
        ["gemini-2.5-flash", "gemini-2.5-flash-lite"],
        index=0,
    )

    st.markdown(
        '<div class="sidebar-footer">'
        'Powered by Google Gemini · v1.0'
        '</div>',
        unsafe_allow_html=True,
    )


# =========================================================================
# PAGE 1 — 🏠 Live NLU Demo
# =========================================================================
def page_live_demo() -> None:
    st.markdown(
        '<div class="hero-title">Live NLU Demo</div>'
        '<div class="hero-subtitle">'
        'Type any message and watch BotTrainer classify its intent, extract entities, '
        'and explain its reasoning in real-time.'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Input area ──
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        user_input: str = st.text_input(
            "Message",
            placeholder="e.g. Book a flight from Delhi to London next Friday",
            label_visibility="collapsed",
        )
    with col_btn:
        analyze_clicked = st.button("Analyze ⚡", type="primary", use_container_width=True)

    if analyze_clicked:
        if not user_input.strip():
            st.warning("Please type a message first.")
            return

        try:
            llm = _get_llm_client()
            classifier = _get_classifier(llm, st.session_state.n_few_shot)
            extractor = _get_extractor(llm)
        except Exception as exc:
            st.error(f"Initialization failed: {exc}")
            return

        with st.spinner(""):
            try:
                start_time = time.perf_counter()
                intent_result = classifier.classify(user_input)
                entities = extractor.extract(user_input, intent_result.get("intent", "unknown"))
                elapsed_ms = round((time.perf_counter() - start_time) * 1000)
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")
                return

        result: dict[str, Any] = {
            "user_message": user_input,
            "intent": intent_result.get("intent", "unknown"),
            "confidence": intent_result.get("confidence", 0.0),
            "entities": entities,
            "reasoning": intent_result.get("reasoning", ""),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model_choice,
            "latency_ms": elapsed_ms,
        }

        # ── Latency badge ──
        if elapsed_ms < 3000:
            lat_class, lat_icon = "latency-fast", "⚡"
        elif elapsed_ms < 6000:
            lat_class, lat_icon = "latency-mid", "⏱️"
        else:
            lat_class, lat_icon = "latency-slow", "🐢"

        st.markdown(
            f'<div style="text-align:right; margin: 8px 0;">'
            f'<span class="latency-badge {lat_class}">{lat_icon} {elapsed_ms:,}ms</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Results Grid ──
        col1, col2, col3 = st.columns([1.2, 1.4, 1.4])

        with col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown(
                f'<div style="margin-bottom:12px; font-size:0.8rem; color:#8b95a5; '
                f'text-transform:uppercase; letter-spacing:1.5px; font-weight:600;">Detected Intent</div>'
                f'<div class="intent-chip">{result["intent"]}</div>'
                f'<div class="confidence-ring">'
                f'<span class="confidence-pct">{result["confidence"]:.0%}</span>'
                f'<span class="confidence-label">confidence</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.progress(result["confidence"])
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown(
                '<div style="margin-bottom:12px; font-size:0.8rem; color:#8b95a5; '
                'text-transform:uppercase; letter-spacing:1.5px; font-weight:600;">'
                'Extracted Entities</div>',
                unsafe_allow_html=True,
            )
            if entities:
                entity_html = ""
                for k, v in entities.items():
                    entity_html += (
                        f'<div class="entity-row">'
                        f'<span class="entity-key">{k}</span>'
                        f'<span class="entity-val">{v}</span>'
                        f'</div>'
                    )
                st.markdown(entity_html, unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div style="color:#6b7280; font-style:italic; padding:12px 0;">'
                    'No entities detected in this message.</div>',
                    unsafe_allow_html=True,
                )
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown(
                '<div style="margin-bottom:12px; font-size:0.8rem; color:#8b95a5; '
                'text-transform:uppercase; letter-spacing:1.5px; font-weight:600;">'
                'AI Reasoning</div>',
                unsafe_allow_html=True,
            )
            reasoning_text = result["reasoning"] or "No reasoning provided."
            st.markdown(
                f'<div class="reasoning-box">'
                f'<span class="reasoning-icon">🧠</span>{reasoning_text}'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Raw JSON ──
        with st.expander("📋 Raw JSON Response"):
            st.json(result)

        # ── Store ──
        st.session_state.predictions.insert(0, result)
        st.session_state.predictions = st.session_state.predictions[:20]

    # ── History ──
    if st.session_state.predictions:
        st.markdown('<div class="pro-divider"></div>', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:1.1rem; font-weight:700; margin-bottom:16px; color:#c8d6e5;">'
            '📜 Prediction History</div>',
            unsafe_allow_html=True,
        )

        for i, pred in enumerate(st.session_state.predictions):
            latency = pred.get("latency_ms", 0)
            conf = pred.get("confidence", 0)
            if elapsed_ms < 3000:
                lat_cls = "latency-fast"
            elif elapsed_ms < 6000:
                lat_cls = "latency-mid"
            else:
                lat_cls = "latency-slow"

            with st.expander(
                f"#{i+1}  ·  {pred['intent']}  ·  {conf:.0%}  ·  {latency:,}ms  ·  \"{pred['user_message'][:55]}…\""
            ):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**Intent:** `{pred['intent']}`")
                    st.markdown(f"**Confidence:** {pred['confidence']:.1%}")
                    st.markdown(f"**Model:** {pred.get('model', 'N/A')}")
                with c2:
                    st.markdown(f"**Entities:** {pred.get('entities', {}) or 'None'}")
                    st.markdown(f"**Latency:** {latency:,}ms")
                    st.markdown(f"**Time:** {pred.get('timestamp', 'N/A')[:19]}")
                st.markdown(f"**Reasoning:** {pred.get('reasoning', 'N/A')}")


# =========================================================================
# PAGE 2 — 📊 Dataset Explorer
# =========================================================================
def page_dataset_explorer() -> None:
    st.markdown(
        '<div class="hero-title">Dataset Explorer</div>'
        '<div class="hero-subtitle">'
        'Explore the CLINC150 dataset — 150 intents across banking, travel, food, '
        'utilities, and more with 15,000 training utterances.'
        '</div>',
        unsafe_allow_html=True,
    )

    data = _load_intents()
    if data is None:
        st.error("intents.json not found. Run `python setup.py` first.")
        return

    intents: list[dict] = data.get("intents", [])
    metadata: dict = data.get("metadata", {})

    total_intents: int = metadata.get("total_intents", len(intents))
    total_examples: int = metadata.get("total_examples", 0)
    avg_examples: float = total_examples / max(total_intents, 1)

    # ── Stat Cards ──
    c1, c2, c3, c4 = st.columns(4)
    stats = [
        (c1, str(total_intents), "Total Intents"),
        (c2, f"{total_examples:,}", "Training Samples"),
        (c3, f"{avg_examples:.0f}", "Avg / Intent"),
        (c4, "10", "Domains"),
    ]
    for col, val, label in stats:
        with col:
            st.markdown(
                f'<div class="stat-card">'
                f'<div class="stat-value">{val}</div>'
                f'<div class="stat-label">{label}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="pro-divider"></div>', unsafe_allow_html=True)

    # ── Distribution Chart ──
    tab_chart, tab_explore, tab_meta = st.tabs(["📊 Distribution", "🔍 Explore Intents", "📋 Metadata"])

    with tab_chart:
        intent_counts = [
            {"intent": i["name"], "examples": len(i["examples"])} for i in intents
        ]
        df_counts = pd.DataFrame(intent_counts).sort_values("examples", ascending=False).head(30)

        fig = px.bar(
            df_counts,
            x="intent",
            y="examples",
            color="examples",
            color_continuous_scale=[[0, "#667eea"], [1, "#764ba2"]],
        )
        fig.update_layout(
            **PLOTLY_LAYOUT,
            height=500,
            xaxis_tickangle=-45,
            showlegend=False,
            coloraxis_showscale=False,
            xaxis_title="",
            yaxis_title="Examples",
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_explore:
        col_search, col_info = st.columns([1, 2])
        with col_search:
            intent_names = sorted([i["name"] for i in intents])
            selected = st.selectbox("Select an intent", intent_names, index=0)

        if selected:
            for i in intents:
                if i["name"] == selected:
                    with col_info:
                        st.markdown(
                            f'<div class="glass-card">'
                            f'<div class="intent-chip" style="font-size:0.95rem; padding:8px 18px;">{selected}</div>'
                            f'<div style="margin-top:10px; color:#8b95a5; font-size:0.85rem;">'
                            f'{len(i["examples"])} training examples</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                    st.dataframe(
                        pd.DataFrame(i["examples"], columns=["Example Utterance"]),
                        use_container_width=True,
                        hide_index=True,
                        height=400,
                    )
                    break

    with tab_meta:
        st.json(metadata)


# =========================================================================
# PAGE 3 — 🧪 Evaluation
# =========================================================================
def page_evaluation() -> None:
    st.markdown(
        '<div class="hero-title">Model Evaluation</div>'
        '<div class="hero-subtitle">'
        'Run batch evaluation against the CLINC150 test split to measure accuracy, '
        'precision, recall, and F1 across all 150 intents.'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Config area ──
    col_slider, col_btn = st.columns([3, 1])
    with col_slider:
        sample_size: int = st.slider(
            "Evaluation samples",
            min_value=20,
            max_value=750,
            value=100,
            step=10,
            help="Number of balanced samples to evaluate. More = more accurate but slower.",
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        run_clicked = st.button("🚀 Run Evaluation", type="primary", use_container_width=True)

    if run_clicked:
        eval_data = _load_eval_dataset()
        if eval_data is None:
            st.error("eval_dataset.json not found. Run `python setup.py` first.")
            return

        try:
            llm = _get_llm_client()
            classifier = _get_classifier(llm, st.session_state.n_few_shot)
        except Exception as exc:
            st.error(f"Initialization failed: {exc}")
            return

        evaluator = Evaluator()
        progress_bar = st.progress(0, text="Classifying samples …")

        try:
            results = evaluator.run_evaluation(
                eval_dataset_path=str(EVAL_DATASET_PATH),
                classifier=classifier,
                sample_size=sample_size,
            )
        except Exception as exc:
            st.error(f"Evaluation failed: {exc}")
            return

        progress_bar.progress(70, text="Computing metrics …")
        metrics = evaluator.compute_metrics(results["y_true"], results["y_pred"])
        progress_bar.progress(90, text="Analyzing errors …")
        error_df = evaluator.get_error_analysis(
            results["y_true"], results["y_pred"], results["texts"]
        )
        progress_bar.progress(100, text="✅ Complete!")

        st.session_state.eval_results = {
            "metrics": metrics,
            "y_true": results["y_true"],
            "y_pred": results["y_pred"],
            "texts": results["texts"],
            "error_df": error_df.to_dict(orient="records"),
            "sample_size": sample_size,
        }

    # ── Display results ──
    if st.session_state.eval_results is not None:
        er = st.session_state.eval_results
        metrics = er["metrics"]

        st.markdown('<div class="pro-divider"></div>', unsafe_allow_html=True)

        # ── Headline metrics ──
        m1, m2, m3, m4 = st.columns(4)
        metric_data = [
            (m1, f"{metrics['overall_accuracy']:.1%}", "Accuracy"),
            (m2, f"{metrics['macro_precision']:.1%}", "Precision"),
            (m3, f"{metrics['macro_recall']:.1%}", "Recall"),
            (m4, f"{metrics['macro_f1']:.1%}", "F1 Score"),
        ]
        for col, val, label in metric_data:
            with col:
                st.markdown(
                    f'<div class="stat-card">'
                    f'<div class="stat-value">{val}</div>'
                    f'<div class="stat-label">{label}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.markdown('<div class="pro-divider"></div>', unsafe_allow_html=True)

        # ── Tabs ──
        tab1, tab2, tab3 = st.tabs(["📈 Per-Intent Breakdown", "🗺️ Confusion Matrix", "❌ Error Analysis"])

        with tab1:
            per_intent = metrics.get("per_intent", {})
            if per_intent:
                pi_df = pd.DataFrame.from_dict(per_intent, orient="index")
                pi_df.index.name = "intent"
                pi_df = pi_df.sort_values("f1", ascending=False).reset_index()

                # Plotly heatmap-style bar chart for F1
                fig = px.bar(
                    pi_df.head(30),
                    x="intent",
                    y="f1",
                    color="f1",
                    color_continuous_scale=[[0, "#e74c3c"], [0.5, "#f39c12"], [1, "#2ecc71"]],
                )
                fig.update_layout(
                    **PLOTLY_LAYOUT,
                    height=400,
                    xaxis_tickangle=-45,
                    showlegend=False,
                    coloraxis_showscale=False,
                    xaxis_title="",
                    yaxis_title="F1 Score",
                )
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(
                    pi_df.style.background_gradient(cmap="RdYlGn", subset=["precision", "recall", "f1"]),
                    use_container_width=True,
                    hide_index=True,
                    height=400,
                )

        with tab2:
            evaluator = Evaluator()
            try:
                fig = evaluator.generate_confusion_matrix(er["y_true"], er["y_pred"])
                st.pyplot(fig)
                plt.close(fig)
            except Exception as exc:
                st.error(f"Could not generate confusion matrix: {exc}")

        with tab3:
            error_df = pd.DataFrame(er["error_df"])
            if error_df.empty:
                st.success("🎉 Perfect score — no misclassifications!")
            else:
                st.markdown(
                    f'<div style="font-size:1rem; color:#e74c3c; font-weight:600; margin-bottom:12px;">'
                    f'⚠️ {len(error_df)} misclassified out of {er.get("sample_size", "?")} samples</div>',
                    unsafe_allow_html=True,
                )

                filter_intents = sorted(error_df["true_intent"].unique().tolist())
                selected_filter = st.selectbox(
                    "Filter by true intent", ["(all)"] + filter_intents
                )
                if selected_filter != "(all)":
                    error_df = error_df[error_df["true_intent"] == selected_filter]

                st.dataframe(error_df, use_container_width=True, hide_index=True, height=400)

        # ── Download ──
        st.markdown('<div class="pro-divider"></div>', unsafe_allow_html=True)
        download_data = {
            "metrics": metrics,
            "total_evaluated": len(er["y_true"]),
            "errors": er["error_df"],
        }
        st.download_button(
            "💾 Download Full Results (JSON)",
            data=json.dumps(download_data, indent=2),
            file_name="bottrainer_eval_results.json",
            mime="application/json",
            use_container_width=True,
        )


# =========================================================================
# PAGE 4 — ⚙️ Settings
# =========================================================================
def page_settings() -> None:
    st.markdown(
        '<div class="hero-title">Settings</div>'
        '<div class="hero-subtitle">'
        'Configure API keys, model parameters, and manage system resources.'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── API Key Status ──
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY", "")

    st.markdown('<div class="settings-section">', unsafe_allow_html=True)
    st.markdown(
        '<div class="settings-title">🔑 API Configuration</div>',
        unsafe_allow_html=True,
    )
    if api_key:
        masked = api_key[:4] + "••••••" + api_key[-4:] if len(api_key) > 8 else "••••"
        st.markdown(
            f'<div style="display:flex; align-items:center; gap:8px;">'
            f'<span class="status-dot status-ok"></span>'
            f'<span style="color:#2ecc71; font-weight:600;">Connected</span>'
            f'<span style="color:#6b7280; margin-left:8px;">Key: {masked}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div style="display:flex; align-items:center; gap:8px;">'
            f'<span class="status-dot status-err"></span>'
            f'<span style="color:#e74c3c; font-weight:600;">Not configured</span>'
            f'<span style="color:#6b7280; margin-left:8px;">Add GEMINI_API_KEY to .env file</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Model Configuration ──
    st.markdown('<div class="settings-section">', unsafe_allow_html=True)
    st.markdown(
        '<div class="settings-title">🧠 Model Configuration</div>',
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            '<div style="padding:12px; border-radius:10px; background:rgba(102,126,234,0.08); '
            'border:1px solid rgba(102,126,234,0.15);">'
            '<div style="font-weight:700; color:#667eea;">Primary Model</div>'
            '<div style="color:#c8d6e5; font-size:0.95rem;">gemini-2.5-flash</div>'
            '<div style="color:#6b7280; font-size:0.8rem; margin-top:4px;">High accuracy · 1024 output tokens</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div style="padding:12px; border-radius:10px; background:rgba(118,75,162,0.08); '
            'border:1px solid rgba(118,75,162,0.15);">'
            '<div style="font-weight:700; color:#a78bfa;">Fast Model</div>'
            '<div style="color:#c8d6e5; font-size:0.95rem;">gemini-2.5-flash-lite</div>'
            '<div style="color:#6b7280; font-size:0.8rem; margin-top:4px;">Lower latency · Batch evaluation</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Few-Shot Tuning ──
    st.markdown('<div class="settings-section">', unsafe_allow_html=True)
    st.markdown(
        '<div class="settings-title">🎯 Classification Tuning</div>',
        unsafe_allow_html=True,
    )
    new_n = st.slider(
        "Few-shot examples per intent",
        min_value=1,
        max_value=5,
        value=st.session_state.n_few_shot,
        help="More examples = better accuracy but higher token usage and latency.",
    )
    if new_n != st.session_state.n_few_shot:
        st.session_state.n_few_shot = new_n
        st.cache_resource.clear()
        st.success(f"Updated to {new_n} examples/intent. Classifier will reload.")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Cache Management ──
    st.markdown('<div class="settings-section">', unsafe_allow_html=True)
    st.markdown(
        '<div class="settings-title">💾 Cache Management</div>',
        unsafe_allow_html=True,
    )
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        if st.button("🔄 Clear Model Cache", use_container_width=True):
            st.cache_resource.clear()
            st.success("Model cache cleared.")
    with col_c2:
        if st.button("🗑️ Clear Data Cache", use_container_width=True):
            st.cache_data.clear()
            st.success("Data cache cleared.")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── System Info ──
    st.markdown('<div class="settings-section">', unsafe_allow_html=True)
    st.markdown(
        '<div class="settings-title">💻 System Information</div>',
        unsafe_allow_html=True,
    )
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"**Python:** {platform.python_version()}")
        st.markdown(f"**OS:** {platform.system()} {platform.release()}")

    with col2:
        st.markdown(f"**Streamlit:** {st.__version__}")
        try:
            import sklearn
            st.markdown(f"**scikit-learn:** {sklearn.__version__}")
        except Exception:
            st.markdown("**scikit-learn:** N/A")

    with col3:
        try:
            import google.genai
            st.markdown(f"**google-genai:** ✅ installed")
        except ImportError:
            st.markdown("**google-genai:** ❌ missing")
        st.markdown(f"**TF-IDF features:** 15,000")

    st.markdown('</div>', unsafe_allow_html=True)


# =========================================================================
# Router
# =========================================================================
if page == "🏠 Live Demo":
    page_live_demo()
elif page == "📊 Dataset Explorer":
    page_dataset_explorer()
elif page == "🧪 Evaluation":
    page_evaluation()
elif page == "⚙️ Settings":
    page_settings()
