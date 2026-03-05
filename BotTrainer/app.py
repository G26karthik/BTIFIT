"""
BotTrainer — Streamlit Web Application.
==========================================
Production-grade NLU demo with polished UI, dataset explorer,
model evaluation dashboard, and system settings.
"""

import json
import os
import platform
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
# Project root (BotTrainer/) — used for locating data/prompt files
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent

from BotTrainer.modules.entity_extractor import EntityExtractor
from BotTrainer.modules.evaluator import Evaluator
from BotTrainer.modules.intent_classifier import IntentClassifier
from BotTrainer.modules.llm_client import GeminiClient
from BotTrainer.modules.utils import load_json_file, setup_logging, esc
from BotTrainer.styles import CUSTOM_CSS

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
# Apply CSS theme
# ---------------------------------------------------------------------------
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
INTENTS_PATH: Path = PROJECT_ROOT / "data" / "intents.json"
EVAL_DATASET_PATH: Path = PROJECT_ROOT / "data" / "eval_dataset.json"


# ---------------------------------------------------------------------------
# Cached initialisation helpers
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Initializing Gemini client ...")
def _get_llm_client(model: str = "gemini-2.5-flash") -> GeminiClient:
    return GeminiClient(model=model)


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
        '<div class="sidebar-logo">🤖</div>'
        '<div class="sidebar-brand-title">BotTrainer</div>'
        '<div class="sidebar-brand-sub">NLU Engine</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-label">Navigation</div>', unsafe_allow_html=True)

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
    st.markdown('<div class="sidebar-section-label">Model</div>', unsafe_allow_html=True)

    model_choice = st.selectbox(
        "Active Model",
        ["gemini-2.5-flash", "gemini-2.5-flash-lite"],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # Quick stats
    load_dotenv()
    _api_ok = bool(os.getenv("GEMINI_API_KEY", ""))
    st.markdown(
        f'<div style="padding:0 16px;">'
        f'<div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">'
        f'<span class="status-dot {"status-ok" if _api_ok else "status-err"}"></span>'
        f'<span style="font-size:0.8rem; color:{"var(--success)" if _api_ok else "var(--error)"}; font-weight:500;">'
        f'{"API Connected" if _api_ok else "API Key Missing"}</span>'
        f'</div>'
        f'<div style="display:flex; align-items:center; gap:8px;">'
        f'<span class="status-dot status-ok"></span>'
        f'<span style="font-size:0.8rem; color:var(--success); font-weight:500;">System Online</span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-footer">'
        'BotTrainer v1.0 · Powered by Gemini'
        '</div>',
        unsafe_allow_html=True,
    )


# =========================================================================
# PAGE 1 — 🏠 Live NLU Demo
# =========================================================================
def page_live_demo() -> None:
    st.markdown(
        '<div class="hero-section">'
        '<div class="hero-title">Live NLU Demo</div>'
        '<div class="hero-subtitle">'
        'Type any message and watch BotTrainer classify its intent, extract entities, '
        'and explain its reasoning — powered by Gemini 2.5 Flash.'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Input area — styled as a modern search bar ──
    col_input, col_btn = st.columns([5, 1], gap="small")
    with col_input:
        user_input: str = st.text_input(
            "Message",
            placeholder="e.g. Book a flight from Delhi to London next Friday …",
            label_visibility="collapsed",
        )
    with col_btn:
        analyze_clicked = st.button("Analyze ⚡", type="primary", use_container_width=True)

    if analyze_clicked:
        if not user_input.strip():
            st.warning("Please type a message first.")
            return

        try:
            llm = _get_llm_client(model_choice)
            classifier = _get_classifier(llm, st.session_state.n_few_shot)
            extractor = _get_extractor(llm)
        except Exception as exc:
            st.error(f"Initialization failed: {exc}")
            return

        # ── Animated loading ──
        with st.status("Analyzing message …", expanded=True) as status:
            st.write("🔍 Classifying intent …")
            start_time = time.perf_counter()
            try:
                intent_result = classifier.classify(user_input)
            except Exception as exc:
                st.error(f"Classification failed: {exc}")
                return
            st.write("🏷️ Extracting entities …")
            try:
                entities = extractor.extract(user_input, intent_result.get("intent", "unknown"))
            except Exception as exc:
                st.error(f"Entity extraction failed: {exc}")
                return
            elapsed_ms = round((time.perf_counter() - start_time) * 1000)
            status.update(label="Analysis complete ✅", state="complete", expanded=False)

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
            f'<div style="text-align:right; margin: 12px 0 4px 0;" class="animate-in">'
            f'<span class="latency-badge {lat_class}">{lat_icon} {elapsed_ms:,}ms</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Results Grid ──
        col1, col2, col3 = st.columns([1.2, 1.4, 1.4], gap="medium")

        with col1:
            conf_pct = result["confidence"]
            # Circular gauge via Plotly
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=conf_pct * 100,
                number={"suffix": "%", "font": {"size": 36, "color": "#667eea", "family": "Inter"}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 0, "tickcolor": "rgba(0,0,0,0)",
                             "tickfont": {"color": "rgba(0,0,0,0)"}},
                    "bar": {"color": "#667eea", "thickness": 0.35},
                    "bgcolor": "rgba(255,255,255,0.03)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 40], "color": "rgba(248,113,113,0.08)"},
                        {"range": [40, 70], "color": "rgba(251,191,36,0.08)"},
                        {"range": [70, 100], "color": "rgba(52,211,153,0.08)"},
                    ],
                    "threshold": {
                        "line": {"color": "#764ba2", "width": 3},
                        "thickness": 0.8,
                        "value": conf_pct * 100,
                    },
                },
            ))
            gauge.update_layout(
                height=200,
                margin=dict(t=30, b=10, l=30, r=30),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, sans-serif"),
            )

            st.markdown(
                '<div class="glass-card">'
                '<div class="card-label">Detected Intent</div>'
                f'<div class="intent-chip">{esc(result["intent"])}</div>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(gauge, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown(
                '<div class="glass-card">'
                '<div class="card-label">Extracted Entities</div>',
                unsafe_allow_html=True,
            )
            if entities:
                entity_html = ""
                for k, v in entities.items():
                    entity_html += (
                        f'<div class="entity-row">'
                        f'<span class="entity-key">{esc(k)}</span>'
                        f'<span class="entity-val">{esc(v)}</span>'
                        f'</div>'
                    )
                st.markdown(entity_html, unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div style="color:var(--text-muted); font-style:italic; padding:20px 0; '
                    'text-align:center; font-size:0.9rem;">'
                    '— No entities detected —</div>',
                    unsafe_allow_html=True,
                )
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            reasoning_text = result["reasoning"] or "No reasoning provided."
            st.markdown(
                '<div class="glass-card">'
                '<div class="card-label">AI Reasoning</div>'
                f'<div class="reasoning-box">{esc(reasoning_text)}</div>'
                '</div>',
                unsafe_allow_html=True,
            )

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
            '<div style="display:flex; align-items:center; justify-content:space-between; '
            'margin-bottom:18px;">'
            '<div style="font-size:1.05rem; font-weight:700; color:var(--text-primary);">'
            '📜 Prediction History</div>'
            f'<div style="font-size:0.78rem; color:var(--text-muted);">'
            f'{len(st.session_state.predictions)} predictions</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        for i, pred in enumerate(st.session_state.predictions):
            latency = pred.get("latency_ms", 0)
            conf = pred.get("confidence", 0)
            intent = pred.get("intent", "unknown")
            msg_preview = pred["user_message"][:60]

            with st.expander(
                f"#{i+1}  ·  {intent}  ·  {conf:.0%}  ·  {latency:,}ms  ·  \"{msg_preview}…\""
            ):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**Intent:** `{intent}`")
                    st.markdown(f"**Confidence:** {pred['confidence']:.1%}")
                    st.markdown(f"**Model:** `{pred.get('model', 'N/A')}`")
                with c2:
                    ent = pred.get("entities", {})
                    st.markdown(f"**Entities:** {ent if ent else 'None'}")
                    st.markdown(f"**Latency:** {latency:,}ms")
                    st.markdown(f"**Time:** {pred.get('timestamp', 'N/A')[:19]}")
                if pred.get("reasoning"):
                    st.markdown(f"**Reasoning:** {pred['reasoning']}")


# =========================================================================
# PAGE 2 — 📊 Dataset Explorer
# =========================================================================
def page_dataset_explorer() -> None:
    st.markdown(
        '<div class="hero-section">'
        '<div class="hero-title">Dataset Explorer</div>'
        '<div class="hero-subtitle">'
        'Explore the CLINC150 dataset — 150 intents across banking, travel, food, '
        'utilities, and more with 15,000 training utterances.'
        '</div>'
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
    c1, c2, c3, c4 = st.columns(4, gap="medium")
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

    # ── Tabs ──
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
            color_continuous_scale=[[0, "#667eea"], [0.5, "#8b5cf6"], [1, "#764ba2"]],
        )
        fig.update_layout(
            **PLOTLY_LAYOUT,
            height=480,
            xaxis_tickangle=-45,
            showlegend=False,
            coloraxis_showscale=False,
            xaxis_title="",
            yaxis_title="Examples",
        )
        fig.update_traces(marker_line_width=0, opacity=0.9)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with tab_explore:
        col_search, col_info = st.columns([1, 2.5], gap="large")
        with col_search:
            intent_names = sorted([i["name"] for i in intents])
            selected = st.selectbox("Select an intent", intent_names, index=0)

        if selected:
            for i in intents:
                if i["name"] == selected:
                    with col_info:
                        st.markdown(
                            f'<div class="glass-card" style="padding:18px 22px;">'
                            f'<div style="display:flex; align-items:center; justify-content:space-between;">'
                            f'<div class="intent-chip" style="font-size:0.9rem; padding:8px 18px;">{selected}</div>'
                            f'<div style="color:var(--text-muted); font-size:0.82rem; font-weight:600;">'
                            f'{len(i["examples"])} examples</div>'
                            f'</div>'
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
        st.markdown(
            '<div class="glass-card"><div class="card-label">Dataset Metadata</div>',
            unsafe_allow_html=True,
        )
        st.json(metadata)
        st.markdown('</div>', unsafe_allow_html=True)


# =========================================================================
# PAGE 3 — 🧪 Evaluation
# =========================================================================
def page_evaluation() -> None:
    st.markdown(
        '<div class="hero-section">'
        '<div class="hero-title">Model Evaluation</div>'
        '<div class="hero-subtitle">'
        'Run batch evaluation against the CLINC150 test split to measure accuracy, '
        'precision, recall, and F1 across all 150 intents.'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Config area ──
    st.markdown(
        '<div class="glass-card" style="padding:20px 24px;">',
        unsafe_allow_html=True,
    )
    col_slider, col_info, col_btn = st.columns([3, 1, 1], gap="medium")
    with col_slider:
        sample_size: int = st.slider(
            "Evaluation samples",
            min_value=20,
            max_value=750,
            value=100,
            step=10,
            help="Number of balanced samples to evaluate. More = more accurate but slower.",
        )
    with col_info:
        est_cost = sample_size * 0.0004
        est_time = sample_size * 2.5
        st.markdown(
            f'<div style="padding-top:12px;">'
            f'<div style="font-size:0.78rem; color:var(--text-muted); text-transform:uppercase; '
            f'letter-spacing:1px; font-weight:600;">Estimates</div>'
            f'<div style="color:var(--text-secondary); font-size:0.88rem; margin-top:6px;">'
            f'⏱️ ~{est_time/60:.0f} min  ·  💰 ~${est_cost:.2f}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col_btn:
        st.markdown("<div style='padding-top:16px;'>", unsafe_allow_html=True)
        run_clicked = st.button("🚀 Run Evaluation", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if run_clicked:
        eval_data = _load_eval_dataset()
        if eval_data is None:
            st.error("eval_dataset.json not found. Run `python setup.py` first.")
            return

        try:
            llm = _get_llm_client(model_choice)
            classifier = _get_classifier(llm, st.session_state.n_few_shot)
        except Exception as exc:
            st.error(f"Initialization failed: {exc}")
            return

        evaluator = Evaluator()

        with st.status("Running evaluation …", expanded=True) as status:
            progress_bar = st.progress(0, text="Classifying …")

            def _on_progress(current: int, total: int) -> None:
                progress_bar.progress(current / total, text=f"Classifying {current}/{total} …")

            try:
                results = evaluator.run_evaluation(
                    eval_dataset_path=str(EVAL_DATASET_PATH),
                    classifier=classifier,
                    sample_size=sample_size,
                    progress_callback=_on_progress,
                )
            except Exception as exc:
                st.error(f"Evaluation failed: {exc}")
                return

            st.write("📊 Computing metrics …")
            metrics = evaluator.compute_metrics(results["y_true"], results["y_pred"])

            st.write("🔍 Analyzing errors …")
            error_df = evaluator.get_error_analysis(
                results["y_true"], results["y_pred"], results["texts"]
            )
            status.update(label="Evaluation complete ✅", state="complete", expanded=False)

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
        m1, m2, m3, m4 = st.columns(4, gap="medium")
        metric_items = [
            (m1, f"{metrics['overall_accuracy']:.1%}", "Accuracy", "🎯"),
            (m2, f"{metrics['macro_precision']:.1%}", "Precision", "📏"),
            (m3, f"{metrics['macro_recall']:.1%}", "Recall", "🔄"),
            (m4, f"{metrics['macro_f1']:.1%}", "F1 Score", "⚖️"),
        ]
        for col, val, label, icon in metric_items:
            with col:
                st.markdown(
                    f'<div class="stat-card">'
                    f'<div style="font-size:1.2rem; margin-bottom:4px;">{icon}</div>'
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

                fig = px.bar(
                    pi_df.head(30),
                    x="intent",
                    y="f1",
                    color="f1",
                    color_continuous_scale=[[0, "#f87171"], [0.5, "#fbbf24"], [1, "#34d399"]],
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
                fig.update_traces(marker_line_width=0, opacity=0.9)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

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
                st.markdown(
                    '<div class="glass-card" style="text-align:center; padding:40px;">'
                    '<div style="font-size:2rem; margin-bottom:8px;">🎉</div>'
                    '<div style="font-size:1.1rem; font-weight:600; color:var(--success);">'
                    'Perfect Score — No Misclassifications!</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )
            else:
                total = er.get("sample_size", len(er["y_true"]))
                err_pct = len(error_df) / max(total, 1) * 100
                st.markdown(
                    f'<div class="glass-card" style="padding:16px 22px; border-left:3px solid var(--error);">'
                    f'<div style="display:flex; align-items:center; gap:12px;">'
                    f'<div style="font-size:1.5rem;">⚠️</div>'
                    f'<div>'
                    f'<div style="font-size:1rem; font-weight:700; color:var(--error);">'
                    f'{len(error_df)} Misclassifications</div>'
                    f'<div style="font-size:0.82rem; color:var(--text-muted);">'
                    f'{err_pct:.1f}% error rate out of {total} samples</div>'
                    f'</div></div></div>',
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
        '<div class="hero-section">'
        '<div class="hero-title">Settings</div>'
        '<div class="hero-subtitle">'
        'Configure API keys, model parameters, and manage system resources.'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── API Key Status ──
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY", "")

    st.markdown('<div class="settings-section">', unsafe_allow_html=True)
    st.markdown('<div class="settings-title">🔑 API Configuration</div>', unsafe_allow_html=True)
    if api_key:
        masked = api_key[:4] + "••••••" + api_key[-4:] if len(api_key) > 8 else "••••"
        st.markdown(
            f'<div style="display:flex; align-items:center; gap:10px;">'
            f'<span class="status-dot status-ok"></span>'
            f'<span style="color:var(--success); font-weight:600; font-size:0.9rem;">Connected</span>'
            f'<span style="color:var(--text-muted); margin-left:4px; font-family:JetBrains Mono,monospace; '
            f'font-size:0.82rem;">{masked}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="display:flex; align-items:center; gap:10px;">'
            '<span class="status-dot status-err"></span>'
            '<span style="color:var(--error); font-weight:600; font-size:0.9rem;">Not configured</span>'
            '<span style="color:var(--text-muted); font-size:0.82rem;">Add GEMINI_API_KEY to .env</span>'
            '</div>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Model Configuration ──
    st.markdown('<div class="settings-section">', unsafe_allow_html=True)
    st.markdown('<div class="settings-title">🧠 Model Configuration</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.markdown(
            '<div class="model-card model-card-primary">'
            '<div class="model-card-name primary">Primary Model</div>'
            '<div class="model-card-model">gemini-2.5-flash</div>'
            '<div class="model-card-desc">High accuracy · 1024 output tokens · Intent classification</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div class="model-card model-card-fast">'
            '<div class="model-card-name fast">Fast Model</div>'
            '<div class="model-card-model">gemini-2.5-flash-lite</div>'
            '<div class="model-card-desc">Lower latency · Batch evaluation · Cost-optimized</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Few-Shot Tuning ──
    st.markdown('<div class="settings-section">', unsafe_allow_html=True)
    st.markdown('<div class="settings-title">🎯 Classification Tuning</div>', unsafe_allow_html=True)
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
    st.markdown('<div class="settings-title">💾 Cache Management</div>', unsafe_allow_html=True)
    col_c1, col_c2 = st.columns(2, gap="medium")
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
    st.markdown('<div class="settings-title">💻 System Information</div>', unsafe_allow_html=True)

    py_ver = platform.python_version()
    os_info = f"{platform.system()} {platform.release()}"
    st_ver = st.__version__
    try:
        import sklearn
        sk_ver = sklearn.__version__
    except Exception:
        sk_ver = "N/A"
    try:
        import google.genai
        genai_ok = True
    except ImportError:
        genai_ok = False

    info_items = [
        ("Python", py_ver),
        ("OS", os_info),
        ("Streamlit", st_ver),
        ("scikit-learn", sk_ver),
        ("google-genai", "✅ installed" if genai_ok else "❌ missing"),
        ("TF-IDF features", "15,000"),
        ("Cache size", "500 entries"),
        ("Few-shot", f"{st.session_state.n_few_shot} examples"),
    ]

    # Render as a clean 2-col info grid via HTML
    rows_html = ""
    for label, val in info_items:
        rows_html += (
            f'<div style="display:flex; justify-content:space-between; padding:8px 0; '
            f'border-bottom:1px solid var(--border-subtle);">'
            f'<span style="color:var(--text-muted); font-size:0.85rem; font-weight:500;">{label}</span>'
            f'<span style="color:var(--text-primary); font-size:0.85rem; font-weight:600; '
            f'font-family:JetBrains Mono,monospace;">{val}</span>'
            f'</div>'
        )
    st.markdown(rows_html, unsafe_allow_html=True)
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
