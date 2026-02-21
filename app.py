"""
BotTrainer — Streamlit Web Application.
==========================================
Multi-page NLU demo, dataset explorer, model evaluation, and settings.
"""

import json
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
import streamlit as st

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
    page_title="BotTrainer 🤖",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Sidebar accent */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    [data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }

    /* Metric card styling */
    [data-testid="stMetric"] {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 16px;
        border-left: 4px solid #4361ee;
    }

    /* Intent badge */
    .intent-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        color: #fff;
        background: linear-gradient(135deg, #4361ee, #3a0ca3);
    }

    /* Confidence bar colour */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #4361ee, #4cc9f0);
    }

    /* Section divider */
    .section-divider {
        border-top: 2px solid #e0e0e0;
        margin: 24px 0;
    }
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
@st.cache_resource(show_spinner="Loading Gemini client …")
def _get_llm_client() -> GeminiClient:
    """Create and cache the GeminiClient singleton."""
    return GeminiClient()


@st.cache_resource(show_spinner="Building intent classifier …")
def _get_classifier(_llm: GeminiClient, n_few_shot: int = 3) -> IntentClassifier:
    """Create and cache the IntentClassifier."""
    return IntentClassifier(
        intents_json_path=str(INTENTS_PATH),
        llm_client=_llm,
        n_few_shot=n_few_shot,
    )


@st.cache_resource(show_spinner="Loading entity extractor …")
def _get_extractor(_llm: GeminiClient) -> EntityExtractor:
    """Create and cache the EntityExtractor."""
    return EntityExtractor(llm_client=_llm)


@st.cache_data(show_spinner="Loading intents …")
def _load_intents() -> Optional[dict]:
    """Load and cache intents.json."""
    return load_json_file(INTENTS_PATH)


@st.cache_data(show_spinner="Loading eval dataset …")
def _load_eval_dataset() -> Optional[dict]:
    """Load and cache eval_dataset.json."""
    return load_json_file(EVAL_DATASET_PATH)


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
if "predictions" not in st.session_state:
    st.session_state.predictions = []
if "n_few_shot" not in st.session_state:
    st.session_state.n_few_shot = 5
if "eval_results" not in st.session_state:
    st.session_state.eval_results = None


# =========================================================================
# SIDEBAR — Navigation
# =========================================================================
with st.sidebar:
    st.title("🤖 BotTrainer")
    st.caption("LLM-Based NLU Trainer & Evaluator")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        options=[
            "🏠 Live Demo",
            "📊 Dataset Explorer",
            "🧪 Evaluation",
            "⚙️ Settings",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.caption("Powered by Gemini 2.5 Flash")


# =========================================================================
# PAGE 1 — 🏠 Live NLU Demo
# =========================================================================
def page_live_demo() -> None:
    """Render the Live NLU Demo page."""
    st.header("🏠 Live NLU Demo")
    st.markdown("Type a message below and let BotTrainer classify its intent and extract entities.")

    # Model selector in sidebar
    with st.sidebar:
        model_choice = st.selectbox(
            "Model",
            ["gemini-2.5-flash", "gemini-2.5-flash-lite"],
            index=0,
        )

    user_input: str = st.text_input(
        "Type a message to analyze …",
        placeholder="e.g. Transfer $50 to John's savings account",
    )

    if st.button("🔍 Analyze", type="primary", use_container_width=True):
        if not user_input.strip():
            st.warning("Please type a message first.")
            return

        try:
            llm = _get_llm_client()
            classifier = _get_classifier(llm, st.session_state.n_few_shot)
            extractor = _get_extractor(llm)
        except Exception as exc:
            st.error(f"Failed to initialise components: {exc}")
            return

        use_fast = model_choice == "gemini-2.5-flash-lite"
        with st.spinner(f"🤖 Analysing with {model_choice} …"):
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

        # Response time badge
        latency_color = "#2ecc71" if elapsed_ms < 3000 else "#f39c12" if elapsed_ms < 6000 else "#e74c3c"
        st.markdown(
            f'<div style="text-align:right; margin-bottom:8px;">'
            f'⏱️ <span style="color:{latency_color}; font-weight:700;">{elapsed_ms:,}ms</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Display results in three columns
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Intent**")
            st.markdown(
                f'<span class="intent-badge">{result["intent"]}</span>',
                unsafe_allow_html=True,
            )
            st.markdown(f"**Confidence:** {result['confidence']:.1%}")
            st.progress(result["confidence"])

        with col2:
            st.markdown("**Entities**")
            if entities:
                ent_df = pd.DataFrame(
                    list(entities.items()), columns=["Entity Type", "Value"]
                )
                st.dataframe(ent_df, use_container_width=True, hide_index=True)
            else:
                st.info("No entities found.")

        with col3:
            st.markdown("**Reasoning**")
            st.info(result["reasoning"] or "No reasoning provided.")

        with st.expander("🔎 View raw JSON output"):
            st.json(result)

        # Store in session
        st.session_state.predictions.insert(0, result)
        st.session_state.predictions = st.session_state.predictions[:10]

    # Recent predictions
    if st.session_state.predictions:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.subheader("Recent Predictions")
        for i, pred in enumerate(st.session_state.predictions):
            latency = pred.get('latency_ms', 0)
            latency_str = f" \u2014 {latency:,}ms" if latency else ""
            with st.expander(
                f"#{i+1} \u2014 {pred['intent']} ({pred['confidence']:.0%}){latency_str} \u2014 \"{pred['user_message'][:60]}\u2026\""
            ):
                st.json(pred)


# =========================================================================
# PAGE 2 — 📊 Dataset Explorer
# =========================================================================
def page_dataset_explorer() -> None:
    """Render the Dataset Explorer page."""
    st.header("📊 Dataset Explorer")

    data = _load_intents()
    if data is None:
        st.error(
            "intents.json not found. Run `python setup.py` first to download and preprocess the dataset."
        )
        return

    intents: list[dict] = data.get("intents", [])
    metadata: dict = data.get("metadata", {})

    total_intents: int = metadata.get("total_intents", len(intents))
    total_examples: int = metadata.get("total_examples", 0)
    avg_examples: float = total_examples / max(total_intents, 1)

    # Top stats
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Intents", total_intents)
    c2.metric("Total Examples", f"{total_examples:,}")
    c3.metric("Avg Examples / Intent", f"{avg_examples:.1f}")

    # Bar chart — top 30 intents by example count
    st.subheader("Top 30 Intents by Example Count")
    intent_counts: list[dict[str, Any]] = [
        {"intent": i["name"], "examples": len(i["examples"])} for i in intents
    ]
    df_counts = pd.DataFrame(intent_counts).sort_values("examples", ascending=False).head(30)

    fig = px.bar(
        df_counts,
        x="intent",
        y="examples",
        color="examples",
        color_continuous_scale="Blues",
        title="Example Distribution Across Top 30 Intents",
    )
    fig.update_layout(xaxis_tickangle=-45, height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Intent search
    st.subheader("Explore an Intent")
    intent_names: list[str] = [i["name"] for i in intents]
    selected = st.selectbox("Select an intent", sorted(intent_names))

    if selected:
        for i in intents:
            if i["name"] == selected:
                st.markdown(f"**Intent:** `{selected}` — **{len(i['examples'])} examples**")
                st.dataframe(
                    pd.DataFrame(i["examples"], columns=["Example Utterance"]),
                    use_container_width=True,
                    hide_index=True,
                )
                break

    # Metadata
    with st.expander("📋 Dataset Metadata"):
        st.json(metadata)


# =========================================================================
# PAGE 3 — 🧪 Model Evaluation
# =========================================================================
def page_evaluation() -> None:
    """Render the Model Evaluation page."""
    st.header("🧪 Model Evaluation")
    st.warning("⚡ This will make API calls to Gemini. Estimated cost: ~$0.01–0.05 depending on sample size.")

    sample_size: int = st.slider(
        "Sample size",
        min_value=20,
        max_value=755,
        value=100,
        step=10,
        help="Number of evaluation samples to classify.",
    )

    if st.button("🚀 Run Evaluation", type="primary", use_container_width=True):
        eval_data = _load_eval_dataset()
        if eval_data is None:
            st.error("eval_dataset.json not found. Run `python setup.py` first.")
            return

        try:
            llm = _get_llm_client()
            classifier = _get_classifier(llm, st.session_state.n_few_shot)
        except Exception as exc:
            st.error(f"Failed to initialise: {exc}")
            return

        evaluator = Evaluator()

        progress_bar = st.progress(0)
        status_text = st.empty()

        # Run evaluation
        status_text.text("🤖 Running batch classification …")
        try:
            results = evaluator.run_evaluation(
                eval_dataset_path=str(EVAL_DATASET_PATH),
                classifier=classifier,
                sample_size=sample_size,
            )
        except Exception as exc:
            st.error(f"Evaluation failed: {exc}")
            return

        progress_bar.progress(70)
        status_text.text("📊 Computing metrics …")

        metrics = evaluator.compute_metrics(results["y_true"], results["y_pred"])
        progress_bar.progress(90)

        error_df = evaluator.get_error_analysis(
            results["y_true"], results["y_pred"], results["texts"]
        )

        progress_bar.progress(100)
        status_text.text("✅ Evaluation complete!")

        # Store results in session
        st.session_state.eval_results = {
            "metrics": metrics,
            "y_true": results["y_true"],
            "y_pred": results["y_pred"],
            "texts": results["texts"],
            "error_df": error_df.to_dict(orient="records"),
        }

    # Display results if available
    if st.session_state.eval_results is not None:
        er = st.session_state.eval_results
        metrics = er["metrics"]

        tab1, tab2, tab3 = st.tabs(["📈 Metrics", "🗺️ Confusion Matrix", "❌ Error Analysis"])

        # ── Tab 1: Metrics ──
        with tab1:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{metrics['overall_accuracy']:.2%}")
            m2.metric("Precision (macro)", f"{metrics['macro_precision']:.2%}")
            m3.metric("Recall (macro)", f"{metrics['macro_recall']:.2%}")
            m4.metric("F1 (macro)", f"{metrics['macro_f1']:.2%}")

            st.subheader("Per-Intent Breakdown")
            per_intent = metrics.get("per_intent", {})
            if per_intent:
                pi_df = pd.DataFrame.from_dict(per_intent, orient="index")
                pi_df.index.name = "intent"
                pi_df = pi_df.sort_values("f1", ascending=False).reset_index()
                st.dataframe(
                    pi_df.style.background_gradient(cmap="Blues", subset=["precision", "recall", "f1"]),
                    use_container_width=True,
                    hide_index=True,
                )

        # ── Tab 2: Confusion Matrix ──
        with tab2:
            st.markdown("*Shows the top-N most frequent intents (max 30) for readability.*")
            evaluator = Evaluator()
            try:
                fig = evaluator.generate_confusion_matrix(er["y_true"], er["y_pred"])
                st.pyplot(fig)
                plt.close(fig)
            except Exception as exc:
                st.error(f"Could not generate confusion matrix: {exc}")

        # ── Tab 3: Error Analysis ──
        with tab3:
            error_df = pd.DataFrame(er["error_df"])
            if error_df.empty:
                st.success("🎉 No misclassifications!")
            else:
                st.markdown(f"**{len(error_df)} misclassified samples**")

                filter_intents = sorted(error_df["true_intent"].unique().tolist())
                selected_filter = st.selectbox(
                    "Filter by true intent", ["(all)"] + filter_intents
                )
                if selected_filter != "(all)":
                    error_df = error_df[error_df["true_intent"] == selected_filter]

                st.dataframe(error_df, use_container_width=True, hide_index=True)

        # Download button
        st.markdown("---")
        download_data = {
            "metrics": metrics,
            "total_evaluated": len(er["y_true"]),
            "errors": er["error_df"],
        }
        st.download_button(
            "💾 Download Results (JSON)",
            data=json.dumps(download_data, indent=2),
            file_name="bottrainer_eval_results.json",
            mime="application/json",
        )


# =========================================================================
# PAGE 4 — ⚙️ Settings
# =========================================================================
def page_settings() -> None:
    """Render the Settings page."""
    st.header("⚙️ Settings")

    # API key status
    import os
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY", "")
    if api_key:
        masked = api_key[:4] + "…" + api_key[-4:] if len(api_key) > 8 else "****"
        st.success(f"✅ GEMINI_API_KEY loaded: `{masked}`")
    else:
        st.error("❌ GEMINI_API_KEY not found — add it to your `.env` file.")

    st.markdown("---")

    # Model info
    st.subheader("Model Configuration")
    st.info("**Primary model:** gemini-2.5-flash\n\n**Fast model:** gemini-2.5-flash-lite")

    # Few-shot slider
    st.subheader("Few-Shot Setting")
    new_n = st.slider(
        "Examples per intent in prompt",
        min_value=1,
        max_value=5,
        value=st.session_state.n_few_shot,
        help="Higher values give more context but increase token usage.",
    )
    if new_n != st.session_state.n_few_shot:
        st.session_state.n_few_shot = new_n
        st.cache_resource.clear()
        st.success(f"Updated n_few_shot to {new_n}. Classifier will reload on next use.")

    st.markdown("---")

    # Reload dataset
    st.subheader("Data Management")
    if st.button("🔄 Reload Dataset Cache"):
        st.cache_data.clear()
        st.success("Dataset cache cleared. Data will reload on next page visit.")

    st.markdown("---")

    # System info
    st.subheader("System Info")
    import platform
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Python:** {platform.python_version()}")
        st.markdown(f"**Platform:** {platform.system()} {platform.release()}")
    with col2:
        try:
            import google.genai as _g
            st.markdown(f"**google-genai:** installed")
        except ImportError:
            st.markdown("**google-genai:** not installed")
        try:
            import streamlit as _st
            st.markdown(f"**Streamlit:** {_st.__version__}")
        except Exception:
            pass
        try:
            import sklearn
            st.markdown(f"**scikit-learn:** {sklearn.__version__}")
        except Exception:
            pass


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
