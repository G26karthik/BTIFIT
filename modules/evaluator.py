"""
Evaluator for BotTrainer.
=============================
Computes classification metrics, confusion matrix, and error analysis.
"""

import logging
import random
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from modules.intent_classifier import IntentClassifier
from modules.utils import load_json_file, save_json_file

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluate intent classification quality with standard NLP metrics.

    Provides accuracy, precision, recall, F1, confusion matrix visualisation,
    and per-sample error analysis.
    """

    # ------------------------------------------------------------------ #
    #  Run full evaluation                                               #
    # ------------------------------------------------------------------ #

    def run_evaluation(
        self,
        eval_dataset_path: str | Path,
        classifier: IntentClassifier,
        sample_size: Optional[int] = None,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Run batch classification on the evaluation dataset.

        Args:
            eval_dataset_path: Path to ``eval_dataset.json``.
            classifier: An initialised :class:`IntentClassifier`.
            sample_size: If given, take a random balanced subset of this size.
            seed: Random seed for subsetting.

        Returns:
            Dict with keys ``y_true``, ``y_pred``, ``texts``, ``raw_results``,
            and ``metadata``.

        Raises:
            FileNotFoundError: If the eval dataset file is missing.
        """
        eval_path = Path(eval_dataset_path)
        data = load_json_file(eval_path)
        if data is None:
            raise FileNotFoundError(f"Eval dataset not found: {eval_path}")

        samples: list[dict[str, Any]] = data.get("eval_samples", [])
        if not samples:
            raise ValueError("eval_dataset.json has no eval_samples.")

        # Optional balanced sub-sampling
        if sample_size is not None and sample_size < len(samples):
            samples = self._balanced_sample(samples, sample_size, seed)

        texts: list[str] = [s["text"] for s in samples]
        y_true: list[str] = [s["true_intent"] for s in samples]

        logger.info("Running evaluation on %d samples …", len(texts))
        raw_results: list[dict[str, Any]] = classifier.batch_classify(texts, show_progress=True)
        y_pred: list[str] = [r.get("intent", "parse_error") for r in raw_results]

        return {
            "y_true": y_true,
            "y_pred": y_pred,
            "texts": texts,
            "raw_results": raw_results,
            "metadata": {
                "total_evaluated": len(texts),
                "source": str(eval_path),
            },
        }

    # ------------------------------------------------------------------ #
    #  Metrics                                                           #
    # ------------------------------------------------------------------ #

    def compute_metrics(
        self, y_true: list[str], y_pred: list[str]
    ) -> dict[str, Any]:
        """Compute overall and per-intent classification metrics.

        Args:
            y_true: Ground-truth intent labels.
            y_pred: Predicted intent labels.

        Returns:
            Dict with ``overall_accuracy``, ``macro_precision``,
            ``macro_recall``, ``macro_f1``, and ``per_intent`` breakdown.
        """
        overall_accuracy: float = accuracy_score(y_true, y_pred)
        macro_precision: float = precision_score(
            y_true, y_pred, average="macro", zero_division=0
        )
        macro_recall: float = recall_score(
            y_true, y_pred, average="macro", zero_division=0
        )
        macro_f1: float = f1_score(
            y_true, y_pred, average="macro", zero_division=0
        )

        # Per-intent breakdown via classification_report
        report: dict = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )

        per_intent: dict[str, dict[str, float]] = {}
        for label, metrics in report.items():
            if label in ("accuracy", "macro avg", "weighted avg"):
                continue
            if isinstance(metrics, dict):
                per_intent[label] = {
                    "precision": round(metrics.get("precision", 0.0), 4),
                    "recall": round(metrics.get("recall", 0.0), 4),
                    "f1": round(metrics.get("f1-score", 0.0), 4),
                    "support": int(metrics.get("support", 0)),
                }

        return {
            "overall_accuracy": round(overall_accuracy, 4),
            "macro_precision": round(macro_precision, 4),
            "macro_recall": round(macro_recall, 4),
            "macro_f1": round(macro_f1, 4),
            "per_intent": per_intent,
        }

    # ------------------------------------------------------------------ #
    #  Confusion Matrix                                                  #
    # ------------------------------------------------------------------ #

    def generate_confusion_matrix(
        self,
        y_true: list[str],
        y_pred: list[str],
        labels: Optional[list[str]] = None,
        top_n: int = 30,
    ) -> plt.Figure:
        """Generate a confusion-matrix heatmap.

        When there are more than *top_n* labels, only the most frequent
        labels (by ground-truth count) are shown.

        Args:
            y_true: Ground-truth labels.
            y_pred: Predicted labels.
            labels: Explicit label ordering. If None, derived from data.
            top_n: Maximum labels to display.

        Returns:
            A :class:`matplotlib.figure.Figure` with the heatmap.
        """
        if labels is None:
            all_labels = sorted(set(y_true) | set(y_pred))
        else:
            all_labels = labels

        # Trim to top-N most frequent in y_true
        if len(all_labels) > top_n:
            freq = Counter(y_true)
            all_labels = [lbl for lbl, _ in freq.most_common(top_n)]

        cm = confusion_matrix(y_true, y_pred, labels=all_labels)

        n = len(all_labels)
        figsize = (max(10, n * 0.45), max(8, n * 0.4))
        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(
            cm,
            annot=n <= 25,
            fmt="d",
            cmap="Blues",
            xticklabels=all_labels,
            yticklabels=all_labels,
            ax=ax,
        )
        ax.set_xlabel("Predicted Intent")
        ax.set_ylabel("True Intent")
        ax.set_title(f"Confusion Matrix (top {min(top_n, n)} intents)")
        plt.xticks(rotation=45, ha="right", fontsize=7)
        plt.yticks(fontsize=7)
        plt.tight_layout()

        return fig

    # ------------------------------------------------------------------ #
    #  Error Analysis                                                    #
    # ------------------------------------------------------------------ #

    def get_error_analysis(
        self,
        y_true: list[str],
        y_pred: list[str],
        texts: list[str],
    ) -> pd.DataFrame:
        """Build a DataFrame of misclassified samples.

        Args:
            y_true: Ground-truth labels.
            y_pred: Predicted labels.
            texts: Corresponding input texts.

        Returns:
            DataFrame with columns ``text``, ``true_intent``,
            ``predicted_intent``, sorted by ``true_intent``.
        """
        errors: list[dict[str, str]] = []
        for text, true, pred in zip(texts, y_true, y_pred):
            if true != pred:
                errors.append(
                    {
                        "text": text,
                        "true_intent": true,
                        "predicted_intent": pred,
                    }
                )

        df = pd.DataFrame(errors)
        if not df.empty:
            df = df.sort_values("true_intent").reset_index(drop=True)
        return df

    # ------------------------------------------------------------------ #
    #  Persistence                                                       #
    # ------------------------------------------------------------------ #

    def save_results(
        self,
        results: dict[str, Any],
        output_path: str | Path = "data/eval_results.json",
    ) -> None:
        """Save evaluation results to a JSON file.

        Args:
            results: The full results dict from :meth:`run_evaluation` or
                :meth:`compute_metrics`.
            output_path: Destination file path.
        """
        save_json_file(results, Path(output_path))
        logger.info("Evaluation results saved to %s", output_path)

    # ------------------------------------------------------------------ #
    #  Helpers                                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _balanced_sample(
        samples: list[dict[str, Any]],
        target_size: int,
        seed: int = 42,
    ) -> list[dict[str, Any]]:
        """Create a balanced random sample from evalution data.

        Tries to pick an equal number from each intent. If that doesn't fill
        *target_size*, fills remaining slots randomly.

        Args:
            samples: Full list of eval samples.
            target_size: Desired sample count.
            seed: Random seed.

        Returns:
            Sub-sampled list.
        """
        rng = random.Random(seed)
        intent_groups: dict[str, list[dict]] = {}
        for s in samples:
            intent_groups.setdefault(s["true_intent"], []).append(s)

        n_intents = len(intent_groups)
        per_intent = max(1, target_size // n_intents)

        chosen: list[dict] = []
        for intent, group in intent_groups.items():
            rng.shuffle(group)
            chosen.extend(group[:per_intent])

        # Fill remaining if needed
        if len(chosen) < target_size:
            remaining = [s for s in samples if s not in chosen]
            rng.shuffle(remaining)
            chosen.extend(remaining[: target_size - len(chosen)])

        rng.shuffle(chosen)
        return chosen[:target_size]
