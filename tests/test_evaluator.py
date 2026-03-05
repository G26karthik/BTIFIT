"""Tests for BotTrainer.modules.evaluator."""

import json

import pytest

from BotTrainer.modules.evaluator import Evaluator
from BotTrainer.modules.intent_classifier import IntentClassifier


def test_compute_metrics():
    ev = Evaluator()
    y_true = ["balance", "weather", "balance", "weather"]
    y_pred = ["balance", "weather", "weather", "weather"]
    metrics = ev.compute_metrics(y_true, y_pred)
    assert 0.0 <= metrics["overall_accuracy"] <= 1.0
    assert "macro_f1" in metrics
    assert "per_intent" in metrics


def test_error_analysis():
    ev = Evaluator()
    y_true = ["a", "b", "c"]
    y_pred = ["a", "c", "c"]  # 1 error: b->c
    df = ev.get_error_analysis(y_true, y_pred, ["t1", "t2", "t3"])
    assert len(df) == 1
    assert df.iloc[0]["true_intent"] == "b"
    assert df.iloc[0]["predicted_intent"] == "c"


def test_run_evaluation_with_progress(sample_intents_file, mock_llm, tmp_path):
    # Create eval dataset
    eval_data = {
        "eval_samples": [
            {"text": "What is my balance?", "true_intent": "balance"},
            {"text": "Book a flight", "true_intent": "book_flight"},
        ]
    }
    eval_path = tmp_path / "eval_dataset.json"
    eval_path.write_text(json.dumps(eval_data), encoding="utf-8")

    # Mock LLM to return a valid classification
    def fake_query(*args, **kwargs):
        return '{"intent": "balance", "confidence": 0.9, "reasoning": "test"}'

    mock_llm.query = fake_query

    clf = IntentClassifier(sample_intents_file, mock_llm, n_few_shot=2)
    ev = Evaluator()

    progress_calls = []

    def on_progress(current, total):
        progress_calls.append((current, total))

    results = ev.run_evaluation(eval_path, clf, progress_callback=on_progress)
    assert len(results["y_pred"]) == 2
    assert len(progress_calls) == 2
    assert progress_calls[-1] == (2, 2)
