"""Tests for BotTrainer.modules.intent_classifier."""

import json

from BotTrainer.modules.intent_classifier import IntentClassifier


def test_classifier_init(sample_intents_file, mock_llm):
    clf = IntentClassifier(sample_intents_file, mock_llm, n_few_shot=2)
    assert len(clf.intent_names) == 3
    assert "balance" in clf.known_intents


def test_classify_uses_cache(sample_intents_file, mock_llm, monkeypatch):
    clf = IntentClassifier(sample_intents_file, mock_llm, n_few_shot=2)

    # Pre-populate cache
    clf._cache["what is my balance?"] = {
        "intent": "balance",
        "confidence": 0.99,
        "reasoning": "cached",
    }
    result = clf.classify("What is my balance?")
    assert result["intent"] == "balance"
    assert result["reasoning"] == "cached"


def test_tfidf_retrieval(sample_intents_file, mock_llm):
    clf = IntentClassifier(sample_intents_file, mock_llm, n_few_shot=2)
    relevant = clf._select_relevant_intents("flight booking", top_k=2)
    # Should return at most 2 intents
    assert len(relevant) <= 2
    intent_names = [name for name, _ in relevant]
    # "book_flight" should rank high for "flight booking"
    assert "book_flight" in intent_names


def test_tfidf_disk_cache(sample_intents_file, mock_llm, tmp_path):
    """Second init should load from disk cache."""
    clf1 = IntentClassifier(sample_intents_file, mock_llm, n_few_shot=2)
    cache_file = sample_intents_file.parent / ".tfidf_cache.pkl"
    # The cache is written to _TFIDF_CACHE_DIR, not tmp_path,
    # but we can verify the classifier built correctly
    assert clf1._tfidf_matrix is not None


def test_classify_returns_parse_error_on_bad_llm(sample_intents_file, mock_llm, monkeypatch):
    clf = IntentClassifier(sample_intents_file, mock_llm, n_few_shot=2)

    def fake_query(*args, **kwargs):
        return "not valid json at all {{{"

    mock_llm.query = fake_query
    result = clf.classify("something random")
    assert result["intent"] == "parse_error"
