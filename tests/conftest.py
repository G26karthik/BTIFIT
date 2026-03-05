"""Shared test fixtures for BotTrainer."""

import pytest

from BotTrainer.modules.llm_client import GeminiClient


@pytest.fixture()
def sample_intents_data():
    """Minimal intents.json payload for testing."""
    return {
        "intents": [
            {
                "name": "balance",
                "examples": ["What is my balance?", "Show me my balance", "Check account balance"],
            },
            {
                "name": "book_flight",
                "examples": ["Book a flight to Mumbai", "I need a flight", "Fly to Delhi"],
            },
            {
                "name": "weather",
                "examples": ["What's the weather?", "Weather forecast", "Is it raining?"],
            },
        ]
    }


@pytest.fixture()
def sample_intents_file(tmp_path, sample_intents_data):
    """Write sample intents to a temp file and return its path."""
    import json
    path = tmp_path / "intents.json"
    path.write_text(json.dumps(sample_intents_data), encoding="utf-8")
    return path


@pytest.fixture()
def mock_llm(monkeypatch):
    """GeminiClient that never calls the real API."""
    client = object.__new__(GeminiClient)
    client.primary_model = "test-model"
    client.fast_model = "test-model-lite"
    client.request_count = 0
    client.last_request_time = None
    return client
