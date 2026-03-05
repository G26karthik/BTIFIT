"""Tests for BotTrainer.modules.llm_client."""

import pytest

from BotTrainer.modules.exceptions import LLMError
from BotTrainer.modules.llm_client import GeminiClient


def test_init_raises_without_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    # Ensure config also has no key
    import BotTrainer.modules.llm_client as llm_mod
    from BotTrainer.config import Config
    monkeypatch.setattr(llm_mod, "app_config", Config(gemini_api_key=""))
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        GeminiClient(api_key="", model="test")


def test_retries_exhaust_raises_llm_error(mock_llm):
    """After all retries, LLMError should be raised."""
    call_count = 0

    def fake_generate(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise ConnectionError("test failure")

    class FakeModels:
        generate_content = staticmethod(fake_generate)

    class FakeClient:
        models = FakeModels()

    mock_llm.client = FakeClient()

    with pytest.raises(LLMError, match="retries exhausted"):
        mock_llm.query("test", use_json_mode=False)

    assert call_count == 3  # default retry_attempts is 3
