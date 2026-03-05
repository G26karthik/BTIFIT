"""Tests for BotTrainer.modules.utils."""

from BotTrainer.modules.utils import (
    esc,
    safe_json_loads,
    sanitize_input,
    validate_intent_result,
)


class TestSafeJsonLoads:
    def test_valid_json(self):
        assert safe_json_loads('{"a": 1}') == {"a": 1}

    def test_markdown_fenced(self):
        assert safe_json_loads('```json\n{"a": 1}\n```') == {"a": 1}

    def test_empty_returns_none(self):
        assert safe_json_loads("") is None

    def test_non_string_returns_none(self):
        assert safe_json_loads(None) is None  # type: ignore[arg-type]

    def test_truncated_json_recovery(self):
        result = safe_json_loads('{"intent": "balance", "confidence": 0.9')
        assert result is not None
        assert result["intent"] == "balance"


class TestSanitizeInput:
    def test_strips_control_chars(self):
        assert sanitize_input("hello\x00world") == "helloworld"

    def test_truncates_long_input(self):
        assert len(sanitize_input("a" * 2000, max_len=100)) == 100

    def test_strips_whitespace(self):
        assert sanitize_input("  hello  ") == "hello"


class TestEsc:
    def test_escapes_html(self):
        assert esc("<script>alert(1)</script>") == "&lt;script&gt;alert(1)&lt;/script&gt;"


class TestValidateIntentResult:
    def test_valid_result(self):
        known = {"balance", "weather"}
        result = validate_intent_result(
            {"intent": "balance", "confidence": 0.95, "reasoning": "ok"}, known
        )
        assert result["intent"] == "balance"
        assert result["confidence"] == 0.95

    def test_unknown_intent_falls_back(self):
        known = {"balance"}
        result = validate_intent_result({"intent": "unknown_xyz"}, known)
        assert result["intent"] == "out_of_scope"

    def test_non_dict_returns_error(self):
        result = validate_intent_result("not a dict", set())  # type: ignore[arg-type]
        assert result["intent"] == "parse_error"

    def test_clamps_confidence(self):
        result = validate_intent_result({"intent": "balance", "confidence": 5.0}, {"balance"})
        assert result["confidence"] == 1.0
