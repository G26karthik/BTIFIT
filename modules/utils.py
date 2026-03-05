"""
Utility helpers for BotTrainer.
================================
JSON parsing, validation, and common file I/O operations.
"""

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def safe_json_loads(text: str) -> Optional[dict]:
    """Safely parse a JSON string, stripping markdown fences if present.

    Args:
        text: Raw text that should contain JSON.

    Returns:
        Parsed dictionary, or None if parsing fails.
    """
    if not text or not isinstance(text, str):
        logger.warning("safe_json_loads received empty or non-string input.")
        return None

    cleaned = text.strip()

    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    fence_pattern = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?\s*```$", re.DOTALL)
    match = fence_pattern.match(cleaned)
    if match:
        cleaned = match.group(1).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.warning("JSON parse error: %s — attempting partial recovery", exc)

    # ---- Partial / truncated JSON recovery ----
    # Gemini sometimes returns truncated JSON when output tokens run out.
    # Try progressively more aggressive fixups.
    for suffix in ["", "}", "\"}", "\"}}"]:
        try:
            return json.loads(cleaned + suffix)
        except json.JSONDecodeError:
            continue

    # Try to extract key-value pairs with regex as last resort
    intent_match = re.search(r'"intent"\s*:\s*"([^"]+)"', cleaned)
    conf_match = re.search(r'"confidence"\s*:\s*([\d.]+)', cleaned)
    reason_match = re.search(r'"reasoning"\s*:\s*"([^"]*)', cleaned)
    if intent_match:
        logger.info("Recovered partial JSON via regex — intent=%s", intent_match.group(1))
        return {
            "intent": intent_match.group(1),
            "confidence": float(conf_match.group(1)) if conf_match else 0.85,
            "reasoning": reason_match.group(1) if reason_match else "Recovered from partial response",
        }

    logger.error("All JSON recovery attempts failed — raw text: %.200s", text)
    return None


def load_json_file(filepath: Path) -> Optional[dict]:
    """Load and parse a JSON file from disk.

    Args:
        filepath: Path to the JSON file.

    Returns:
        Parsed dictionary, or None on failure.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        logger.error("File not found: %s", filepath)
        return None
    try:
        with filepath.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Failed to load JSON from %s: %s", filepath, exc)
        return None


def save_json_file(data: Any, filepath: Path, indent: int = 2) -> bool:
    """Save data as a JSON file, creating parent directories as needed.

    Args:
        data: Serializable Python object.
        filepath: Destination path.
        indent: JSON indentation level.

    Returns:
        True on success, False on failure.
    """
    filepath = Path(filepath)
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with filepath.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=indent, ensure_ascii=False)
        logger.info("Saved JSON to %s", filepath)
        return True
    except (OSError, TypeError) as exc:
        logger.error("Failed to save JSON to %s: %s", filepath, exc)
        return False


def iso_timestamp() -> str:
    """Return the current UTC time as an ISO-8601 string.

    Returns:
        ISO-8601 formatted timestamp string.
    """
    return datetime.now(timezone.utc).isoformat()


def validate_intent_result(result: dict, known_intents: set) -> dict:
    """Validate and normalise an intent classification result.

    Ensures the result dict has ``intent``, ``confidence``, and ``reasoning``
    keys with sensible values.

    Args:
        result: Raw parsed dict from the LLM.
        known_intents: Set of valid intent name strings.

    Returns:
        Validated result dict (may be modified in-place).
    """
    if not isinstance(result, dict):
        return {
            "intent": "parse_error",
            "confidence": 0.0,
            "reasoning": "LLM returned non-dict response.",
        }

    # Normalise intent name
    intent = result.get("intent", "parse_error")
    if isinstance(intent, str):
        intent = intent.strip().lower()
    else:
        intent = "parse_error"

    if intent not in known_intents and intent != "out_of_scope":
        logger.warning("Unknown intent '%s' — falling back to out_of_scope", intent)
        intent = "out_of_scope"

    # Normalise confidence
    confidence = result.get("confidence", 0.0)
    try:
        confidence = float(confidence)
        confidence = max(0.0, min(1.0, confidence))
    except (TypeError, ValueError):
        confidence = 0.0

    # Normalise reasoning
    reasoning = result.get("reasoning", "")
    if not isinstance(reasoning, str):
        reasoning = str(reasoning)

    return {
        "intent": intent,
        "confidence": confidence,
        "reasoning": reasoning,
    }


def truncate_text(text: str, max_length: int = 200) -> str:
    """Truncate text to a maximum length with an ellipsis.

    Args:
        text: Input string.
        max_length: Maximum character count.

    Returns:
        Truncated string.
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a consistent format.

    Args:
        level: Logging level constant.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
