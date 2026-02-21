"""
NLU Pipeline for BotTrainer.
================================
Orchestrates intent classification and entity extraction into a single
predict() call, plus an interactive CLI mode.
"""

import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from modules.entity_extractor import EntityExtractor
from modules.intent_classifier import IntentClassifier
from modules.llm_client import GeminiClient
from modules.utils import setup_logging

logger = logging.getLogger(__name__)

PROJECT_ROOT: Path = Path(__file__).resolve().parent
INTENTS_PATH: Path = PROJECT_ROOT / "data" / "intents.json"


class NLUPipeline:
    """End-to-end NLU pipeline: intent classification + entity extraction.

    Attributes:
        llm: Shared Gemini client.
        classifier: TF-IDF + LLM intent classifier.
        extractor: LLM-based entity extractor.
    """

    def __init__(
        self,
        intents_path: str | Path = INTENTS_PATH,
        n_few_shot: int = 5,
    ) -> None:
        """Initialise all pipeline components.

        Args:
            intents_path: Path to ``intents.json``.
            n_few_shot: Examples per intent in the LLM prompt.
        """
        self.llm: GeminiClient = GeminiClient()
        self.classifier: IntentClassifier = IntentClassifier(
            intents_json_path=intents_path,
            llm_client=self.llm,
            n_few_shot=n_few_shot,
        )
        self.extractor: EntityExtractor = EntityExtractor(llm_client=self.llm)
        logger.info("NLUPipeline ready.")

    def predict(self, user_message: str) -> dict[str, Any]:
        """Run the full NLU pipeline on a single message.

        1. Classify intent (+ entity extraction runs concurrently with
           a preliminary intent for speed when possible).
        2. Package results with timing info.

        Args:
            user_message: The input text to analyse.

        Returns:
            Dict with keys ``user_message``, ``intent``, ``confidence``,
            ``entities``, ``reasoning``, ``timestamp``, ``model``,
            ``latency_ms``.
        """
        import time
        start = time.perf_counter()

        intent_result: dict[str, Any] = self.classifier.classify(user_message)
        intent_name = intent_result.get("intent", "unknown")

        entities: dict[str, str] = self.extractor.extract(
            user_message=user_message,
            predicted_intent=intent_name,
        )

        elapsed_ms = round((time.perf_counter() - start) * 1000)

        return {
            "user_message": user_message,
            "intent": intent_name,
            "confidence": intent_result.get("confidence", 0.0),
            "entities": entities,
            "reasoning": intent_result.get("reasoning", ""),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": self.llm.primary_model,
            "latency_ms": elapsed_ms,
        }


# -------------------------------------------------------------------- #
#  Interactive CLI                                                      #
# -------------------------------------------------------------------- #


def _print_banner() -> None:
    """Print the BotTrainer welcome banner."""
    banner = r"""
╔══════════════════════════════════════════════════════╗
║           🤖  BotTrainer — NLU Pipeline  🤖          ║
║          Powered by Gemini 2.5 Flash                 ║
╚══════════════════════════════════════════════════════╝
Type a message and press Enter to classify it.
Type 'exit' or 'quit' to stop.
"""
    print(banner)


def _format_result(result: dict[str, Any]) -> str:
    """Pretty-format a pipeline result for the terminal.

    Args:
        result: Output of :meth:`NLUPipeline.predict`.

    Returns:
        Multi-line formatted string.
    """
    lines = [
        "─" * 50,
        f"  Intent     : {result['intent']}",
        f"  Confidence : {result['confidence']:.2%}",
        f"  Entities   : {result['entities'] if result['entities'] else '(none)'}",
        f"  Reasoning  : {result['reasoning']}",
        f"  Model      : {result['model']}",
        "─" * 50,
    ]
    return "\n".join(lines)


def main() -> None:
    """Launch the interactive CLI loop."""
    setup_logging()
    _print_banner()

    try:
        pipeline = NLUPipeline()
    except Exception as exc:
        print(f"❌ Failed to initialise pipeline: {exc}")
        sys.exit(1)

    while True:
        try:
            user_input = input("\n💬 You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("👋 Goodbye!")
                break

            print("🤖 Analysing …")
            result = pipeline.predict(user_input)
            print(_format_result(result))

        except KeyboardInterrupt:
            print("\n👋 Interrupted — goodbye!")
            break
        except Exception as exc:
            print(f"⚠️  Error: {exc}")


if __name__ == "__main__":
    main()
