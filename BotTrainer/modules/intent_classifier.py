"""
Intent Classifier for BotTrainer.
=====================================
Uses TF-IDF for smart few-shot selection and Gemini 2.5 Flash for classification.
"""

import hashlib
import logging
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from BotTrainer.config import config as app_config
from BotTrainer.modules.exceptions import LLMError, ParseError
from BotTrainer.modules.llm_client import GeminiClient
from BotTrainer.modules.utils import load_json_file, safe_json_loads, sanitize_input, validate_intent_result

logger = logging.getLogger(__name__)

# Prompt template lives alongside the project
PROMPT_TEMPLATE_PATH: Path = Path(__file__).resolve().parent.parent / "prompts" / "intent_prompt.txt"
_TFIDF_CACHE_DIR: Path = Path(__file__).resolve().parent.parent / "data"


class IntentClassifier:
    """Intent classifier combining TF-IDF retrieval with LLM classification.

    The classifier first narrows the 151 CLINC150 intents down to the most
    relevant *top_k* using TF-IDF cosine similarity, then asks Gemini to
    pick the best match from that reduced candidate set.

    Attributes:
        llm: The Gemini client used for classification.
        n_few_shot: Number of example utterances per intent in the prompt.
        intent_names: Sorted list of all known intent names.
        known_intents: Set of all known intent names (for validation).
        intent_to_examples: Mapping ``{intent: [example_utterances]}``.
    """

    def __init__(
        self,
        intents_json_path: str | Path,
        llm_client: GeminiClient,
        n_few_shot: int = 3,
    ) -> None:
        """Load intents and build the TF-IDF index.

        Args:
            intents_json_path: Path to ``intents.json``.
            llm_client: An initialised :class:`GeminiClient`.
            n_few_shot: Max examples per intent included in prompts.

        Raises:
            FileNotFoundError: If the intents file does not exist.
            ValueError: If the intents file is malformed.
        """
        self.llm: GeminiClient = llm_client
        self.n_few_shot: int = n_few_shot

        intents_path = Path(intents_json_path)
        data = load_json_file(intents_path)
        if data is None:
            raise FileNotFoundError(f"Intents file not found or invalid: {intents_path}")

        intents_list: list[dict[str, Any]] = data.get("intents", [])
        if not intents_list:
            raise ValueError("intents.json contains no intents.")

        # Build internal index
        self.intent_to_examples: dict[str, list[str]] = {}
        for entry in intents_list:
            name: str = entry["name"]
            examples: list[str] = entry.get("examples", [])
            self.intent_to_examples[name] = examples

        self.intent_names: list[str] = sorted(self.intent_to_examples.keys())
        self.known_intents: set[str] = set(self.intent_names)

        # Build TF-IDF over all examples for retrieval
        self._example_texts: list[str] = []
        self._example_intents: list[str] = []
        for name in self.intent_names:
            for ex in self.intent_to_examples[name]:
                self._example_texts.append(ex)
                self._example_intents.append(name)

        self._vectorizer, self._tfidf_matrix = self._load_or_build_tfidf(intents_path)

        # Load prompt template
        self._prompt_template: str = self._load_prompt_template()

        # Response cache: message -> result (avoids duplicate API calls)
        self._cache: dict[str, dict[str, Any]] = {}
        self._cache_max_size: int = app_config.cache_max_size

        logger.info(
            "IntentClassifier ready — %d intents, %d examples, n_few_shot=%d.",
            len(self.intent_names),
            len(self._example_texts),
            self.n_few_shot,
        )

    # ------------------------------------------------------------------ #
    #  TF-IDF Disk Cache                                                 #
    # ------------------------------------------------------------------ #

    def _load_or_build_tfidf(self, intents_path: Path):
        """Load cached TF-IDF artifacts or build and cache them."""
        data_hash = hashlib.md5(intents_path.read_bytes()).hexdigest()  # noqa: S324
        cache_path = _TFIDF_CACHE_DIR / ".tfidf_cache.pkl"

        if cache_path.exists():
            try:
                cached = joblib.load(cache_path)
                if cached.get("hash") == data_hash:
                    logger.info("TF-IDF cache hit — skipping rebuild.")
                    return cached["vectorizer"], cached["matrix"]
            except Exception:
                logger.warning("Corrupt TF-IDF cache — rebuilding.")

        vectorizer = TfidfVectorizer(
            max_features=app_config.tfidf_max_features,
            ngram_range=app_config.tfidf_ngram_range,
            stop_words="english",
            sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform(self._example_texts)

        try:
            joblib.dump({"hash": data_hash, "vectorizer": vectorizer, "matrix": matrix}, cache_path)
            logger.info("TF-IDF cache saved to %s", cache_path)
        except OSError as exc:
            logger.warning("Could not save TF-IDF cache: %s", exc)

        return vectorizer, matrix

    # ------------------------------------------------------------------ #
    #  TF-IDF Retrieval                                                  #
    # ------------------------------------------------------------------ #

    def _select_relevant_intents(
        self, user_message: str, top_k: int = 20
    ) -> list[tuple[str, list[str]]]:
        """Find the most relevant intents for a user message via TF-IDF.

        Uses max + mean score blending per intent for better retrieval and
        includes a diversity pass to catch intents from different domains.

        Args:
            user_message: The input text to classify.
            top_k: Maximum number of candidate intents to return.

        Returns:
            List of ``(intent_name, [examples])`` tuples ranked by relevance.
        """
        query_vec = self._vectorizer.transform([user_message])
        scores = cosine_similarity(query_vec, self._tfidf_matrix).flatten()

        # Aggregate scores by intent: blend max + mean for robustness
        intent_max_scores: dict[str, float] = {}
        intent_sum_scores: dict[str, float] = {}
        intent_count: dict[str, int] = {}
        for idx, score in enumerate(scores):
            intent_name = self._example_intents[idx]
            if intent_name not in intent_max_scores or score > intent_max_scores[intent_name]:
                intent_max_scores[intent_name] = score
            intent_sum_scores[intent_name] = intent_sum_scores.get(intent_name, 0.0) + score
            intent_count[intent_name] = intent_count.get(intent_name, 0) + 1

        # Blended score: 0.7 * max + 0.3 * mean
        intent_scores: dict[str, float] = {}
        for name in intent_max_scores:
            mean_score = intent_sum_scores[name] / intent_count[name]
            intent_scores[name] = 0.7 * intent_max_scores[name] + 0.3 * mean_score

        # Sort by score descending and take top_k
        ranked = sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        result: list[tuple[str, list[str]]] = []
        for intent_name, _score in ranked:
            examples = self.intent_to_examples[intent_name][: self.n_few_shot]
            result.append((intent_name, examples))

        return result

    # ------------------------------------------------------------------ #
    #  Prompt Construction                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _load_prompt_template() -> str:
        """Load the intent prompt template from disk.

        Returns:
            The template string with ``{few_shot_examples}`` and
            ``{user_message}`` placeholders.
        """
        try:
            return PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
        except OSError:
            logger.warning("Prompt template not found at %s — using fallback.", PROMPT_TEMPLATE_PATH)
            return (
                "Classify the intent of this message. "
                "Available intents:\n{few_shot_examples}\n\n"
                'User message: "{user_message}"\n\n'
                "Return JSON: {\"intent\": \"<name>\", \"confidence\": <float>, \"reasoning\": \"<text>\"}"
            )

    def _build_few_shot_block(self, relevant_intents: list[tuple[str, list[str]]]) -> str:
        """Format selected intents and examples into a prompt block.

        Args:
            relevant_intents: Output of :meth:`_select_relevant_intents`.

        Returns:
            Formatted multi-line string suitable for prompt injection.
        """
        lines: list[str] = []
        for intent_name, examples in relevant_intents:
            example_quoted = ", ".join(f'"{ex}"' for ex in examples)
            lines.append(f"- {intent_name}: [{example_quoted}]")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  Classification                                                    #
    # ------------------------------------------------------------------ #

    def classify(self, user_message: str) -> dict[str, Any]:
        """Classify a single user message.

        Pipeline:
        1. Check cache for previous identical query.
        2. Select top-k relevant intents via TF-IDF.
        3. Build few-shot prompt.
        4. Query Gemini with JSON mode.
        5. Parse, validate, and cache the response.

        Args:
            user_message: The text to classify.

        Returns:
            Dict with keys ``intent``, ``confidence``, ``reasoning``.
            On failure, ``intent`` is ``"parse_error"``.
        """
        # Check cache first
        user_message = sanitize_input(user_message)
        cache_key = user_message.strip().lower()
        if cache_key in self._cache:
            logger.debug("Cache hit for '%.60s'", user_message)
            return self._cache[cache_key]

        try:
            relevant = self._select_relevant_intents(user_message, top_k=30)
            few_shot_block = self._build_few_shot_block(relevant)

            prompt = self._prompt_template.replace("{few_shot_examples}", few_shot_block)
            prompt = prompt.replace("{user_message}", user_message)

            raw_response = self.llm.query(prompt, use_json_mode=True, max_tokens=1024)
            parsed = safe_json_loads(raw_response)

            if parsed is None:
                return {
                    "intent": "parse_error",
                    "confidence": 0.0,
                    "reasoning": f"Failed to parse LLM response: {raw_response[:200]}",
                }

            result = validate_intent_result(parsed, self.known_intents)

            # Cache the result (evict oldest if full)
            if len(self._cache) >= self._cache_max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            self._cache[cache_key] = result

            return result

        except LLMError:
            raise
        except Exception as exc:
            logger.error("Classification failed for '%.80s': %s", user_message, exc)
            return {
                "intent": "parse_error",
                "confidence": 0.0,
                "reasoning": str(exc),
            }

    def batch_classify(
        self,
        messages: list[str],
        show_progress: bool = True,
    ) -> list[dict[str, Any]]:
        """Classify a batch of messages.

        Args:
            messages: List of user utterances.
            show_progress: Show a ``tqdm`` progress bar.

        Returns:
            List of classification result dicts (same order as input).
        """
        results: list[dict[str, Any]] = []
        iterator = tqdm(messages, desc="Classifying", disable=not show_progress)
        for msg in iterator:
            result = self.classify(msg)
            results.append(result)
            if show_progress:
                iterator.set_postfix(intent=result.get("intent", "?")[:20])
        return results


# ── Standalone demo ───────────────────────────────────────────────────
if __name__ == "__main__":
    from BotTrainer.modules.utils import setup_logging

    setup_logging()

    print("═" * 50)
    print("  Module 3: intent_classifier.py — Classifier Demo")
    print("═" * 50)

    # Resolve intents.json path
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    _INTENTS_PATH = _PROJECT_ROOT / "data" / "intents.json"

    if not _INTENTS_PATH.exists():
        print(f"\n❌ intents.json not found at {_INTENTS_PATH}")
        print("   Run 'python -m BotTrainer.modules.data_loader' first.")
    else:
        try:
            llm = GeminiClient()
            classifier = IntentClassifier(
                intents_json_path=_INTENTS_PATH,
                llm_client=llm,
                n_few_shot=3,
            )
            print(f"\n✅ IntentClassifier loaded")
            print(f"   Intents : {len(classifier.intent_names)}")
            print(f"   Examples: {len(classifier._example_texts)}")

            # Classify sample messages
            test_messages = [
                "What's my account balance?",
                "Book a flight to Mumbai",
                "Tell me a joke",
            ]
            print("\n⏳ Classifying sample messages ...\n")
            for msg in test_messages:
                result = classifier.classify(msg)
                print(f"   💬 \"{msg}\"")
                print(f"      → Intent: {result['intent']} ({result['confidence']:.0%})")
                print()

        except ValueError as exc:
            print(f"\n❌ {exc}")
        except Exception as exc:
            print(f"\n❌ Error: {exc}")

    print()
