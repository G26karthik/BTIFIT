"""
Entity Extractor for BotTrainer.
====================================
Uses Gemini 2.5 Flash to extract named entities from user messages.
"""

import logging
from pathlib import Path
from typing import Any

from modules.llm_client import GeminiClient
from modules.utils import safe_json_loads

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE_PATH: Path = (
    Path(__file__).resolve().parent.parent / "prompts" / "entity_prompt.txt"
)


class EntityExtractor:
    """Extract named entities from user messages using Gemini.

    Attributes:
        llm: The Gemini client used for extraction.
    """

    def __init__(self, llm_client: GeminiClient) -> None:
        """Initialise the extractor.

        Args:
            llm_client: An initialised :class:`GeminiClient`.
        """
        self.llm: GeminiClient = llm_client
        self._prompt_template: str = self._load_prompt_template()
        self._cache: dict[str, dict[str, str]] = {}
        self._cache_max_size: int = 500
        logger.info("EntityExtractor ready.")

    # ------------------------------------------------------------------ #
    #  Internal                                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _load_prompt_template() -> str:
        """Load the entity extraction prompt template from disk.

        Returns:
            Template string with ``{predicted_intent}`` and ``{user_message}``
            placeholders.
        """
        try:
            return PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
        except OSError:
            logger.warning(
                "Entity prompt template not found at %s — using fallback.",
                PROMPT_TEMPLATE_PATH,
            )
            return (
                "Extract named entities from this message.\n"
                'Intent: "{predicted_intent}"\n'
                'Message: "{user_message}"\n\n'
                "Return JSON mapping entity_type to value. "
                "Return {} if no entities found."
            )

    # ------------------------------------------------------------------ #
    #  Public API                                                        #
    # ------------------------------------------------------------------ #

    def extract(
        self,
        user_message: str,
        predicted_intent: str,
    ) -> dict[str, str]:
        """Extract entities from a user message.

        Builds a prompt from the template, queries Gemini in JSON mode, and
        parses the response into a flat ``{entity_type: value}`` dict.

        Args:
            user_message: The input text.
            predicted_intent: The intent already predicted for this message.

        Returns:
            Dictionary of ``{entity_type: entity_value}``, or ``{}`` if no
            entities were found or an error occurred.
        """
        # Check cache first
        cache_key = f"{user_message.strip().lower()}|{predicted_intent}"
        if cache_key in self._cache:
            logger.debug("Entity cache hit for '%.60s'", user_message)
            return self._cache[cache_key]

        try:
            prompt = self._prompt_template.replace("{predicted_intent}", predicted_intent)
            prompt = prompt.replace("{user_message}", user_message)

            raw_response: str = self.llm.query(
                prompt,
                use_json_mode=True,
                temperature=0.0,
                max_tokens=512,
            )

            parsed: Any = safe_json_loads(raw_response)

            if parsed is None:
                logger.warning(
                    "Entity extraction returned unparseable response for '%.80s'.",
                    user_message,
                )
                return {}

            if not isinstance(parsed, dict):
                logger.warning(
                    "Entity extraction returned non-dict (%s) for '%.80s'.",
                    type(parsed).__name__,
                    user_message,
                )
                return {}

            # Ensure all values are strings
            entities: dict[str, str] = {}
            for key, value in parsed.items():
                entities[str(key)] = str(value)

            # Cache the result
            if len(self._cache) >= self._cache_max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            self._cache[cache_key] = entities

            return entities

        except Exception as exc:
            logger.error(
                "Entity extraction failed for '%.80s': %s",
                user_message,
                exc,
            )
            return {}
