"""
Gemini LLM Client for BotTrainer.
====================================
Wraps the google-genai SDK for intent classification and entity extraction.
Uses gemini-2.5-flash (primary) and gemini-2.5-flash-lite (fast fallback).
"""

import logging
import random
import time
from typing import Optional

from google import genai
from google.genai import types
from google.genai.errors import APIError

from BotTrainer.config import config as app_config
from BotTrainer.modules.exceptions import LLMError

logger = logging.getLogger(__name__)


class GeminiClient:
    """Thin wrapper around the Google GenAI SDK for BotTrainer.

    Attributes:
        client: The underlying ``genai.Client`` instance.
        primary_model: Default model identifier.
        fast_model: Lighter model for high-volume batches.
        request_count: Running count of API calls made in this session.
        last_request_time: Epoch time of the most recent request (for rate limiting).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        """Initialise the Gemini client.

        Args:
            api_key: Gemini API key. Falls back to config/env if not provided.
            model: Override the primary model identifier.

        Raises:
            ValueError: If no API key is available.
        """
        api_key = api_key or app_config.gemini_api_key
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. "
                "Add it to your .env file (see .env.example)."
            )

        self.client: genai.Client = genai.Client(api_key=api_key)
        self.primary_model: str = model or app_config.primary_model
        self.fast_model: str = app_config.fast_model
        self.request_count: int = 0
        self.last_request_time: Optional[float] = None

        logger.info(
            "GeminiClient initialised — primary=%s, fast=%s",
            self.primary_model,
            self.fast_model,
        )

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def query(
        self,
        prompt: str,
        use_json_mode: bool = True,
        temperature: float = 0.1,
        max_tokens: int = 512,
        use_fast_model: bool = False,
    ) -> str:
        """Send a prompt to Gemini and return the response text.

        Args:
            prompt: The full prompt string to send.
            use_json_mode: If True, sets ``response_mime_type`` to
                ``application/json`` so Gemini returns valid JSON natively.
            temperature: Sampling temperature (0.0-2.0).
            max_tokens: Maximum output tokens.
            use_fast_model: If True, use the fast model instead
                of the primary model.

        Returns:
            The raw response text from Gemini.

        Raises:
            APIError: After all retry attempts are exhausted.
        """
        model = self.fast_model if use_fast_model else self.primary_model
        mime_type = "application/json" if use_json_mode else "text/plain"

        gen_config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            response_mime_type=mime_type,
        )

        last_exception: Optional[Exception] = None

        for attempt in range(app_config.retry_attempts):
            try:
                self._rate_limit()
                response = self.client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=gen_config,
                )
                self.request_count += 1
                logger.debug(
                    "Gemini response (model=%s, attempt=%d): %.120s",
                    model,
                    attempt + 1,
                    response.text,
                )
                return response.text
            except (APIError, ConnectionError, TimeoutError) as exc:
                last_exception = exc
                wait = 2 ** attempt + random.uniform(0, 0.5)
                logger.warning(
                    "%s on attempt %d/%d (model=%s): %s — retrying in %.1fs",
                    type(exc).__name__,
                    attempt + 1,
                    app_config.retry_attempts,
                    model,
                    exc,
                    wait,
                )
                if attempt < app_config.retry_attempts - 1:
                    time.sleep(wait)

        # All retries exhausted
        raise LLMError(f"All {app_config.retry_attempts} retries exhausted") from last_exception

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _rate_limit(self) -> None:
        """Enforce a maximum of ~60 requests per minute (1 req/s)."""
        if self.last_request_time is not None:
            elapsed = time.time() - self.last_request_time
            if elapsed < 1.0:
                time.sleep(1.0 - elapsed)
        self.last_request_time = time.time()


# -- Standalone demo ---------------------------------------------------
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    print("=" * 50)
    print("  Module 2: llm_client.py -- Gemini Client Demo")
    print("=" * 50)

    try:
        client = GeminiClient()
        print(f"\nGeminiClient initialised")
        print(f"   Primary model : {client.primary_model}")
        print(f"   Fast model    : {client.fast_model}")
        print(f"   Request count : {client.request_count}")

        print("\nSending test query to Gemini ...")
        response = client.query(
            'Return JSON: {"status": "ok", "message": "BotTrainer LLM client working"}',
            use_json_mode=True,
            temperature=0.0,
        )
        print(f"Response: {response}")
        print(f"   Requests made: {client.request_count}")

    except ValueError as exc:
        print(f"\n{exc}")
        print("   Set GEMINI_API_KEY in your .env file.")
    except Exception as exc:
        print(f"\nUnexpected error: {exc}")

    print()
