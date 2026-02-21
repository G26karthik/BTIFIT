"""
Gemini LLM Client for BotTrainer.
====================================
Wraps the google-genai SDK for intent classification and entity extraction.
Uses gemini-2.5-flash (primary) and gemini-2.5-flash-lite (fast fallback).
"""

import logging
import os
import time
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError

logger = logging.getLogger(__name__)


class GeminiClient:
    """Thin wrapper around the Google GenAI SDK for BotTrainer.

    Attributes:
        client: The underlying ``genai.Client`` instance.
        primary_model: Default model identifier (gemini-2.5-flash).
        fast_model: Lighter model for high-volume batches (gemini-2.5-flash-lite).
        request_count: Running count of API calls made in this session.
        last_request_time: Epoch time of the most recent request (for rate limiting).
    """

    def __init__(self) -> None:
        """Initialise the Gemini client from environment variables.

        Raises:
            ValueError: If ``GEMINI_API_KEY`` is not set.
        """
        load_dotenv()
        api_key: Optional[str] = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found in environment. "
                "Add it to your .env file (see .env.example)."
            )

        self.client: genai.Client = genai.Client(api_key=api_key)
        self.primary_model: str = "gemini-2.5-flash"
        self.fast_model: str = "gemini-2.5-flash-lite"
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
            temperature: Sampling temperature (0.0–2.0).
            max_tokens: Maximum output tokens.
            use_fast_model: If True, use ``gemini-2.5-flash-lite`` instead
                of the primary model.

        Returns:
            The raw response text from Gemini.

        Raises:
            APIError: After 3 failed retry attempts.
        """
        model = self.fast_model if use_fast_model else self.primary_model
        mime_type = "application/json" if use_json_mode else "text/plain"

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            response_mime_type=mime_type,
        )

        last_exception: Optional[Exception] = None

        for attempt in range(3):
            try:
                self._rate_limit()
                response = self.client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=config,
                )
                self.request_count += 1
                logger.debug(
                    "Gemini response (model=%s, attempt=%d): %.120s",
                    model,
                    attempt + 1,
                    response.text,
                )
                return response.text
            except APIError as exc:
                last_exception = exc
                wait = 2 ** attempt
                logger.warning(
                    "APIError on attempt %d/%d (model=%s): %s — retrying in %ds",
                    attempt + 1,
                    3,
                    model,
                    exc,
                    wait,
                )
                if attempt < 2:
                    time.sleep(wait)

        # All retries exhausted
        raise last_exception  # type: ignore[misc]

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
