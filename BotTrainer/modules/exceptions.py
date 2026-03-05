"""
BotTrainer — Custom Exceptions.
=================================
Structured error types for clear error handling across the pipeline.
"""


class BotTrainerError(Exception):
    """Base exception for all BotTrainer errors."""


class LLMError(BotTrainerError):
    """Raised when the LLM API call fails after retries."""


class ParseError(BotTrainerError):
    """Raised when LLM response cannot be parsed into valid JSON."""
