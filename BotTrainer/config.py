"""
BotTrainer — Centralized Configuration.
==========================================
Single source of truth for all tuneable parameters.
Loads from environment variables with sensible defaults.
"""

from dataclasses import dataclass, field
from os import getenv

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    """Application-wide settings loaded once at import time."""

    gemini_api_key: str = field(default_factory=lambda: getenv("GEMINI_API_KEY", ""))
    primary_model: str = "gemini-2.5-flash"
    fast_model: str = "gemini-2.5-flash-lite"
    cache_max_size: int = 500
    tfidf_max_features: int = 15000
    tfidf_ngram_range: tuple[int, int] = (1, 3)
    n_few_shot: int = 5
    max_input_length: int = 1000
    retry_attempts: int = 3


config = Config()
