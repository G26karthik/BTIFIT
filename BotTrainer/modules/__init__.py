"""
BotTrainer Modules Package
===========================
LLM-Based NLU Model Trainer & Evaluator for Chatbots.

All public classes are importable from this package::

    from BotTrainer.modules import GeminiClient, IntentClassifier, ...

Imports are lazy — individual modules can be used without pulling in
every dependency.
"""

__all__ = [
    "GeminiClient",
    "DataLoader",
    "IntentClassifier",
    "EntityExtractor",
    "Evaluator",
]


def __getattr__(name: str):
    """Lazy-import public classes on first access."""
    if name == "GeminiClient":
        from BotTrainer.modules.llm_client import GeminiClient
        return GeminiClient
    if name == "DataLoader":
        from BotTrainer.modules.data_loader import DataLoader
        return DataLoader
    if name == "IntentClassifier":
        from BotTrainer.modules.intent_classifier import IntentClassifier
        return IntentClassifier
    if name == "EntityExtractor":
        from BotTrainer.modules.entity_extractor import EntityExtractor
        return EntityExtractor
    if name == "Evaluator":
        from BotTrainer.modules.evaluator import Evaluator
        return Evaluator
    raise AttributeError(f"module 'BotTrainer.modules' has no attribute {name!r}")
