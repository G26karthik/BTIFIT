"""
BotTrainer — Module Runner.
================================
Run individual modules or all modules in sequence.

Usage (from the project root — AIIP/):
    python -m BotTrainer                    # Run all modules in sequence
    python -m BotTrainer utils              # Module 1 — Utility functions
    python -m BotTrainer llm_client         # Module 2 — Gemini LLM client
    python -m BotTrainer data_loader        # Module 3 — Dataset download
    python -m BotTrainer intent_classifier  # Module 4 — Intent classifier
    python -m BotTrainer entity_extractor   # Module 5 — Entity extractor
    python -m BotTrainer evaluator          # Module 6 — Evaluator
    python -m BotTrainer pipeline           # Full interactive CLI pipeline
"""

import importlib
import sys
from pathlib import Path

# Module run order (name → dotted import path)
MODULES = [
    ("utils",              "BotTrainer.modules.utils"),
    ("llm_client",         "BotTrainer.modules.llm_client"),
    ("data_loader",        "BotTrainer.modules.data_loader"),
    ("intent_classifier",  "BotTrainer.modules.intent_classifier"),
    ("entity_extractor",   "BotTrainer.modules.entity_extractor"),
    ("evaluator",          "BotTrainer.modules.evaluator"),
]

VALID_NAMES = {name for name, _ in MODULES} | {"pipeline"}


def _run_module(dotted_path: str) -> None:
    """Import and execute a module's __main__ block via runpy."""
    import runpy
    runpy.run_module(dotted_path, run_name="__main__", alter_sys=True)


def _run_pipeline() -> None:
    """Run the interactive CLI pipeline."""
    from BotTrainer.pipeline import main
    main()


def _run_all() -> None:
    """Run every module demo in sequence."""
    print("╔" + "═" * 58 + "╗")
    print("║   BotTrainer — Running All Modules                       ║")
    print("╚" + "═" * 58 + "╝")
    print()

    for i, (name, dotted_path) in enumerate(MODULES, 1):
        print(f"{'─' * 60}")
        print(f"  [{i}/{len(MODULES)}]  {name}")
        print(f"{'─' * 60}")
        try:
            _run_module(dotted_path)
        except SystemExit:
            pass
        except Exception as exc:
            print(f"  ⚠️  {name} failed: {exc}")
        print()

    print("═" * 60)
    print("  ✅ All modules executed.")
    print("═" * 60)


def main() -> None:
    args = sys.argv[1:]

    if not args:
        _run_all()
        return

    target = args[0].lower()

    if target in ("--help", "-h", "help"):
        print(__doc__)
        return

    target = target.replace("-", "_")

    if target == "pipeline":
        _run_pipeline()
        return

    if target not in VALID_NAMES:
        print(f"❌ Unknown module: '{args[0]}'")
        print(f"   Valid modules: {', '.join(sorted(VALID_NAMES))}")
        sys.exit(1)

    dotted_path = dict(MODULES)[target]
    _run_module(dotted_path)


if __name__ == "__main__":
    main()
