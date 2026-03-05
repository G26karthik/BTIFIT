#!/usr/bin/env python
"""
BotTrainer — Standard Entry Point.
====================================
Run from the project root (AIIP/) to launch the interactive CLI pipeline.

Usage:
    python run.py              # Interactive CLI
    python run.py --app        # Launch Streamlit web app
    python run.py --setup      # Run project bootstrap/setup
"""

import sys
from pathlib import Path


def main() -> None:
    """Dispatch to the appropriate BotTrainer entry point."""
    if "--app" in sys.argv:
        import subprocess

        app_path = Path(__file__).resolve().parent / "BotTrainer" / "app.py"
        sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(app_path)]))

    elif "--setup" in sys.argv:
        from BotTrainer.bootstrap import main as setup_main

        setup_main()

    else:
        from BotTrainer.pipeline import main as pipeline_main

        pipeline_main()


if __name__ == "__main__":
    main()
