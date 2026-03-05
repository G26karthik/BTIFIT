"""
BotTrainer — One-Command Project Setup.
==========================================
Run with: python setup.py
Creates a virtual environment, installs dependencies, configures Kaggle,
downloads CLINC150, and generates all data files.
"""

import json
import os
import platform
import shutil
import subprocess
import sys
import venv
from pathlib import Path

# ── Project paths ──────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent
VENV_DIR: Path = PROJECT_ROOT / ".venv"
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROMPTS_DIR: Path = PROJECT_ROOT / "prompts"
MODULES_DIR: Path = PROJECT_ROOT / "modules"
REQ_FILE: Path = PROJECT_ROOT / "requirements.txt"
ENV_FILE: Path = PROJECT_ROOT / ".env"
ENV_EXAMPLE: Path = PROJECT_ROOT / ".env.example"
INTENTS_FILE: Path = DATA_DIR / "intents.json"
EVAL_FILE: Path = DATA_DIR / "eval_dataset.json"

# ── Known Kaggle credential locations (checked in order) ──────────────
KAGGLE_SEARCH_PATHS: list[Path] = [
    Path.home() / ".kaggle" / "kaggle.json",
    Path(r"C:\Users\saita\Downloads\kaggle.json"),
    Path.home() / "Downloads" / "kaggle.json",
    Path.home() / "Desktop" / "kaggle.json",
]
KAGGLE_DEST: Path = Path.home() / ".kaggle" / "kaggle.json"


def _header(text: str) -> None:
    """Print a section header."""
    width = 56
    print(f"\n{'═' * width}")
    print(f"  {text}")
    print(f"{'═' * width}")


def _check(label: str, ok: bool) -> None:
    """Print a status line."""
    icon = "✅" if ok else "❌"
    print(f"  {icon}  {label}")


def _get_venv_python() -> Path:
    """Return the path to the Python executable inside the venv."""
    if sys.platform == "win32":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def _get_venv_pip() -> Path:
    """Return the path to pip inside the venv."""
    if sys.platform == "win32":
        return VENV_DIR / "Scripts" / "pip.exe"
    return VENV_DIR / "bin" / "pip"


# ── Step 1: Python version check ──────────────────────────────────────
def check_python() -> bool:
    """Verify Python >= 3.10."""
    major, minor = sys.version_info[:2]
    ok = (major, minor) >= (3, 10)
    _check(f"Python {major}.{minor}.{sys.version_info.micro}", ok)
    if not ok:
        print("      ⛔ Python 3.10+ is required. Please upgrade.")
    return ok


# ── Step 2: Create virtual environment ────────────────────────────────
def create_virtualenv() -> bool:
    """Create a .venv virtual environment if it doesn't already exist."""
    _header("Setting up virtual environment …")
    venv_python = _get_venv_python()

    if venv_python.exists():
        _check(f"Virtual environment already exists at .venv", True)
        return True

    try:
        print("  ⏳ Creating .venv (this takes a moment) …")
        venv.create(str(VENV_DIR), with_pip=True, clear=False)
        if venv_python.exists():
            _check(f"Virtual environment created at .venv", True)
            return True
        else:
            _check("venv created but python not found inside", False)
            return False
    except Exception as exc:
        _check(f"Failed to create venv: {exc}", False)
        return False


# ── Step 3: Install dependencies inside venv ──────────────────────────
def install_dependencies() -> bool:
    """Run pip install -r requirements.txt inside the virtual environment."""
    _header("Installing dependencies in .venv …")
    venv_pip = _get_venv_pip()
    venv_python = _get_venv_python()

    # Use python -m pip for robustness
    pip_cmd = str(venv_pip) if venv_pip.exists() else f"{venv_python} -m pip"

    try:
        # Upgrade pip first (silently)
        subprocess.run(
            [str(venv_python), "-m", "pip", "install", "--upgrade", "pip"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )

        # Install requirements with visible output so user sees progress
        print("  ⏳ Installing packages (output shown below) …\n")
        result = subprocess.run(
            [str(venv_python), "-m", "pip", "install", "-r", str(REQ_FILE)],
            check=True,
        )
        print()
        _check("Dependencies installed", True)
        return True
    except subprocess.CalledProcessError as exc:
        print()
        _check(f"pip install failed (exit code {exc.returncode})", False)
        print("       Try running manually:")
        print(f"       {venv_python} -m pip install -r requirements.txt")
        return False


# ── Step 4: Create directories ────────────────────────────────────────
def create_directories() -> None:
    """Ensure all required directories exist."""
    for d in (DATA_DIR, RAW_DIR, PROMPTS_DIR, MODULES_DIR):
        d.mkdir(parents=True, exist_ok=True)
    _check("Directories created", True)


# ── Step 5: Check / copy Kaggle credentials ───────────────────────────
def check_kaggle() -> bool:
    """Find kaggle.json and copy it to ~/.kaggle/ if needed."""
    # Already in the standard location?
    if KAGGLE_DEST.exists():
        _check(f"Kaggle credentials found at {KAGGLE_DEST}", True)
        return True

    # Search known locations
    for src in KAGGLE_SEARCH_PATHS:
        if src.exists() and src != KAGGLE_DEST:
            try:
                KAGGLE_DEST.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(src), str(KAGGLE_DEST))
                # Kaggle warns if permissions are too open (Linux/macOS)
                if sys.platform != "win32":
                    os.chmod(str(KAGGLE_DEST), 0o600)
                _check(f"Kaggle credentials copied from {src} → {KAGGLE_DEST}", True)
                return True
            except Exception as exc:
                _check(f"Failed to copy kaggle.json: {exc}", False)
                return False

    print("  ⚠️   Kaggle credentials not found in any of:")
    for p in KAGGLE_SEARCH_PATHS:
        print(f"        • {p}")
    print("       Download from https://www.kaggle.com/settings → API → Create New Token")
    return False


# ── Step 6: Check / create .env ───────────────────────────────────────
def check_env() -> None:
    """Copy .env.example → .env if .env doesn't exist."""
    if ENV_FILE.exists():
        _check(".env file exists", True)
    elif ENV_EXAMPLE.exists():
        shutil.copy2(str(ENV_EXAMPLE), str(ENV_FILE))
        _check(".env created from .env.example", True)
        print("       ⚠️  Remember to add your GEMINI_API_KEY to .env")
    else:
        _check(".env.example not found — creating .env stub", False)
        ENV_FILE.write_text(
            "# BotTrainer Environment\nGEMINI_API_KEY=your_key_here\n",
            encoding="utf-8",
        )


# ── Step 7: Download & preprocess dataset (runs inside venv) ─────────
def run_data_pipeline() -> bool:
    """Download CLINC150 and generate intents.json + eval_dataset.json.

    Runs data_loader.py as a subprocess using the venv Python so that all
    installed packages are available.
    """
    _header("Running data pipeline …")
    venv_python = _get_venv_python()

    # Build a small inline script that imports and runs the loader
    script = (
        "import sys, pathlib; "
        f"sys.path.insert(0, {str(PROJECT_ROOT.parent)!r}); "
        "from BotTrainer.modules.data_loader import DataLoader; "
        "from BotTrainer.modules.utils import setup_logging; "
        "setup_logging(); "
        "loader = DataLoader(); "
        "ok = loader.run_full_pipeline(); "
        "sys.exit(0 if ok else 1)"
    )

    try:
        result = subprocess.run(
            [str(venv_python), "-c", script],
            check=True,
        )
    except subprocess.CalledProcessError:
        _check("Data pipeline failed — see errors above", False)
        return False

    # Verify output files
    ok = True
    try:
        with INTENTS_FILE.open("r", encoding="utf-8") as f:
            intents_data = json.load(f)
        n_intents = intents_data.get("metadata", {}).get("total_intents", "?")
        n_examples = intents_data.get("metadata", {}).get("total_examples", "?")
        _check(f"intents.json created ({n_intents} intents, {n_examples} examples)", True)
    except Exception:
        _check("intents.json", INTENTS_FILE.exists())
        ok = INTENTS_FILE.exists()

    try:
        with EVAL_FILE.open("r", encoding="utf-8") as f:
            eval_data = json.load(f)
        n_samples = eval_data.get("metadata", {}).get("total_samples", "?")
        _check(f"eval_dataset.json created ({n_samples} samples)", True)
    except Exception:
        _check("eval_dataset.json", EVAL_FILE.exists())
        ok = ok and EVAL_FILE.exists()

    return ok


# ── Step 8: Print summary ─────────────────────────────────────────────
def print_summary(all_ok: bool) -> None:
    """Print the final setup checklist."""
    _header("Setup Summary")

    venv_python = _get_venv_python()
    _check(f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}", True)
    _check(f"Virtual environment at .venv", venv_python.exists())

    # Check key packages by importing inside venv
    for pkg_display, pkg_import in [
        ("google-genai", "google.genai"),
        ("streamlit", "streamlit"),
        ("scikit-learn", "sklearn"),
        ("pandas", "pandas"),
        ("plotly", "plotly"),
        ("kaggle", "kaggle"),
    ]:
        try:
            r = subprocess.run(
                [str(venv_python), "-c", f"import {pkg_import}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            _check(f"{pkg_display} installed", r.returncode == 0)
        except Exception:
            _check(f"{pkg_display} installed", False)

    _check("Kaggle credentials", KAGGLE_DEST.exists())
    _check("intents.json exists", INTENTS_FILE.exists())
    _check("eval_dataset.json exists", EVAL_FILE.exists())

    # .env check
    if ENV_FILE.exists():
        try:
            env_text = ENV_FILE.read_text(encoding="utf-8")
            has_key = False
            for line in env_text.splitlines():
                if line.strip().startswith("GEMINI_API_KEY="):
                    val = line.split("=", 1)[1].strip()
                    if val and val != "your_gemini_api_key_here":
                        has_key = True
            if has_key:
                _check("GEMINI_API_KEY configured", True)
            else:
                print("  ⚠️   Add your GEMINI_API_KEY to .env before running the app")
        except Exception:
            print("  ⚠️   Could not read .env")
    else:
        print("  ⚠️   .env not found")

    # Activation instructions
    print()
    if all_ok:
        print("  🎉 Setup complete! Next steps:")
        print()
        if sys.platform == "win32":
            print("     # Activate the virtual environment:")
            print("     .venv\\Scripts\\activate")
        else:
            print("     # Activate the virtual environment:")
            print("     source .venv/bin/activate")
        print()
        print("     # Install as editable package (from project root):")
        print("     pip install -e ..")
        print()
        print("     # Add your Gemini API key:")
        print("     # Edit .env → set GEMINI_API_KEY=your_key")
        print()
        print("     # Launch the app:")
        print("     cd .. && python run.py --app")
        print()
    else:
        print("  ⚠️  Some steps had issues — see above for details.")
        print()


# ── Main ──────────────────────────────────────────────────────────────
def main() -> None:
    """Run the full setup pipeline."""
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║        🤖  BotTrainer — Project Setup  🤖            ║")
    print("╚══════════════════════════════════════════════════════╝")

    # Step 1 — Python check
    _header("Checking Python version …")
    if not check_python():
        sys.exit(1)

    # Step 2 — Virtual environment
    venv_ok = create_virtualenv()
    if not venv_ok:
        print("\n  ⛔ Cannot continue without a virtual environment.")
        sys.exit(1)

    # Step 3 — Dependencies
    dep_ok = install_dependencies()

    # Step 4 — Directories
    _header("Creating directories …")
    create_directories()

    # Step 5 — Kaggle credentials
    _header("Checking Kaggle credentials …")
    kaggle_ok = check_kaggle()

    # Step 6 — .env
    _header("Checking environment file …")
    check_env()

    # Step 7 — Data pipeline
    data_ok = False
    if kaggle_ok and dep_ok:
        data_ok = run_data_pipeline()
    else:
        if not kaggle_ok:
            print("\n  ⏭️  Skipping data download (no Kaggle credentials).")
            print("     Place kaggle.json in Downloads or ~/.kaggle/ and re-run.")
        if not dep_ok:
            print("\n  ⏭️  Skipping data download (dependencies not installed).")

    # Step 8 — Summary
    print_summary(all_ok=dep_ok and data_ok)


if __name__ == "__main__":
    main()
