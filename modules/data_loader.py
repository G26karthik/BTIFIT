"""
Data Loader for BotTrainer.
==============================
Downloads CLINC150 from Kaggle, preprocesses into intents.json and eval_dataset.json.
"""

import json
import logging
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from modules.utils import iso_timestamp, load_json_file, save_json_file

logger = logging.getLogger(__name__)

# Project root is one level above this file
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"

DATASET_ID: str = "hongtrung/clinc150-dataset"
RAW_FILE: str = "data_full.json"


class DataLoader:
    """Handles CLINC150 dataset acquisition and preprocessing.

    Attributes:
        data_dir: Resolved path to ``data/``.
        raw_dir: Resolved path to ``data/raw/``.
        raw_file: Path to the expected raw JSON file after download.
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        raw_dir: Optional[Path] = None,
    ) -> None:
        """Initialise the data loader with configurable directories.

        Args:
            data_dir: Override for the ``data/`` directory.
            raw_dir: Override for the ``data/raw/`` directory.
        """
        self.data_dir: Path = Path(data_dir) if data_dir else DATA_DIR
        self.raw_dir: Path = Path(raw_dir) if raw_dir else RAW_DIR
        self.raw_file: Path = self.raw_dir / RAW_FILE

        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Download                                                          #
    # ------------------------------------------------------------------ #

    def download_dataset(self, force: bool = False) -> bool:
        """Download the CLINC150 dataset from Kaggle.

        Skips the download if ``data/raw/data_full.json`` already exists
        (unless *force* is True).

        Args:
            force: Re-download even if the file exists.

        Returns:
            True if the raw file is available after this call.
        """
        if self.raw_file.exists() and not force:
            logger.info("Raw dataset already exists at %s — skipping download.", self.raw_file)
            return True

        try:
            from kaggle.api.kaggle_api_extended import KaggleApi

            api = KaggleApi()
            api.authenticate()
            logger.info("Downloading CLINC150 dataset (%s) …", DATASET_ID)
            api.dataset_download_files(
                DATASET_ID,
                path=str(self.raw_dir),
                unzip=True,
            )
            if self.raw_file.exists():
                logger.info("Download complete — %s", self.raw_file)
                return True
            else:
                # Sometimes the zip structure nests files one level deeper
                nested = list(self.raw_dir.rglob(RAW_FILE))
                if nested:
                    import shutil
                    shutil.move(str(nested[0]), str(self.raw_file))
                    logger.info("Moved nested file to %s", self.raw_file)
                    return True
                logger.error("Download succeeded but %s not found.", RAW_FILE)
                return False
        except Exception as exc:
            logger.error("Failed to download dataset: %s", exc)
            return False

    # ------------------------------------------------------------------ #
    #  Preprocessing                                                     #
    # ------------------------------------------------------------------ #

    def _load_raw(self) -> Optional[dict]:
        """Load the raw CLINC150 JSON file.

        Returns:
            Parsed dict with ``train``, ``test``, ``val`` keys, or None.
        """
        return load_json_file(self.raw_file)

    def build_intents_json(self, output_path: Optional[Path] = None) -> bool:
        """Convert the train split into ``intents.json``.

        Groups utterances by intent and writes the standardised BotTrainer
        intents format.

        Args:
            output_path: Override output location (default: ``data/intents.json``).

        Returns:
            True on success.
        """
        output_path = Path(output_path) if output_path else self.data_dir / "intents.json"

        raw = self._load_raw()
        if raw is None:
            logger.error("Cannot build intents.json — raw data not loaded.")
            return False

        train_data: list = raw.get("train", [])
        if not train_data:
            logger.error("Train split is empty or missing.")
            return False

        # Group examples by intent
        intent_map: dict[str, list[str]] = defaultdict(list)
        for utterance, intent_name in train_data:
            intent_map[intent_name].append(utterance)

        intents_list: list[dict[str, Any]] = []
        for name in sorted(intent_map.keys()):
            intents_list.append(
                {
                    "name": name,
                    "examples": intent_map[name],
                    "entities": [],
                }
            )

        payload: dict[str, Any] = {
            "intents": intents_list,
            "metadata": {
                "total_intents": len(intents_list),
                "total_examples": sum(len(i["examples"]) for i in intents_list),
                "source": "CLINC150",
                "split": "train",
                "created_at": iso_timestamp(),
            },
        }

        success = save_json_file(payload, output_path)
        if success:
            logger.info(
                "intents.json saved — %d intents, %d examples.",
                payload["metadata"]["total_intents"],
                payload["metadata"]["total_examples"],
            )
        return success

    def build_eval_dataset(
        self,
        output_path: Optional[Path] = None,
        per_intent: int = 5,
        seed: int = 42,
    ) -> bool:
        """Build a balanced evaluation dataset from the test split.

        Selects exactly *per_intent* examples for every intent present in
        the test set.

        Args:
            output_path: Override output location (default: ``data/eval_dataset.json``).
            per_intent: Number of examples to pick per intent.
            seed: Random seed for reproducibility.

        Returns:
            True on success.
        """
        output_path = Path(output_path) if output_path else self.data_dir / "eval_dataset.json"

        raw = self._load_raw()
        if raw is None:
            logger.error("Cannot build eval_dataset.json — raw data not loaded.")
            return False

        test_data: list = raw.get("test", [])
        if not test_data:
            logger.error("Test split is empty or missing.")
            return False

        # Group by intent
        intent_map: dict[str, list[str]] = defaultdict(list)
        for utterance, intent_name in test_data:
            intent_map[intent_name].append(utterance)

        rng = random.Random(seed)
        eval_samples: list[dict[str, Any]] = []
        sample_id = 1

        for intent_name in sorted(intent_map.keys()):
            examples = intent_map[intent_name][:]
            rng.shuffle(examples)
            selected = examples[:per_intent]
            for text in selected:
                eval_samples.append(
                    {
                        "id": sample_id,
                        "text": text,
                        "true_intent": intent_name,
                    }
                )
                sample_id += 1

        intents_covered = len(intent_map)

        payload: dict[str, Any] = {
            "eval_samples": eval_samples,
            "metadata": {
                "total_samples": len(eval_samples),
                "intents_covered": intents_covered,
                "samples_per_intent": per_intent,
                "source": "CLINC150 test split",
                "seed": seed,
                "created_at": iso_timestamp(),
            },
        }

        success = save_json_file(payload, output_path)
        if success:
            logger.info(
                "eval_dataset.json saved — %d samples across %d intents.",
                len(eval_samples),
                intents_covered,
            )
        return success

    # ------------------------------------------------------------------ #
    #  Convenience                                                       #
    # ------------------------------------------------------------------ #

    def run_full_pipeline(self, force_download: bool = False) -> bool:
        """Run the complete data acquisition and preprocessing pipeline.

        1. Download CLINC150 from Kaggle (if needed).
        2. Build ``intents.json`` from the train split.
        3. Build ``eval_dataset.json`` from the test split.

        Args:
            force_download: Re-download even if raw data exists.

        Returns:
            True if all steps succeeded.
        """
        ok = self.download_dataset(force=force_download)
        if not ok:
            return False
        ok = self.build_intents_json()
        if not ok:
            return False
        ok = self.build_eval_dataset()
        return ok


# Allow direct execution for standalone preprocessing
if __name__ == "__main__":
    from modules.utils import setup_logging

    setup_logging()
    loader = DataLoader()
    success = loader.run_full_pipeline()
    if success:
        print("✅ Data pipeline complete.")
    else:
        print("❌ Data pipeline failed — check logs above.")
