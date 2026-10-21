"""Chemins et constantes du projet."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = Path(
    os.environ.get("DATASET_DIR", PROJECT_ROOT / "Data")
).expanduser().resolve()

REPORTS_DIR = Path(
    os.environ.get("REPORTS_DIR", PROJECT_ROOT / "reports")
).expanduser().resolve()

FIGURES_DIR = REPORTS_DIR / "figures"
SPLITS_DIR = REPORTS_DIR / "splits"

ALLOWED_EXTENSIONS = {".jpeg", ".jpg", ".bmp", ".png"}

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_STATE = 42

SAMPLES_PER_CLASS = 4
