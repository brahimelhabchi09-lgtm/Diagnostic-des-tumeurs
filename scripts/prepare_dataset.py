"""Préparation du dataset : chargement, exploration, visualisation, split."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: F401  — visualisations
import numpy as np  # noqa: F401
import pandas as pd
import seaborn as sns  # noqa: F401
from PIL import Image  # noqa: F401
from sklearn.model_selection import train_test_split  # noqa: F401

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_pipeline.config import (  # noqa: E402
    DATASET_DIR,
    FIGURES_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    SPLITS_DIR,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)
from data_pipeline.exploration import build_catalog, class_statistics  # noqa: E402
from data_pipeline.loading import load_dataset_images  # noqa: E402
from data_pipeline.splitting import split_dataset, split_summary  # noqa: E402
from data_pipeline.visualization import plot_class_countplot, plot_class_samples  # noqa: E402

logger = logging.getLogger("prepare_dataset")


def _configure_logging() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(REPORTS_DIR / "prepare_dataset.log", encoding="utf-8"),
        ],
    )


def _import_tensorflow() -> str:
    """Importe TensorFlow/Keras et retourne un message de statut."""
    try:
        import tensorflow as tf
        from tensorflow import keras  # noqa: F401

        return (
            f"TensorFlow {tf.__version__} / Keras {keras.__version__} importés avec succès "
            f"(exécution {tf.executing_eagerly()})."
        )
    except Exception as exc:  # noqa: BLE001
        return (
            "TensorFlow/Keras n'ont pas pu être importés "
            f"({exc}). Le pipeline de données a tout de même été exécuté."
        )


def _save_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8")
    logger.info("Fichier enregistré : %s (%s lignes)", path, len(frame))


def run() -> int:
    _configure_logging()
    tf_status = _import_tensorflow()
    logger.info("%s", tf_status)

    logger.info("Répertoire du dataset : %s", DATASET_DIR)
    if not DATASET_DIR.exists():
        logger.error(
            "Dataset introuvable. Placez les images dans %s "
            "(un sous-dossier par classe) ou définissez DATASET_DIR.",
            DATASET_DIR,
        )

    load_result = load_dataset_images(DATASET_DIR)
    print(load_result.summary())

    invalid_df = pd.DataFrame(load_result.invalid_records)
    skipped_df = pd.DataFrame(load_result.skipped_records)
    _save_frame(invalid_df, REPORTS_DIR / "invalid_images.csv")
    _save_frame(skipped_df, REPORTS_DIR / "skipped_files.csv")

    catalog = build_catalog(load_result)
    _save_frame(catalog, REPORTS_DIR / "dataset_catalog.csv")

    stats = class_statistics(catalog)
    _save_frame(stats, REPORTS_DIR / "class_statistics.csv")
    if not stats.empty:
        print("\nStatistiques par classe :")
        print(stats.to_string(index=False))

    countplot_path = plot_class_countplot(catalog, FIGURES_DIR)
    samples_path = plot_class_samples(catalog, FIGURES_DIR)
    print(f"\nFigures enregistrées :\n  - {countplot_path}\n  - {samples_path}")

    split_result = split_dataset(
        catalog,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        random_state=RANDOM_STATE,
    )
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    _save_frame(split_result.train, SPLITS_DIR / "train.csv")
    _save_frame(split_result.validation, SPLITS_DIR / "validation.csv")
    _save_frame(split_result.test, SPLITS_DIR / "test.csv")

    summary = split_summary(split_result)
    _save_frame(summary, REPORTS_DIR / "split_summary.csv")
    print("\nRépartition train / validation / test :")
    print(summary.to_string(index=False))
    print(
        "\nStratification : "
        f"{'oui' if split_result.stratified else 'non'} | "
        "Regroupement patient : "
        f"{'oui' if split_result.grouped_by_patient else 'non'}"
    )

    if split_result.warnings:
        print("\nAvertissements :")
        for warning in split_result.warnings:
            print(f"  - {warning}")

    if load_result.n_valid == 0:
        logger.error("Aucune image valide n'a été chargée.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(run())
