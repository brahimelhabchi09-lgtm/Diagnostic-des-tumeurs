"""Exploration des classes et construction du catalogue Pandas."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from .loading import DatasetLoadResult, discover_class_directories

logger = logging.getLogger(__name__)

PATIENT_PATTERNS = (
    re.compile(r"(?:patient|pat|subj|subject|pid)[-_]?([A-Za-z0-9]+)", re.IGNORECASE),
    re.compile(r"(?:^|[-_])(P\d{3,})(?:[-_.]|$)", re.IGNORECASE),
)


def detect_classes(dataset_dir: Path) -> list[str]:
    """Détecte automatiquement les classes à partir des sous-dossiers."""
    return [directory.name for directory in discover_class_directories(dataset_dir)]


def extract_patient_id(filepath: str | Path) -> str:
    """Extrait un identifiant patient du nom de fichier s'il est présent.

    Si aucun motif patient n'est reconnu, chaque image constitue son propre
    groupe (pas de fuite inter-patients possible).
    """
    stem = Path(filepath).stem
    for pattern in PATIENT_PATTERNS:
        match = pattern.search(stem)
        if match:
            return match.group(1).lower()
    return f"image::{Path(filepath).resolve()}"


def build_catalog(load_result: DatasetLoadResult) -> pd.DataFrame:
    """Construit un DataFrame contenant les chemins, labels et identifiants patient."""
    if not load_result.valid_records:
        columns = ["filepath", "label", "patient_id"]
        return pd.DataFrame(columns=columns)

    catalog = pd.DataFrame(load_result.valid_records)
    catalog["filepath"] = catalog["filepath"].astype(str)
    catalog["label"] = catalog["label"].astype(str)
    catalog["patient_id"] = catalog["filepath"].map(extract_patient_id)
    catalog = catalog.sort_values(["label", "filepath"], kind="mergesort").reset_index(drop=True)
    return catalog


def class_statistics(catalog: pd.DataFrame) -> pd.DataFrame:
    """Calcule le nombre d'images et le pourcentage par classe."""
    if catalog.empty:
        return pd.DataFrame(columns=["label", "n_images", "percentage", "n_patients"])

    counts = catalog.groupby("label", sort=True).agg(
        n_images=("filepath", "size"),
        n_patients=("patient_id", "nunique"),
    )
    total = counts["n_images"].sum()
    counts["percentage"] = (counts["n_images"] / total * 100).round(2)
    stats = counts.reset_index()
    logger.info("Statistiques par classe :\n%s", stats.to_string(index=False))
    return stats
