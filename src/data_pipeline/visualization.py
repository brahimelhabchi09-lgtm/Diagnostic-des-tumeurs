"""Visualisations Seaborn/Matplotlib du dataset."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image

from .config import FIGURES_DIR, SAMPLES_PER_CLASS

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", context="talk")


def plot_class_countplot(
    catalog: pd.DataFrame,
    output_dir: Path | None = None,
    filename: str = "countplot_images_par_classe.png",
) -> Path:
    """Crée un countplot du nombre d'images par classe et l'enregistre."""
    figures_dir = Path(output_dir) if output_dir is not None else FIGURES_DIR
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / filename

    if catalog.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title("Nombre d'images par classe (aucune image valide)")
        ax.set_xlabel("Classe")
        ax.set_ylabel("Nombre d'images")
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        logger.warning("Countplot vide enregistré dans %s", output_path)
        return output_path

    order = sorted(catalog["label"].unique())
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(data=catalog, x="label", hue="label", order=order, palette="Set2", legend=False, ax=ax)
    ax.set_title("Nombre d'images par classe")
    ax.set_xlabel("Classe")
    ax.set_ylabel("Nombre d'images")
    ax.tick_params(axis="x", rotation=15)

    for patch in ax.patches:
        height = patch.get_height()
        ax.annotate(
            f"{int(height)}",
            (patch.get_x() + patch.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=11,
            xytext=(0, 4),
            textcoords="offset points",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Countplot enregistré : %s", output_path)
    return output_path


def plot_class_samples(
    catalog: pd.DataFrame,
    output_dir: Path | None = None,
    samples_per_class: int = SAMPLES_PER_CLASS,
    filename: str = "echantillons_par_classe.png",
) -> Path:
    """Affiche une grille d'images représentatives pour chaque classe."""
    figures_dir = Path(output_dir) if output_dir is not None else FIGURES_DIR
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / filename

    classes = sorted(catalog["label"].unique()) if not catalog.empty else []
    n_classes = max(len(classes), 1)
    n_cols = max(samples_per_class, 1)

    fig, axes = plt.subplots(n_classes, n_cols, figsize=(3.2 * n_cols, 3.2 * n_classes))
    axes_array = _normalize_axes(axes, n_classes, n_cols)

    if not classes:
        axes_array[0][0].set_title("Aucune image à afficher")
        axes_array[0][0].axis("off")
    else:
        for row, label in enumerate(classes):
            class_paths = catalog.loc[catalog["label"] == label, "filepath"].tolist()
            selected = class_paths[:n_cols]
            for col in range(n_cols):
                ax = axes_array[row][col]
                ax.axis("off")
                if col >= len(selected):
                    continue
                try:
                    with Image.open(selected[col]) as image:
                        ax.imshow(image.convert("RGB"))
                    ax.set_title(label, fontsize=11)
                except Exception as exc:  # noqa: BLE001
                    ax.set_title(f"{label}\n(erreur)", fontsize=10)
                    logger.warning("Impossible d'afficher %s : %s", selected[col], exc)

    fig.suptitle("Échantillon d'images par classe", fontsize=16, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Grille d'échantillons enregistrée : %s", output_path)
    return output_path


def _normalize_axes(axes, n_rows: int, n_cols: int) -> list[list]:
    if n_rows == 1 and n_cols == 1:
        return [[axes]]
    if n_rows == 1:
        return [list(axes)]
    if n_cols == 1:
        return [[ax] for ax in axes]
    return [list(row) for row in axes]
