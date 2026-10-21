"""Chargement et validation des images du dataset."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from .config import ALLOWED_EXTENSIONS, DATASET_DIR

logger = logging.getLogger(__name__)


@dataclass
class DatasetLoadResult:
    """Résultat du parcours et de la validation des images."""

    valid_records: list[dict] = field(default_factory=list)
    invalid_records: list[dict] = field(default_factory=list)
    skipped_records: list[dict] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    dataset_dir: Path = DATASET_DIR

    @property
    def n_valid(self) -> int:
        return len(self.valid_records)

    @property
    def n_invalid(self) -> int:
        return len(self.invalid_records)

    @property
    def n_skipped(self) -> int:
        return len(self.skipped_records)

    def summary(self) -> str:
        lines = [
            "Résumé du chargement des images",
            f"  Répertoire          : {self.dataset_dir}",
            f"  Classes détectées   : {len(self.classes)} ({', '.join(self.classes) or 'aucune'})",
            f"  Images valides      : {self.n_valid}",
            f"  Images invalides    : {self.n_invalid}",
            f"  Fichiers ignorés    : {self.n_skipped} (extension non autorisée)",
        ]
        return "\n".join(lines)


def is_allowed_image(path: Path) -> bool:
    """Retourne True si l'extension est .jpeg/.jpg/.bmp/.png (insensible à la casse)."""
    return path.suffix.lower() in ALLOWED_EXTENSIONS


def validate_image(path: Path) -> tuple[bool, str | None]:
    """Vérifie qu'une image peut être ouverte et décodée.

    `Image.verify()` contrôle l'intégrité puis l'image est rouverte et chargée
    en mémoire, car `verify()` laisse le fichier dans un état inutilisable.
    """
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
            image.convert("RGB")
        return True, None
    except FileNotFoundError as exc:
        return False, f"Fichier introuvable : {exc}"
    except UnidentifiedImageError as exc:
        return False, f"Image non identifiée : {exc}"
    except (OSError, ValueError, SyntaxError) as exc:
        return False, f"Image corrompue ou illisible : {exc}"
    except Exception as exc:  # noqa: BLE001 — journaliser toute erreur de lecture
        return False, f"Erreur inattendue lors du chargement : {exc}"


def discover_class_directories(dataset_dir: Path) -> list[Path]:
    """Retourne les sous-dossiers de premier niveau (noms de classes)."""
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Le répertoire du dataset est introuvable : {dataset_dir}. "
            "Définissez la variable d'environnement DATASET_DIR ou placez les "
            "images dans le dossier Data/ à la racine du projet "
            "(un sous-dossier par classe)."
        )
    if not dataset_dir.is_dir():
        raise NotADirectoryError(f"Le chemin du dataset n'est pas un dossier : {dataset_dir}")

    class_dirs = sorted(
        [path for path in dataset_dir.iterdir() if path.is_dir() and not path.name.startswith(".")],
        key=lambda path: path.name.lower(),
    )
    return class_dirs


def load_dataset_images(dataset_dir: Path | None = None) -> DatasetLoadResult:
    """Parcourt le dataset, valide les images et ignore les fichiers non autorisés.

    Les noms de sous-dossiers sont utilisés comme labels de classes.
    """
    root = Path(dataset_dir) if dataset_dir is not None else DATASET_DIR
    result = DatasetLoadResult(dataset_dir=root)

    try:
        class_dirs = discover_class_directories(root)
    except (FileNotFoundError, NotADirectoryError) as exc:
        logger.error("%s", exc)
        result.invalid_records.append(
            {
                "filepath": str(root),
                "label": None,
                "reason": str(exc),
                "status": "dataset_missing",
            }
        )
        return result

    result.classes = [directory.name for directory in class_dirs]
    if not result.classes:
        logger.warning("Aucun sous-dossier de classe trouvé dans %s", root)
        return result

    for class_dir in class_dirs:
        label = class_dir.name
        for path in sorted(class_dir.rglob("*")):
            if not path.is_file():
                continue
            if not is_allowed_image(path):
                result.skipped_records.append(
                    {
                        "filepath": str(path),
                        "label": label,
                        "reason": f"Extension non autorisée : {path.suffix or '(aucune)'}",
                        "status": "skipped_extension",
                    }
                )
                continue

            is_valid, error = validate_image(path)
            if is_valid:
                result.valid_records.append(
                    {
                        "filepath": str(path.resolve()),
                        "label": label,
                    }
                )
            else:
                result.invalid_records.append(
                    {
                        "filepath": str(path.resolve()),
                        "label": label,
                        "reason": error,
                        "status": "invalid",
                    }
                )
                logger.warning("Image ignorée (%s) : %s", error, path)

    logger.info("%s", result.summary())
    return result
