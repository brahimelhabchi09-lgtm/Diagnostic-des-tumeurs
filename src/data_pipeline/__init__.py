"""Pipeline de préparation du dataset d'IRM cérébrales."""

from .config import ALLOWED_EXTENSIONS, DATASET_DIR, FIGURES_DIR, PROJECT_ROOT, SPLITS_DIR
from .exploration import build_catalog, class_statistics, detect_classes
from .loading import DatasetLoadResult, load_dataset_images
from .splitting import SplitResult, split_dataset
from .visualization import plot_class_countplot, plot_class_samples

__all__ = [
    "ALLOWED_EXTENSIONS",
    "DATASET_DIR",
    "FIGURES_DIR",
    "PROJECT_ROOT",
    "SPLITS_DIR",
    "DatasetLoadResult",
    "SplitResult",
    "build_catalog",
    "class_statistics",
    "detect_classes",
    "load_dataset_images",
    "plot_class_countplot",
    "plot_class_samples",
    "split_dataset",
]
