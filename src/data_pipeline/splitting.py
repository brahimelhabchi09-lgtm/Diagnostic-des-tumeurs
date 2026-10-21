"""Séparation stratifiée train / validation / test sans fuite de données."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import RANDOM_STATE, TEST_RATIO, TRAIN_RATIO, VAL_RATIO

logger = logging.getLogger(__name__)

TARGET_RATIOS = {"train": TRAIN_RATIO, "validation": VAL_RATIO, "test": TEST_RATIO}


@dataclass
class SplitResult:
    """Ensembles produits et diagnostics de la séparation."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    grouped_by_patient: bool = False
    stratified: bool = False

    def as_dict(self) -> dict[str, pd.DataFrame]:
        return {"train": self.train, "validation": self.validation, "test": self.test}


def split_dataset(
    catalog: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    random_state: int = RANDOM_STATE,
) -> SplitResult:
    """Divise le catalogue en train/val/test (70/15/15 par défaut).

    - Stratification par classe lorsque c'est possible.
    - Si plusieurs images partagent un identifiant patient, toutes restent
      dans le même ensemble (aucune fuite inter-patients).
    """
    _validate_ratios(train_ratio, val_ratio, test_ratio)

    empty = catalog.iloc[0:0].copy() if not catalog.empty else pd.DataFrame(
        columns=["filepath", "label", "patient_id"]
    )
    if catalog.empty:
        warning = "Catalogue vide : impossible de créer les ensembles train/validation/test."
        logger.warning(warning)
        return SplitResult(train=empty, validation=empty, test=empty, warnings=[warning])

    if len(catalog) < 3:
        warning = (
            "Répartition exacte 70/15/15 impossible : le dataset contient "
            f"seulement {len(catalog)} image(s). Toutes les images sont placées "
            "dans l'ensemble d'entraînement."
        )
        logger.warning(warning)
        return SplitResult(
            train=catalog.reset_index(drop=True),
            validation=empty,
            test=empty.copy(),
            warnings=[warning],
            grouped_by_patient=_has_multi_image_patients(catalog),
            stratified=False,
        )

    grouped = _has_multi_image_patients(catalog)
    if grouped:
        return _split_by_patient_groups(
            catalog,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_state=random_state,
        )

    logger.info(
        "Aucun identifiant patient multi-images détecté : séparation au niveau image."
    )
    return _split_images(
        catalog,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
        grouped_by_patient=False,
    )


def split_summary(result: SplitResult) -> pd.DataFrame:
    """Tableau des effectifs, proportions et distributions de classes."""
    rows: list[dict] = []
    total = sum(len(frame) for frame in result.as_dict().values())
    labels = sorted(
        set().union(*[set(frame["label"].unique()) for frame in result.as_dict().values() if "label" in frame])
    )

    for split_name, frame in result.as_dict().items():
        row: dict = {
            "split": split_name,
            "n_images": len(frame),
            "percentage": round(len(frame) / total * 100, 2) if total else 0.0,
            "target_percentage": round(TARGET_RATIOS[split_name] * 100, 2),
        }
        counts = frame["label"].value_counts() if not frame.empty else pd.Series(dtype=int)
        for label in labels:
            row[f"n_{label}"] = int(counts.get(label, 0))
        rows.append(row)
    summary = pd.DataFrame(rows)
    logger.info("Répartition finale :\n%s", summary.to_string(index=False))
    return summary


def _validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"La somme des ratios doit valoir 1, reçu {total}.")


def _has_multi_image_patients(catalog: pd.DataFrame) -> bool:
    return int(catalog["patient_id"].nunique()) < len(catalog)


def _can_stratify(labels: pd.Series, n_splits_needed: int = 2) -> tuple[bool, str | None]:
    counts = labels.value_counts()
    too_small = counts[counts < n_splits_needed]
    if too_small.empty:
        return True, None
    classes = ", ".join(f"{name} ({count})" for name, count in too_small.items())
    message = (
        "Répartition exacte 70/15/15 stratifiée impossible pour les classes "
        f"suivantes (trop peu d'images) : {classes}."
    )
    return False, message


def _split_images(
    catalog: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_state: int,
    grouped_by_patient: bool,
) -> SplitResult:
    warnings: list[str] = []
    holdout_ratio = val_ratio + test_ratio
    val_share_of_holdout = val_ratio / holdout_ratio

    stratify_first, warning_first = _can_stratify(catalog["label"], n_splits_needed=2)
    if warning_first:
        warnings.append(warning_first)
        logger.warning(warning_first)

    try:
        train_df, holdout_df = train_test_split(
            catalog,
            test_size=holdout_ratio,
            random_state=random_state,
            shuffle=True,
            stratify=catalog["label"] if stratify_first else None,
        )
        stratified = stratify_first
    except ValueError as exc:
        message = (
            "La stratification a échoué lors de la première séparation "
            f"({exc}). Repli sur une séparation non stratifiée."
        )
        warnings.append(message)
        logger.warning(message)
        train_df, holdout_df = train_test_split(
            catalog,
            test_size=holdout_ratio,
            random_state=random_state,
            shuffle=True,
            stratify=None,
        )
        stratified = False

    if len(holdout_df) < 2:
        warning = (
            "Répartition exacte 70/15/15 impossible : le hold-out ne contient "
            f"que {len(holdout_df)} image(s), insuffisant pour créer validation et test."
        )
        warnings.append(warning)
        logger.warning(warning)
        empty_holdout = holdout_df.iloc[0:0].copy()
        return SplitResult(
            train=train_df.reset_index(drop=True),
            validation=holdout_df.reset_index(drop=True),
            test=empty_holdout,
            warnings=warnings,
            grouped_by_patient=grouped_by_patient,
            stratified=False,
        )

    stratify_second, warning_second = _can_stratify(holdout_df["label"], n_splits_needed=2)
    if warning_second:
        warnings.append(
            "La stratification validation/test n'est pas possible pour toutes les classes "
            "dans le hold-out. " + warning_second
        )
        logger.warning(warnings[-1])

    try:
        validation_df, test_df = train_test_split(
            holdout_df,
            test_size=1 - val_share_of_holdout,
            random_state=random_state,
            shuffle=True,
            stratify=holdout_df["label"] if stratify_second else None,
        )
        stratified = stratified and stratify_second
    except ValueError as exc:
        message = (
            "La stratification a échoué lors de la séparation validation/test "
            f"({exc}). Repli sur une séparation non stratifiée."
        )
        warnings.append(message)
        logger.warning(message)
        try:
            validation_df, test_df = train_test_split(
                holdout_df,
                test_size=1 - val_share_of_holdout,
                random_state=random_state,
                shuffle=True,
                stratify=None,
            )
        except ValueError as fallback_exc:
            warning = (
                "Répartition validation/test impossible "
                f"({fallback_exc}). Le hold-out est affecté à la validation."
            )
            warnings.append(warning)
            logger.warning(warning)
            return SplitResult(
                train=train_df.reset_index(drop=True),
                validation=holdout_df.reset_index(drop=True),
                test=holdout_df.iloc[0:0].copy(),
                warnings=warnings,
                grouped_by_patient=grouped_by_patient,
                stratified=False,
            )
        stratified = False

    result = SplitResult(
        train=train_df.reset_index(drop=True),
        validation=validation_df.reset_index(drop=True),
        test=test_df.reset_index(drop=True),
        warnings=warnings,
        grouped_by_patient=grouped_by_patient,
        stratified=stratified,
    )
    _check_target_proportions(result, warnings)
    _assert_no_leakage(result)
    return result


def _split_by_patient_groups(
    catalog: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_state: int,
) -> SplitResult:
    logger.info(
        "Identifiants patient détectés : %s patients pour %s images. "
        "Toutes les images d'un patient restent dans le même ensemble.",
        catalog["patient_id"].nunique(),
        len(catalog),
    )
    patients = (
        catalog.groupby("patient_id", as_index=False)
        .agg(label=("label", lambda series: series.mode().iloc[0]))
        .sort_values("patient_id")
    )
    split_patients = _split_images(
        patients,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
        grouped_by_patient=True,
    )

    mapping = {}
    for split_name, frame in split_patients.as_dict().items():
        for patient_id in frame["patient_id"]:
            mapping[patient_id] = split_name

    assigned = catalog.copy()
    assigned["split"] = assigned["patient_id"].map(mapping)

    result = SplitResult(
        train=assigned.loc[assigned["split"] == "train"].drop(columns=["split"]).reset_index(drop=True),
        validation=assigned.loc[assigned["split"] == "validation"].drop(columns=["split"]).reset_index(drop=True),
        test=assigned.loc[assigned["split"] == "test"].drop(columns=["split"]).reset_index(drop=True),
        warnings=list(split_patients.warnings),
        grouped_by_patient=True,
        stratified=split_patients.stratified,
    )
    _check_target_proportions(result, result.warnings)
    _assert_no_leakage(result)
    return result


def _check_target_proportions(result: SplitResult, warnings: list[str]) -> None:
    total = sum(len(frame) for frame in result.as_dict().values())
    if total == 0:
        return
    for split_name, frame in result.as_dict().items():
        actual = len(frame) / total
        target = TARGET_RATIOS[split_name]
        if abs(actual - target) > 0.02:
            message = (
                f"La proportion de l'ensemble {split_name} est {actual:.1%} "
                f"(cible {target:.0%}). L'écart peut venir d'un nombre insuffisant "
                "d'images ou du regroupement par patient."
            )
            warnings.append(message)
            logger.warning(message)


def _identity_column(frame: pd.DataFrame) -> str:
    if "filepath" in frame.columns:
        return "filepath"
    if "patient_id" in frame.columns:
        return "patient_id"
    return frame.columns[0]


def _assert_no_leakage(result: SplitResult) -> None:
    identity = _identity_column(next(iter(result.as_dict().values())))
    sets = {
        name: set(frame[identity])
        for name, frame in result.as_dict().items()
        if not frame.empty
    }
    overlaps = []
    names = list(sets)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            common = sets[left] & sets[right]
            if common:
                overlaps.append(f"{left}/{right}: {len(common)} chemins partagés")
    if overlaps:
        raise RuntimeError("Fuite de données détectée entre les ensembles : " + "; ".join(overlaps))

    if result.grouped_by_patient:
        patient_sets = {
            name: set(frame["patient_id"])
            for name, frame in result.as_dict().items()
            if not frame.empty
        }
        patient_names = list(patient_sets)
        for i, left in enumerate(patient_names):
            for right in patient_names[i + 1 :]:
                common = patient_sets[left] & patient_sets[right]
                if common:
                    raise RuntimeError(
                        "Fuite patient détectée entre "
                        f"{left} et {right} ({len(common)} patients)."
                    )
