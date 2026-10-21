"""Vérifications du chargement, des extensions et de la séparation."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_pipeline.exploration import build_catalog, extract_patient_id
from data_pipeline.loading import is_allowed_image, load_dataset_images, validate_image
from data_pipeline.splitting import split_dataset


class ExtensionTests(unittest.TestCase):
    def test_allowed_extensions_are_case_insensitive(self) -> None:
        for name in ("a.jpg", "a.JPG", "a.jpeg", "a.JPEG", "a.png", "a.PNG", "a.bmp", "a.BMP"):
            self.assertTrue(is_allowed_image(Path(name)))

    def test_disallowed_extensions_are_rejected(self) -> None:
        for name in ("a.gif", "a.tif", "a.txt", "a", "a.dcm"):
            self.assertFalse(is_allowed_image(Path(name)))


class LoadingTests(unittest.TestCase):
    def test_corrupt_image_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.jpg"
            path.write_bytes(b"not-an-image")
            ok, error = validate_image(path)
            self.assertFalse(ok)
            self.assertTrue(error)

    def test_missing_dataset_is_reported(self) -> None:
        missing = PROJECT_ROOT / "this_dataset_does_not_exist"
        result = load_dataset_images(missing)
        self.assertEqual(result.n_valid, 0)
        self.assertTrue(result.invalid_records)


class SplitTests(unittest.TestCase):
    def test_patient_images_stay_in_one_split(self) -> None:
        records = []
        for patient in range(40):
            label = ["glioma", "meningioma", "notumor", "pituitary"][patient % 4]
            for slice_idx in range(3):
                records.append(
                    {
                        "filepath": f"/tmp/{label}/p{patient}_{slice_idx}.jpg",
                        "label": label,
                        "patient_id": f"p{patient}",
                    }
                )
        import pandas as pd

        catalog = pd.DataFrame(records)
        result = split_dataset(catalog)
        self.assertTrue(result.grouped_by_patient)
        patient_to_split = {}
        for name, frame in result.as_dict().items():
            for patient_id in frame["patient_id"].unique():
                self.assertNotIn(patient_id, patient_to_split)
                patient_to_split[patient_id] = name

        paths = [set(frame["filepath"]) for frame in result.as_dict().values()]
        self.assertFalse(paths[0] & paths[1])
        self.assertFalse(paths[0] & paths[2])
        self.assertFalse(paths[1] & paths[2])

    def test_insufficient_class_samples_emits_warning(self) -> None:
        import pandas as pd

        catalog = pd.DataFrame(
            [
                {"filepath": "/tmp/a.jpg", "label": "glioma", "patient_id": "a"},
                {"filepath": "/tmp/b.jpg", "label": "glioma", "patient_id": "b"},
            ]
        )
        result = split_dataset(catalog)
        self.assertTrue(result.warnings)

    def test_patient_id_fallback_is_unique_per_image(self) -> None:
        first = extract_patient_id("C:/data/glioma/Te-gl_0001.jpg")
        second = extract_patient_id("C:/data/glioma/Te-gl_0002.jpg")
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
