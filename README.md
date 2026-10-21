# Diagnostic des tumeurs cérébrales

Projet de classification d'IRM cérébrales (glioma, meningioma, pituitary, notumor) avec TensorFlow/Keras.

## Dataset

Le dataset attendu est un dossier **un sous-dossier par classe**. Le chemin par défaut est :

```text
Data/
  glioma/
  meningioma/
  notumor/
  pituitary/
```

Extensions autorisées : `.jpeg`, `.jpg`, `.bmp`, `.png` (majuscules/minuscules indifférentes). Les autres fichiers sont ignorés. Les images corrompues sont journalisées dans `reports/invalid_images.csv`.

Si le dataset n'est pas à cet emplacement :

```powershell
$env:DATASET_DIR = "C:\chemin\vers\images"
.venv\Scripts\python.exe scripts\prepare_dataset.py
```

## Installation

Python 3.11 est requis (compatibilité TensorFlow). Avec [uv](https://github.com/astral-sh/uv) :

```powershell
uv venv --python 3.11
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Ou :

```powershell
uv sync
```

## Préparation des données

Cette étape charge et valide les images, explore les classes, produit les visualisations et sépare le dataset en **train 70 % / validation 15 % / test 15 %** (`random_state=42`, stratification par classe). Si des identifiants patient sont présents dans les noms de fichiers, toutes les images d'un même patient restent dans un seul ensemble.

```powershell
.venv\Scripts\python.exe scripts\prepare_dataset.py
```

### Fichiers générés

| Fichier | Contenu |
| --- | --- |
| `reports/dataset_catalog.csv` | Chemins et labels des images valides |
| `reports/class_statistics.csv` | Effectifs par classe |
| `reports/invalid_images.csv` | Images corrompues ou illisibles |
| `reports/skipped_files.csv` | Fichiers d'extension non autorisée |
| `reports/split_summary.csv` | Proportions et distributions après split |
| `reports/splits/train.csv` | Ensemble d'entraînement |
| `reports/splits/validation.csv` | Ensemble de validation |
| `reports/splits/test.csv` | Ensemble de test |
| `reports/figures/countplot_images_par_classe.png` | Nombre d'images par classe |
| `reports/figures/echantillons_par_classe.png` | Grille d'échantillons |

## Tests

```powershell
.venv\Scripts\python.exe -m unittest tests.test_data_pipeline -v
```

## Structure

```text
src/data_pipeline/     # chargement, exploration, visualisation, split
scripts/prepare_dataset.py
Data/                  # images brutes (non versionnées si trop volumineuses)
reports/               # artefacts de préparation
```
