# Test #1 & Hackathon — Machine Learning & Deep Learning

> Dry Bean classification (Kernel SVM vs ANN) and Rock-Paper-Scissors image classification (custom CNN vs Transfer Learning with VGG16).

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-SVM-yellow)
![Colab](https://img.shields.io/badge/Google%20Colab-ready-green)


---

## Table of contents

1. [Overview](#overview)
2. [Repository structure](#repository-structure)
3. [Part 1 — Dry Bean classification (tabular data)](#part-1--dry-bean-classification-tabular-data)
4. [Part 2 — Rock-Paper-Scissors (images)](#part-2--rock-paper-scissors-images)
5. [How to run](#how-to-run)
6. [Configuration flags](#configuration-flags)
7. [Good practices used](#good-practices-used)
8. [Tech stack](#tech-stack)
9. [Author](#author)

---

## Overview

This project is the solution to the *Test #1 (Final) & Hackathon* of the Machine Learning module. It is made of two mandatory parts, both implemented in **one Jupyter Notebook** (`Test1_Hackathon.ipynb`):

| Part | Subject | Models compared |
|---|---|---|
| **Part 1** | Dry Bean classification — 7 classes, 16 features, 13 611 grains | Kernel SVMs (linear, polynomial, RBF, sigmoid) **vs** ANN |
| **Part 2** | Rock-Paper-Scissors (*pierre*, *papier*, *ciseaux*) — ~5 000 colour images | Custom CNN (v1 → improved v2) **vs** Transfer Learning (VGG16) |

The notebook works both on **Google Colab** (with Google Drive) and on a **local Jupyter** environment.

---

## Dataset

The dataset used for this project (including the image files and resources) is hosted on Google Drive because of its large size. You can download it from the link below:

* **[Download Dataset (Google Drive)](https://drive.google.com/drive/u/0/folders/1gJSgHCwAmtXdSPPPBd6oB57zHeLcmwgJ)**

---

## Part 1 — Dry Bean classification (tabular data)

**Goal:** classify 7 varieties of dry beans — *Seker, Barbunya, Bombay, Cali, Dermason, Horoz, Sira* — from 16 geometric measurements of each grain (12 dimensions + 4 shape factors: Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRatio, Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, Roundness, Compactness, ShapeFactor1–4).

### Pipeline

1. **Import** the dataset (CSV or Excel, separator auto-detected).
2. **Missing values & duplicates:** median imputation, duplicated rows removed (avoids train/test leakage).
3. **Exploratory analysis:** class histogram, correlation matrix, PCA (dimensionality check).
4. **Encoding:** `LabelEncoder` on the target.
5. **Split first:** 80 % train / 10 % validation / 10 % test, *stratified*.
6. **Scaling:** `StandardScaler` fitted on the train set only.
7. **Class balancing:** **SMOTE** applied on the **training set only** (validation and test sets keep the real distribution → honest scores).
8. **Kernel SVMs:** `linear`, `poly` (degree 3), `rbf`, `sigmoid`, then a grid search on `C` and `gamma` for the RBF kernel (scored on the validation set).
9. **ANN:** `Input(16) → Dense(128) → BN → ReLU → Dropout(0.3) → Dense(64) → BN → ReLU → Dropout(0.3) → Dense(32) → Dense(7, softmax)`, Adam + sparse categorical cross-entropy, EarlyStopping + ReduceLROnPlateau.
10. **Comparison:** accuracy, macro-F1, training time, confusion matrices, per-class F1.

### Main findings (from the author's run)

- **RBF SVM and ANN are practically tied** at ≈ **92 % accuracy** (macro-F1 ≈ 0.93); the difference is within the statistical noise of the test set.
- The **linear SVM** is only ≈ 1 point lower, the **polynomial** ≈ 2 points lower, and the **sigmoid** kernel is far behind (≈ 69 %) because it is not tuned and not a valid kernel in general.
- **Bombay** is perfectly classified (much larger beans); most errors are between **Sira, Dermason, Seker, Barbunya and Cali**, which have very similar shapes.
- PCA shows that ≈ 4 components keep 95 % of the variance: the features are highly redundant, so a more complex model (ANN) brings no real gain over a kernel SVM, which trains much faster.

> Exact numbers can move by a few tenths of a point depending on the machine; the results table printed by the notebook is the reference.

---

## Part 2 — Rock-Paper-Scissors (images)

**Goal:** classify hand images (white background, RGB, resized to 128×128) into **pierre / papier / ciseaux**.

### Pipeline

1. **Load & visualise** the images, plot the class-frequency histogram.
2. **Split first:** 80 % train / 10 % validation / 10 % test, stratified (done *before* augmentation → no leakage).
3. **Augmentation**, used for two purposes:
   - *(a)* balance the classes (augmented copies of the minority classes in the training set),
   - *(b)* create variability (augmentation layers inside the models: flip, rotation, zoom, translation, contrast).
4. **CNN v1:** 3 × [Conv2D + MaxPooling] → Flatten → Dense(128) → Dropout → Dense(3, softmax). Learning rate and dropout tuned on the validation set.
5. **CNN v2 (improved):** in-model augmentation, 4 blocks of 2 × Conv2D (32-64-128-256) with BatchNorm, GlobalAveragePooling instead of Flatten, dropout, EarlyStopping + ReduceLROnPlateau.
6. **Transfer Learning:** **VGG16** pre-trained on ImageNet, base **frozen**, custom head `GlobalAveragePooling → Dense(256) → Dropout(0.5) → Dense(3, softmax)`.
7. **Bonus — fine-tuning:** unfreeze `block5` of VGG16 and continue training with a very small learning rate (`1e-5`).
8. **Comparison:** accuracy, macro-F1, number of (trainable) parameters, epochs, training time, inference time, confusion matrices, misclassified images.

### Main findings

- v1 tends to **overfit** (large `Flatten → Dense` layer); v2 reduces the train/validation gap thanks to augmentation, BatchNorm, GlobalAveragePooling and dropout.
- **Transfer learning** converges in very few epochs with only ≈ 132 k trainable parameters (out of ≈ 14.8 M) and generalises well.
- The task is easy (3 very different shapes, uniform background), so both families can reach a very high accuracy. The custom CNN is **much lighter and faster** at inference — a better candidate for embedded / mobile systems — while VGG16 is heavier (≈ 15 M parameters).

> The exact scores of each model are written by the notebook in `part2_models/part2_results.csv`.

> **Note on the split:** the statement says *"80 % pour test et 20 % pour la validation"*, which is considered a typo (80 % should be for **training**). The notebook uses 80 / 10 / 10, which covers both readings of the 20 %.

---

## How to run

### Option A — Google Colab (recommended)

1. In your Google Drive (*My Drive*), create a folder **`Test1_Hackathon`**.
2. Put inside it:
   - `Dry_Bean_Dataset.csv` (or the `.xlsx` version) — [Dry Bean Dataset on UCI](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset),
   - the images folder `Papier-pierre-sciseaux` (sub-folders `papier`, `pierre`, `sciseaux`) **or** the zip `pierre-papier-ciseaux.zip`.
3. Open the notebook in Colab, then `Runtime → Change runtime type → GPU` (T4) — it makes Part 2 much faster.
4. `Runtime → Run all` and allow access to your Drive once.

### Option B — Local Jupyter

```bash
git clone https://github.com/Naadix/<your-repo-name>.git
cd <your-repo-name>

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn \
            tensorflow pillow joblib openpyxl jupyter

jupyter notebook Test1_Hackathon.ipynb
```

Create a `Test1_Hackathon/` folder next to the notebook and place the datasets inside it (same layout as above). Outside Colab the notebook automatically uses `./Test1_Hackathon`.

### Save / load of the models

Every trained model (SVMs, ANN, CNN v1/v2, VGG16) is **saved automatically** in `part1_models/` and `part2_models/`. On the next run the saved models are **loaded instead of retrained** (seconds instead of minutes). The last cells of each part show how to predict a new bean / a new image from the saved models.

---

## Configuration flags

Set in the configuration cell at the top of the notebook:

| Flag | Default | Effect |
|---|---|---|
| `RETRAIN` | `False` | `True` forces a new training even if a saved model exists |
| `QUICK` | `False` | `True` runs a tiny 2-epoch version to check that everything works |
| `IMG_SIZE` | `128` | Images are resized to `IMG_SIZE × IMG_SIZE` |
| `BATCH` | `32` | Batch size for Part 2 |
| `EPOCHS` | `{ann:150, cnn1:20, cnn2:40, vgg:15, vgg_ft:8}` | Maximum epochs (early stopping stops before) |
| `RUN_FINE_TUNING` | `True` | Set to `False` to skip the VGG16 fine-tuning bonus |
| `DATA_DIR` | `None` | Set the images path by hand if the automatic search fails |

---

## Good practices used

- **Split before balancing / augmenting** → no data leakage.
- **Scaler fitted on the train set only.**
- **SMOTE / augmentation on the training set only** → validation and test sets stay realistic.
- **Validation set for tuning**, test set used only at the very end.
- **Macro-F1** reported in addition to accuracy (the Dry Bean test set is still imbalanced).
- **Fixed random seed** (`42`) for reproducibility.
- **Models saved and reloadable** (`.joblib` for scikit-learn, `.keras` for TensorFlow).

---

## Tech stack

Python · NumPy · Pandas · Matplotlib · Seaborn · scikit-learn · imbalanced-learn (SMOTE) · TensorFlow / Keras · Pillow · Joblib · Jupyter / Google Colab
