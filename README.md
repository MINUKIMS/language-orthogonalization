# Language Orthogonalization for Cross-lingual Parkinson's Detection

Official implementation of  
**[Language Orthogonalization of Self-Supervised Speech Representations for Cross-lingual Parkinson's Detection](https://arxiv.org/abs/2609.09499)**  
Minu Kim et al., submitted to IEEE SLT 2026.

This repository implements language orthogonalization (LO), a method for reducing language-dependent variation in self-supervised speech representations for cross-lingual Parkinson's disease (PD) detection. It includes feature extraction and pooling, the proposed LO method, the language shift (LS) baseline from Hernández et al. (2024), and the source-to-target evaluation protocol used in the paper.

## Repository structure

```text
language_orthogonalization/
├── config.py             # Paths, backbones, and hyperparameter grids
├── extract_features.py   # Layer-wise S3M feature extraction
├── pooling.py            # Feature loading and speaker-level pooling
├── methods.py            # LO and LS implementations
├── classify.py           # Classifier and threshold utilities
├── protocol.py           # Cross-lingual evaluation protocol
├── run_sens_curve.py     # Main experiments
└── run_lid_check.py      # Language-information analysis
```

## Data preparation

Features for each backbone should be stored under:

```text
$LOPD_FEAT_ROOT/<backbone>/
├── embeddings.npy
└── index.csv
```

`embeddings.npy` should have shape `(L, N, D)` for layer-wise S3M features or `(N, D)` for utterance-level models such as ECAPA-TDNN. Here, `L`, `N`, and `D` denote the number of layers, utterances, and feature dimensions, respectively.

`index.csv` should contain one row per utterance, aligned with axis `N` of `embeddings.npy`, with the following columns:

```text
speaker_id, lang, cohort, group, task
```

The speaker-level cross-validation split is specified by `$LOPD_SPLIT_PATH`:

```text
speaker_id, lang, cohort, group, age, gender, has_metadata, fold
```

The `fold` column takes values from 0 to 4 and defines a language-stratified five-fold split of the healthy control (HC) speakers.

The VoxLingua107 language embeddings should follow the same format and be placed under:

```text
$LOPD_FEAT_ROOT/voxlingua_lid/
```

We use 256-dimensional embeddings from SpeechBrain's VoxLingua107 ECAPA-CNN model. Other utterance-level language embeddings can be used in the same format.

## Feature extraction

`extract_features.py` extracts features from the five S3M backbones considered in the paper, as well as Whisper-Large and AST. For models supporting `output_hidden_states=True`, it stores all hidden states, including layer 0 corresponding to the CNN feature encoder.

ECAPA-TDNN features and VoxLingua107 language embeddings must be extracted separately using SpeechBrain.

## Configuration

The following environment variables control the input and output paths:

```bash
LOPD_FEAT_ROOT=./features
LOPD_MANIFEST=./manifests/manifest.csv
LOPD_SPLIT_PATH=./manifests/splits_5fold.csv
LOPD_OUT_DIR=./results
```

## Usage

Install the required packages:

```bash
pip install -r language_orthogonalization/requirements.txt
```

Extract features for one backbone:

```bash
python -m language_orthogonalization.extract_features hubert_large
```

Use `all` in place of the backbone name to extract features for all supported models.

Run the main experiments:

```bash
python -m language_orthogonalization.run_sens_curve
```

Run the experiments for the non-S3M controls:

```bash
python -m language_orthogonalization.run_sens_curve --nonssl
```

Evaluate the language information remaining after orthogonalization:

```bash
python -m language_orthogonalization.run_lid_check --per-task
```

Results are saved as CSV files under `$LOPD_OUT_DIR`.

## Method overview

For layer-wise S3M representations, each layer is L2-normalized and the representations are uniformly averaged across layers. Utterance-level features are then averaged within each speaker and speech task.

LO estimates the component of the S3M representation that is predictable from an external language embedding. A ridge regression is fitted using only HC speakers in the training fold, and its prediction is subtracted from both HC and PD representations:

```text
W = Ridge(alpha).fit(G_HC, X_HC)
X_LO = X - G W^T
```

Here, `X` denotes the S3M representation and `G` denotes the standardized VoxLingua107 embedding. Standardization and ridge fitting use only the training-fold HC speakers.

For each target language and fold, a class-balanced logistic regression classifier is trained using all source-language speakers and the target-language HC training speakers. It is evaluated on the held-out target-language HC speakers and the target-language PD speakers. The reported metrics are sensitivity, specificity, and F1 score.
