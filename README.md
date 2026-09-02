# Language Orthogonalization of Self-Supervised Speech Representations for Cross-lingual Parkinson's Detection

Reference implementation of language orthogonalization (LO) for
cross-lingual Parkinson's detection. Covers S3M feature pooling, the
LO transform (HC-only ridge residualization against a VoxLingua107 LID
embedding), the LS baseline from Hernández et al. 2024, and the
source-to-target evaluation protocol used in the paper.

## Layout

```
language_orthogonalization/
├── config.py         # paths, backbone list, sweep grids
├── pooling.py        # feature loading, speaker pooling
├── methods.py        # LO and LS transforms
├── classify.py       # classifier and threshold utilities
├── protocol.py       # cross-lingual folds and evaluation
├── run_sens_curve.py # main experiment
└── run_lid_check.py  # LID classification check
```

## Data layout expected

Each backbone lives under `$LOPD_FEAT_ROOT/<backbone>/`:

* `embeddings.npy`: array of shape `(L, N, D)` for layer-wise S3M
  backbones (per-layer L2-normalized frame means; the loader applies
  the uniform layer mean), or `(N, D)` for utterance-level backbones
  such as ECAPA-TDNN.
* `index.csv`: one row per utterance with columns
  `speaker_id, lang, cohort, group, task` (rows aligned to
  `embeddings.npy` axis `N`).

The 5-fold speaker split lives at `$LOPD_SPLIT_PATH`:

```
speaker_id, lang, cohort, group, age, gender, has_metadata, fold
```

`fold ∈ {0,…,4}` is a per-language stratified split over the HC pool.

The language-nuisance embedding is expected at
`$LOPD_FEAT_ROOT/voxlingua_lid/` in the same format (SpeechBrain's
VoxLingua107 ECAPA-CNN, 256-d).  Any per-utterance language embedding
of comparable capacity is a drop-in replacement.

## Environment variables

```
LOPD_FEAT_ROOT   # default: ./features
LOPD_SPLIT_PATH  # default: ./manifests/splits_5fold.csv
LOPD_OUT_DIR     # default: ./results
```

## Running

```bash
pip install -r language_orthogonalization/requirements.txt

# Main experiment: 5 S3M backbones x 3 tasks x 3 targets x 5 folds
python -m language_orthogonalization.run_sens_curve

# Non-S3M controls (ECAPA-TDNN / Whisper-Large / AST)
python -m language_orthogonalization.run_sens_curve --nonssl

# Check whether LO removes language information from the features
python -m language_orthogonalization.run_lid_check --per-task
```

Each script writes a tidy CSV under `$LOPD_OUT_DIR/`.

## Method summary

Given per-utterance layer-wise S3M embeddings `E ∈ ℝ^(L×N×D)`, each
`(N,D)` layer is L2-normalized and the uniform layer mean produces
`X ∈ ℝ^(N×D)`.  Speaker features are the within-speaker mean of `X`
for a given task.  The nuisance embedding `G` is the speaker mean of a
VoxLingua107 LID model, standardized on the training-fold HC pool.
LO fits `W = Ridge(α).fit(G_HC, X_HC)` on healthy speakers only and
returns `X_clean = X − G · Wᵀ`; the same `W` is applied to PD.  On
each held-out target-language HC fold a class-balanced logistic
regression is trained on all source-language speakers plus the target
HC training fold, and (sens, spec, F1) are evaluated on the target
HC test fold together with all target PD.
