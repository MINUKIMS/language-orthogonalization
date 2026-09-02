# Language Orthogonalization of Self-Supervised Speech Representations for Cross-lingual Parkinson's Detection

Reference implementation of language orthogonalization (LO) and the
cross-lingual PD-detection protocol used in the paper.  The scope of
this repository is:

* layer pooling of self-supervised speech-model (S3M) features,
* the LO transform — HC-only ridge residualization of S3M features
  against a VoxLingua107 language-identification embedding,
* the language-shift (LS) baseline of Hernández et al. 2024, included
  for reproducibility of the paper's comparison tables,
* the source-to-target cross-lingual evaluation protocol
  (train: source-language HC + PD plus target HC train fold;
   evaluate: target HC test fold + target PD).

Feature-extraction pipelines, figure code, and ablations are omitted;
bring your own per-utterance S3M layer stack and a per-utterance
VoxLingua107 LID embedding.

## Layout

```
language_orthogonalization/
├── config.py         # paths (via env vars), backbone list, α / sens grids
├── pooling.py        # load_features, speaker_pool
├── methods.py        # lid_residualize (LO), apply_ls (Hernández baseline)
├── classify.py       # LR train/predict, inner-OOF thresholding, metrics
├── protocol.py       # cross-lingual folds + representation builder
├── run_sens_curve.py # sens-target × α sweep
└── run_lid_check.py  # LID-classification sanity check
```

## Data layout expected

Each backbone lives under `$LOPD_FEAT_ROOT/<backbone>/`:

* `embeddings.npy` — array of shape `(L, N, D)` for layer-wise S3M
  backbones (per-layer L2-normalized frame means; the loader applies
  the uniform layer mean), or `(N, D)` for utterance-level backbones
  such as ECAPA-TDNN.
* `index.csv` — one row per utterance with columns
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

# Main experiment — 5 S3M backbones × 3 tasks × 3 targets × 5 folds
python -m language_orthogonalization.run_sens_curve

# Non-S3M controls (ECAPA-TDNN / Whisper-Large / AST)
python -m language_orthogonalization.run_sens_curve --nonssl

# LID-classification sanity check (does LO suppress language cues?)
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
HC training slice, and (sens, spec, F1) are evaluated on the target
HC test fold together with all target PD.

## Notes

* Ridge with `α = 0` is clamped to `1e-8` for numerical stability;
  the fit is under-determined whenever the HC training set is smaller
  than the LID embedding dimension.
