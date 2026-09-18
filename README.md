# Language Orthogonalization of Self-Supervised Speech Representations for Cross-lingual Parkinson's Detection

Minu Kim, Eunjung Yeo, Kwanghee Choi, June-Woo Kim. IEEE SLT 2026. **[Paper](https://arxiv.org/abs/2609.09499)**

Official implementation of language orthogonalization (LO) for cross-lingual Parkinson's detection. The repository includes S3M feature extraction and pooling, the proposed LO method, the language shift (LS) baseline from Hernández et al. (2024), and the evaluation protocol used in the paper.

## Structure

```text
language_orthogonalization/
├── config.py
├── extract_features.py
├── pooling.py
├── methods.py
├── classify.py
├── protocol.py
├── run_sens_curve.py
└── run_lid_check.py
```

## Data

Each backbone should have the following structure:

```text
$LOPD_FEAT_ROOT/<backbone>/
├── embeddings.npy
└── index.csv
```

`embeddings.npy` has shape `(L, N, D)` for layer-wise models or `(N, D)` for utterance-level models. `index.csv` contains:

```text
speaker_id, lang, cohort, group, task
```

VoxLingua107 LID embeddings should be placed under `$LOPD_FEAT_ROOT/voxlingua_lid/` in the same format.

## Usage

```bash
pip install -r language_orthogonalization/requirements.txt

# Extract features for one backbone or all backbones
python -m language_orthogonalization.extract_features hubert_large

# Main experiments
python -m language_orthogonalization.run_sens_curve

# Non-S3M controls
python -m language_orthogonalization.run_sens_curve --nonssl

# Language-information analysis
python -m language_orthogonalization.run_lid_check --per-task
```

Paths can be configured with:

```bash
LOPD_FEAT_ROOT=./features
LOPD_MANIFEST=./manifests/manifest.csv
LOPD_SPLIT_PATH=./manifests/splits_5fold.csv
LOPD_OUT_DIR=./results
```

Results are saved under `$LOPD_OUT_DIR`.

## Citation

If you find this repository or our method useful, please cite our papers using the following BibTeX entries:

```bibtex
@article{kim2026parkinson,
  title={Language Orthogonalization of Self-Supervised Speech Representations for Cross-lingual Parkinson's Detection},
  author={Kim, Minu and Yeo, Eunjung and Choi, Kwanghee and Kim, June-Woo},
  journal={arXiv preprint arXiv:2609.09499},
  year={2026}
}
```
