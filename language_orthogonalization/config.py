"""Paths (via env vars, see README), backbone list, and sweep grids."""
from __future__ import annotations
import os
from pathlib import Path

FEAT_ROOT   = Path(os.environ.get("LOPD_FEAT_ROOT",  "./features"))
SPLIT_PATH  = Path(os.environ.get("LOPD_SPLIT_PATH", "./manifests/splits_5fold.csv"))
OUT_DIR     = Path(os.environ.get("LOPD_OUT_DIR",    "./results"))

LID_DIR = "voxlingua_lid"

SSL_MODELS = [
    "hubert_large_layers",
    "wavlm_layers",
    "xls_r_300m_layers",
    "mms_300m_layers",
    "w2v2_large_lv60_layers",
]

NONSSL_MODELS = [
    "ecapa",
    "whisper_large_layers",
    "ast_audioset_layers",
]

TASKS = ["vowel", "ddk", "read"]
LANGS = ["deu", "cze", "spa"]

ALPHAS       = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
SENS_TARGETS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
INNER_K = 5
