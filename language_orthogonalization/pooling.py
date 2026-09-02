"""Feature loading + speaker pooling."""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd


def load_features(feat_root: Path, name: str):
    """Load features and matching index.  Auto-detects (L,N,D) vs (N,D)."""
    e = np.load(Path(feat_root) / name / "embeddings.npy",
                mmap_mode="r").astype(np.float32)
    if e.ndim == 2:
        feats = e
    else:
        # per-layer L2 normalize then uniform mean across layers
        norm = np.linalg.norm(e, axis=-1, keepdims=True) + 1e-12
        feats = (e / norm).mean(axis=0)
    idx = pd.read_csv(Path(feat_root) / name / "index.csv")
    return np.asarray(feats, dtype=np.float32), idx


def speaker_pool(emb: np.ndarray, idx: pd.DataFrame, task: str) -> pd.DataFrame:
    """Average utterance features per speaker for a single task."""
    df = idx.copy()
    df["_row"] = np.arange(len(df))
    df = df[df["task"] == task]
    rows = []
    for sid, g in df.groupby("speaker_id"):
        rows.append({
            "speaker_id": sid,
            "lang":       g["lang"].iloc[0],
            "group":      g["group"].iloc[0],
            "feat":       emb[g["_row"].to_numpy()].mean(axis=0),
        })
    return pd.DataFrame(rows)
