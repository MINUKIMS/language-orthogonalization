"""LID-classification sanity check: does LO make language harder to
recover from speaker-pooled features? See README for usage."""
from __future__ import annotations
import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .config import (ALPHAS, FEAT_ROOT, LID_DIR, NONSSL_MODELS,
                     OUT_DIR, SPLIT_PATH, SSL_MODELS, TASKS)
from .methods import apply_ls, hc_centroids, lid_residualize
from .pooling import load_features, speaker_pool

warnings.filterwarnings("ignore")

N_FOLDS   = 5
LS_TARGET = "cze"   # arbitrary canonical anchor language


def cv_metrics(X, y_lang, groups, langs, G=None,
                method: str = "raw", alpha=None):
    m_hc = {"acc": [], "bacc": [], "mf1": []}
    m_pd = {"acc": [], "bacc": [], "mf1": []}
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=0)
    for tr, te in skf.split(X, y_lang):
        if method == "raw":
            X_in = X
        elif method == "ls":
            mus = hc_centroids(X, langs, groups,
                                np.isin(np.arange(len(X)), tr))
            X_in = apply_ls(X, langs, mus, LS_TARGET)
        elif method == "lo":
            hc_tr = tr[groups[tr] == "HC"]
            if len(hc_tr) < 10:
                continue
            mask = np.zeros(len(X), dtype=bool); mask[hc_tr] = True
            sc_g = StandardScaler().fit(G[hc_tr])
            Gs = sc_g.transform(G)
            X_in = lid_residualize(X, Gs, mask, alpha)
        else:
            raise ValueError(method)
        sc = StandardScaler().fit(X_in[tr])
        clf = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced")
        clf.fit(sc.transform(X_in[tr]), y_lang[tr])
        pred = clf.predict(sc.transform(X_in[te]))
        for grp, bank in (("HC", m_hc), ("PD", m_pd)):
            sel = groups[te] == grp
            if not sel.any():
                continue
            y, p = y_lang[te][sel], pred[sel]
            bank["acc"].append((p == y).mean())
            bank["bacc"].append(balanced_accuracy_score(y, p))
            bank["mf1"].append(f1_score(y, p, labels=[0, 1, 2], average="macro"))

    def _agg(d):
        return {k: (float(np.mean(v)), float(np.std(v))) for k, v in d.items()}
    return _agg(m_hc), _agg(m_pd)


def _row(rep_name, mh, mp, alpha, **extra):
    row = {"rep": rep_name, "alpha": alpha, **extra}
    for k in ("acc", "bacc", "mf1"):
        row[f"{k}_hc"], row[f"std_{k}_hc"] = mh[k]
        row[f"{k}_pd"], row[f"std_{k}_pd"] = mp[k]
    return row


def run_for(feats, idx, lid_emb, lid_idx, splits, task, label, per_task_col):
    lid_sp = speaker_pool(lid_emb, lid_idx, task)
    sp = speaker_pool(feats, idx, task)
    df = (sp.rename(columns={"feat": "f"})
            .merge(lid_sp[["speaker_id", "feat"]]
                   .rename(columns={"feat": "g"}),
                   on="speaker_id", how="inner")
            .merge(splits[["speaker_id", "fold"]],
                   on="speaker_id", how="inner"))
    if len(df) == 0:
        return []
    X = np.stack(df["f"].to_list(), axis=0).astype(np.float32)
    G = np.stack(df["g"].to_list(), axis=0).astype(np.float32)
    groups = df["group"].to_numpy()
    langs  = df["lang"].to_numpy()
    y_lang = np.array([{"deu": 0, "cze": 1, "spa": 2}[l] for l in langs])

    extra = {"backbone": label}
    if per_task_col:
        extra["task"] = task
    rows = []
    mh, mp = cv_metrics(X, y_lang, groups, langs, method="raw")
    rows.append(_row("raw", mh, mp, None, **extra))
    mh, mp = cv_metrics(X, y_lang, groups, langs, method="ls")
    rows.append(_row("+LS", mh, mp, None, **extra))
    for a in ALPHAS:
        mh, mp = cv_metrics(X, y_lang, groups, langs, G=G,
                             method="lo", alpha=a)
        rows.append(_row(f"+LO α={a}", mh, mp, a, **extra))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nonssl", action="store_true")
    p.add_argument("--per-task", action="store_true",
                   help="run each of vowel/ddk/read (default: read only)")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    models = NONSSL_MODELS if args.nonssl else SSL_MODELS
    tasks  = TASKS if args.per_task else ["read"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_per_task" if args.per_task else ("_nonssl" if args.nonssl else "")
    out_csv = args.out or str(OUT_DIR / f"results_lid_clf{suffix}.csv")

    splits = pd.read_csv(SPLIT_PATH)
    splits = splits[(splits["cohort"] == "main") & (splits["fold"] >= 0)].copy()
    lid_emb, lid_idx = load_features(FEAT_ROOT, LID_DIR)

    rows = []
    for model in models:
        print(f"=== {model} ===", flush=True)
        feats, idx = load_features(FEAT_ROOT, model)
        for task in tasks:
            rows.extend(run_for(feats, idx, lid_emb, lid_idx, splits,
                                 task, model, per_task_col=args.per_task))

    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    print(f"\nsaved: {out_csv}  ({len(out)} rows)")


if __name__ == "__main__":
    main()
