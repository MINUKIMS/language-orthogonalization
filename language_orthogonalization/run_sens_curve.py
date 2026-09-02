"""Main experiment: sens-target curve across raw / +LS / +LO(α).
See README for usage."""
from __future__ import annotations
import argparse

import numpy as np
import pandas as pd

from .config import (ALPHAS, FEAT_ROOT, LANGS, LID_DIR, NONSSL_MODELS,
                     OUT_DIR, SENS_TARGETS, SPLIT_PATH, SSL_MODELS, TASKS)
from .pooling import load_features, speaker_pool
from .protocol import build_reps, cross_lingual_folds, evaluate_rep


def _merge_speakers(sp_feat, sp_lid, splits):
    return (sp_feat.rename(columns={"feat": "f"})
            .merge(sp_lid[["speaker_id", "feat"]]
                   .rename(columns={"feat": "g"}),
                   on="speaker_id", how="inner")
            .merge(splits[["speaker_id", "fold"]],
                   on="speaker_id", how="inner"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nonssl", action="store_true",
                   help="use ECAPA/Whisper/AST instead of the 5 SSL backbones")
    p.add_argument("--model", type=str, default=None,
                   help="run a single backbone name (overrides --nonssl)")
    p.add_argument("--out", type=str, default=None,
                   help="output CSV path (default: $LOPD_OUT_DIR/results_sens_curve[_nonssl].csv)")
    args = p.parse_args()

    if args.model:
        models = [args.model]
    elif args.nonssl:
        models = NONSSL_MODELS
    else:
        models = SSL_MODELS

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = args.out or str(
        OUT_DIR / ("results_sens_curve_nonssl.csv" if args.nonssl
                    else "results_sens_curve.csv"))

    splits = pd.read_csv(SPLIT_PATH)
    splits = splits[(splits["cohort"] == "main") & (splits["fold"] >= 0)].copy()
    lid_emb, lid_idx = load_features(FEAT_ROOT, LID_DIR)

    rows = []
    for task in TASKS:
        lid_sp = speaker_pool(lid_emb, lid_idx, task)
        for model in models:
            print(f"=== {model} | {task} ===", flush=True)
            feats, idx = load_features(FEAT_ROOT, model)
            sp = speaker_pool(feats, idx, task)
            df = _merge_speakers(sp, lid_sp, splits)
            if len(df) == 0:
                continue
            X      = np.stack(df["f"].to_list(), axis=0).astype(np.float32)
            G      = np.stack(df["g"].to_list(), axis=0).astype(np.float32)
            langs  = df["lang"].to_numpy()
            groups = df["group"].to_numpy()
            folds  = df["fold"].to_numpy()
            y_all  = (groups == "PD").astype(int)

            for target in LANGS:
                for k, train_mask, eval_mask, hc_tr in \
                        cross_lingual_folds(langs, groups, folds, target):
                    if (train_mask.sum() == 0 or eval_mask.sum() == 0):
                        continue
                    reps = build_reps(X, langs, groups, target,
                                       train_mask, hc_tr, G, ALPHAS)
                    for rep_name, X_rep in reps:
                        got = evaluate_rep(X_rep, train_mask, eval_mask,
                                            y_all, SENS_TARGETS)
                        if got is None:
                            continue
                        for r in got:
                            rows.append({"model": model, "task": task,
                                         "target": target, "fold": k,
                                         "rep": rep_name, **r})

    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    print(f"\nsaved: {out_csv}  ({len(out)} rows)")

    # quick summary
    rep_order = ["raw", "+LS"] + [f"+LO_a{a}" for a in ALPHAS]
    print("\n---- mean (sens, spec, F1) per rep × sens_target ----")
    for st in SENS_TARGETS:
        sub = out[out["sens_target"] == st]
        agg = (sub.groupby("rep")[["sens", "spec", "f1"]]
                .mean().round(3).reindex(rep_order))
        print(f"\n= sens_target = {st} =")
        print(agg.to_string())


if __name__ == "__main__":
    main()
