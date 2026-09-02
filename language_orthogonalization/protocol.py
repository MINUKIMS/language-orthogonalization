"""Cross-lingual transfer protocol (source → target) — see README."""
from __future__ import annotations
from typing import Iterable

import numpy as np
from sklearn.preprocessing import StandardScaler

from .classify import (lr_fit_predict, lr_inner_oof,
                       threshold_for_sens, metrics_at)
from .methods import apply_ls, hc_centroids, lid_residualize


def build_reps(X, langs, groups, target, train_mask, hc_train_mask,
                G, alphas: Iterable[float]):
    """Return [(rep_name, X_transformed), ...] for raw / +LS / LO(α)."""
    reps = [("raw", X)]
    mu = hc_centroids(X, langs, groups, train_mask)
    reps.append(("+LS", apply_ls(X, langs, mu, target)))
    if G is not None:
        sc_g = StandardScaler().fit(G[hc_train_mask])
        Gs = sc_g.transform(G)
        for a in alphas:
            reps.append((f"+LO_a{a}", lid_residualize(X, Gs, hc_train_mask, a)))
    return reps


def cross_lingual_folds(langs, groups, folds, target):
    """Yield (fold_id, train_mask, eval_mask, hc_train_mask) for one target."""
    is_t     = langs == target
    is_t_hc  = is_t & (groups == "HC")
    is_t_pd  = is_t & (groups == "PD")
    for k in range(5):
        test_hc     = is_t_hc & (folds == k)
        train_hc_t  = is_t_hc & (folds != k)
        train_mask  = (~is_t) | train_hc_t
        eval_mask   = test_hc | is_t_pd
        hc_train_mask = train_mask & (groups == "HC")
        yield k, train_mask, eval_mask, hc_train_mask


def evaluate_rep(X_rep, train_mask, eval_mask, y_all,
                  sens_targets: Iterable[float]):
    """Return list of {sens_target, sens, spec, f1} per target."""
    y_tr = y_all[train_mask]
    y_te = y_all[eval_mask]
    if len(set(y_tr)) < 2 or len(set(y_te)) < 2:
        return None
    oof  = lr_inner_oof(X_rep[train_mask], y_tr)
    s_te = lr_fit_predict(X_rep[train_mask], y_tr, X_rep[eval_mask])
    rows = []
    for st in sens_targets:
        thr = threshold_for_sens(y_tr, oof, st)
        if not np.isfinite(thr):
            sens = spec = f1 = float("nan")
        else:
            sens, spec, f1 = metrics_at(s_te, y_te, thr)
        rows.append({"sens_target": st, "sens": sens, "spec": spec, "f1": f1})
    return rows
