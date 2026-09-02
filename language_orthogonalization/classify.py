"""Classifier + operating-point utilities."""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .config import INNER_K


def lr_fit_predict(X_tr, y_tr, X_te):
    """LR with class_weight='balanced' → decision_function is the LLR
    under a 0.5 training prior."""
    sc = StandardScaler().fit(X_tr)
    clf = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced")
    clf.fit(sc.transform(X_tr), y_tr)
    return clf.decision_function(sc.transform(X_te))


def lr_inner_oof(X, y, seed: int = 0):
    """5-fold stratified OOF LLR scores on the training set, used to
    pick a threshold at a given target sensitivity without touching the
    held-out target evaluation set."""
    skf = StratifiedKFold(INNER_K, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=np.float64)
    for tr, val in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced")
        clf.fit(sc.transform(X[tr]), y[tr])
        oof[val] = clf.decision_function(sc.transform(X[val]))
    return oof


def threshold_for_sens(y, s, t: float) -> float:
    """Smallest threshold on ROC that attains sensitivity ≥ t."""
    if len(set(y)) < 2:
        return float("nan")
    fpr, tpr, thr = roc_curve(y, s)
    idx = np.searchsorted(tpr, t, side="left")
    if idx >= len(thr):
        idx = len(thr) - 1
    return float(thr[idx]) if idx >= 0 else float("nan")


def metrics_at(s_te, y_te, thr):
    """sens, spec, binary-F1 at a given threshold."""
    pred = (s_te >= thr).astype(int)
    cm = confusion_matrix(y_te, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    f1 = float(f1_score(y_te, pred, average="binary",
                        pos_label=1, zero_division=0))
    return sens, spec, f1
