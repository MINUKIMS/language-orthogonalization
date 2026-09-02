"""Language-nuisance removal: LO (this paper) and the LS baseline
(Hernández et al. 2024). See README for the method summary."""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import Ridge

from .config import LANGS


def lid_residualize(X, Gs, hc_mask, alpha):
    """X − Gs·Wᵀ, W = Ridge(α).fit(Gs[HC], X[HC]). Gs must be pre-standardized;
    α is clamped away from 0 for sklearn's Ridge."""
    a = max(float(alpha), 1e-8)
    ridge = Ridge(alpha=a).fit(Gs[hc_mask], X[hc_mask])
    return X - Gs @ ridge.coef_.T


def hc_centroids(X, langs, groups, mask):
    """Per-language HC-only mean of X, restricted to `mask` rows."""
    return {L: (X[mask & (langs == L) & (groups == "HC")].mean(axis=0)
                if (mask & (langs == L) & (groups == "HC")).any() else None)
            for L in LANGS}


def apply_ls(X, langs, mus, target):
    """Shift each utterance by (target HC centroid − own-language HC centroid)."""
    if mus.get(target) is None:
        return X
    Y = X.copy()
    mu_t = mus[target]
    for L in LANGS:
        if mus.get(L) is None:
            continue
        sel = langs == L
        if sel.any():
            Y[sel] = Y[sel] - mus[L] + mu_t
    return Y
