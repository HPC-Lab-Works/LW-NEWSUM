"""Paired bootstrap significance testing with Holm-Bonferroni correction.

For a proposed system A and a comparison system B evaluated on the same N test
documents, d_i = s_i(A) - s_i(B). We draw ``B`` bootstrap resamples of the
paired differences and report the observed mean difference, the 95% percentile
interval and a two-sided p-value p = min(1, 2 * (k + 1) / (B + 1)), where k is
the number of resamples whose mean difference has the opposite sign to (or is
zero relative to) the observed difference. Seeded systems are first averaged
per document across seeds, so the test reflects the seed-averaged system.
"""
from __future__ import annotations

import numpy as np


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 10000, seed: int = 42) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape
    d = a - b
    obs = float(d.mean())
    rng = np.random.default_rng(seed)
    n = len(d)
    means = np.empty(n_boot)
    for i in range(0, n_boot, 1000):  # chunked to bound memory
        m = min(1000, n_boot - i)
        idx = rng.integers(0, n, size=(m, n))
        means[i:i + m] = d[idx].mean(axis=1)
    k = int((means <= 0).sum()) if obs > 0 else int((means >= 0).sum())
    p = min(1.0, 2 * (k + 1) / (n_boot + 1))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"diff": obs, "ci_low": float(lo), "ci_high": float(hi), "p": p, "n": n}


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (monotone)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def mean_std(values: list[float]) -> tuple[float, float]:
    v = np.asarray(values, float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0
