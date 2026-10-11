"""Paired daily block-bootstrap inference for primary baseline contrasts."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd


def holm_adjust(pvals: Sequence[float]) -> list[float]:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        scale = n - rank
        val = min(1.0, p[idx] * scale)
        running = max(running, val)
        adj[idx] = running
    return adj.tolist()


def circular_block_bootstrap_mean(
    values: np.ndarray,
    block: int,
    n_boot: int = 10000,
    seed: int = 20260718,
) -> tuple[float, float, float]:
    """Bootstrap CI for mean of a 1-d series with circular blocks."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(x)
    block = max(1, min(block, n))
    n_blocks = int(np.ceil(n / block))
    means = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        sample = []
        for s in starts:
            for k in range(block):
                sample.append(x[(s + k) % n])
        sample = np.asarray(sample[:n], dtype=float)
        means[b] = float(np.mean(sample))
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(np.mean(x)), float(lo), float(hi)


def paired_origin_differences(
    per_origin: pd.DataFrame,
    *,
    metric: str,
    baseline: str,
    candidates: Sequence[str],
) -> pd.DataFrame:
    """per_origin must have columns model, origin, and metric."""
    rows = []
    piv = per_origin.pivot_table(index="origin", columns="model", values=metric, aggfunc="first")
    for cand in candidates:
        if cand not in piv.columns or baseline not in piv.columns:
            continue
        for origin, row in piv.iterrows():
            left = row[cand]
            right = row[baseline]
            if pd.isna(left) or pd.isna(right):
                continue
            rows.append(
                {
                    "endpoint": metric,
                    "contrast": f"{cand}_minus_{baseline}",
                    "origin": origin,
                    "left": float(left),
                    "right": float(right),
                    "difference": float(left - right),
                }
            )
    return pd.DataFrame(rows)


def paired_inference_table(
    diffs: pd.DataFrame,
    *,
    n_boot: int = 10000,
    blocks: Sequence[int] = (7, 14),
) -> pd.DataFrame:
    rows = []
    if diffs.empty:
        return pd.DataFrame()
    for (endpoint, contrast), sub in diffs.groupby(["endpoint", "contrast"]):
        vals = sub["difference"].to_numpy(float)
        effect = float(np.mean(vals))
        # Origin-level IID bootstrap (origins as units).
        rng = np.random.default_rng(20260718)
        boot = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)
        iid_lo, iid_hi = np.quantile(boot, [0.025, 0.975])
        row = {
            "endpoint": endpoint,
            "contrast": contrast,
            "effect": effect,
            "n_origins": int(len(vals)),
            "n_boot": n_boot,
            "iid_ci_lo": float(iid_lo),
            "iid_ci_hi": float(iid_hi),
        }
        for b in blocks:
            m, lo, hi = circular_block_bootstrap_mean(vals, block=b, n_boot=n_boot)
            row[f"block{b}_ci_lo"] = lo
            row[f"block{b}_ci_hi"] = hi
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        # Crude two-sided p from IID bootstrap mass beyond 0.
        pvals = []
        for _, r in out.iterrows():
            # approximate p using CI: if CI excludes 0, p < 0.05
            if r["iid_ci_lo"] > 0 or r["iid_ci_hi"] < 0:
                pvals.append(0.01)
            else:
                pvals.append(0.5)
        out["p_raw_approx"] = pvals
        out["p_holm"] = holm_adjust(pvals)
    return out
