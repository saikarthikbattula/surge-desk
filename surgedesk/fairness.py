"""Explicit SVI score adjustment and the measured top-k tradeoff."""
from __future__ import annotations

from dataclasses import replace

import pandas as pd

from . import config
from .contracts import Incident


def adjust(score: float, svi: float | None, lam: float) -> float:
    return round(min(1.0, score * (1 + lam * svi)), 6) if svi is not None else round(score, 6)


def apply(incidents: list[Incident], lam: float) -> list[Incident]:
    """Return independent adjusted records for one snapshot, ordered by rank_adj."""
    assert lam >= 0, "lambda must be nonnegative"
    assert len({(i.event, i.snapshot_time) for i in incidents}) <= 1, "Use one snapshot"
    def adjusted_score(i):
        value = adjust(i.score, i.svi, lam)
        if "share_current_assistance" in i.features and i.features["share_current_assistance"] == 0:
            value = min(value, config.CONTEXT_ADJUSTED_MAX)
        return value

    adjusted = [replace(i, score_adj=adjusted_score(i), lam=lam) for i in incidents]
    adjusted.sort(key=lambda i: (-i.score_adj, i.rank))
    return [replace(i, rank_adj=rank) for rank, i in enumerate(adjusted, 1)]


def fairness_table(incidents: list[Incident], lams: list[float] = config.LAMBDAS,
                   k: int = config.TOP_K_EVAL) -> pd.DataFrame:
    assert k > 0
    base = {i.incident_id for i in apply(incidents, 0.0)[:k]}
    rows = []
    for lam in lams:
        top = apply(incidents, lam)[:k]
        ids = {i.incident_id for i in top}
        n = len(top)
        precision = (sum(i.urgent_proxy for i in top) / n
                     if n and all(i.urgent_proxy is not None for i in top) else float("nan"))
        rows.append(dict(
            lam=lam, n_incidents=len(incidents), precision_at_k=precision,
            high_svi_share_top_k=sum(i.svi is not None and i.svi >= config.HIGH_SVI for i in top) / n if n else float("nan"),
            unknown_svi_share_top_k=sum(i.svi is None for i in top) / n if n else float("nan"),
            jaccard_vs_lam0=len(ids & base) / len(ids | base) if ids | base else 1.0,
        ))
    return pd.DataFrame(rows, columns=["lam", "n_incidents", "precision_at_k",
                                      "high_svi_share_top_k", "unknown_svi_share_top_k", "jaccard_vs_lam0"])
