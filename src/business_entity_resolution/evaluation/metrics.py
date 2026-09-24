"""Competition metric: F0.5 per Source-1 entity, macro-averaged over all evaluated entities.

Per entity with predicted set P and true set T:
  * T empty and P empty            -> 1.0
  * T empty and P non-empty        -> 0.0
  * T non-empty and |P ∩ T| = 0    -> 0.0   (includes P empty)
  * otherwise F0.5 = 1.25 p r / (0.25 p + r),  p = |P∩T|/|P|,  r = |P∩T|/|T|

Pairs are DataFrames with columns ``s1`` and ``cand`` (int64 codes from utils.ids, or
strings). Duplicate pairs are ignored. Entities are defined by the ``s1_ids`` argument, so
entities missing from the prediction count as empty predictions (never silently dropped).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BETA2 = 0.25


def _counts(pairs: pd.DataFrame, s1_index: pd.Index) -> pd.Series:
    return pairs.groupby("s1").size().reindex(s1_index, fill_value=0)


def entity_table(pred: pd.DataFrame, truth: pd.DataFrame, s1_ids) -> pd.DataFrame:
    """Per-entity counts and F0.5 for the entities in ``s1_ids``."""
    idx = pd.Index(pd.unique(np.asarray(s1_ids)), name="s1")
    pred = pred[["s1", "cand"]].drop_duplicates()
    truth = truth[["s1", "cand"]].drop_duplicates()
    pred = pred[pred["s1"].isin(idx)]
    truth = truth[truth["s1"].isin(idx)]
    tp = pred.merge(truth, on=["s1", "cand"])
    ent = pd.DataFrame({
        "n_pred": _counts(pred, idx),
        "n_true": _counts(truth, idx),
        "tp": _counts(tp, idx),
    })
    n_pred, n_true, tpv = (ent[c].to_numpy(dtype=float) for c in ("n_pred", "n_true", "tp"))
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(n_pred > 0, tpv / n_pred, 0.0)
        r = np.where(n_true > 0, tpv / n_true, 0.0)
        f = np.where(tpv > 0, (1 + BETA2) * p * r / (BETA2 * p + r), 0.0)
    f = np.where(n_true == 0, (n_pred == 0).astype(float), f)
    ent["precision"] = p
    ent["recall"] = r
    ent["f05"] = f
    return ent


def summarize(ent: pd.DataFrame) -> dict:
    single = ent["n_true"] == 0
    predicted = ent["n_pred"] > 0
    wrong = ent["n_pred"] > ent["tp"]
    return {
        "n_entities": int(len(ent)),
        "f05": float(ent["f05"].mean()),
        "f05_nonsingleton": float(ent.loc[~single, "f05"].mean()) if (~single).any() else float("nan"),
        "micro_precision": float(ent["tp"].sum() / max(1, ent["n_pred"].sum())),
        "micro_recall": float(ent["tp"].sum() / max(1, ent["n_true"].sum())),
        "macro_precision_predicted": float(ent.loc[predicted, "precision"].mean()) if predicted.any() else float("nan"),
        "macro_recall_nonsingleton": float(ent.loc[~single, "recall"].mean()) if (~single).any() else float("nan"),
        "singleton_rate": float(single.mean()),
        "singleton_accuracy": float((ent.loc[single, "n_pred"] == 0).mean()) if single.any() else float("nan"),
        "false_merge_rate": float(wrong[predicted].mean()) if predicted.any() else 0.0,
        "empty_prediction_rate": float((~predicted).mean()),
        "avg_predicted": float(ent["n_pred"].mean()),
    }


def evaluate(pred: pd.DataFrame, truth: pd.DataFrame, s1_ids, groups: pd.Series | None = None) -> dict:
    """Macro F0.5 and diagnostics. ``groups`` (indexed by s1) adds a per-group F0.5 breakdown."""
    ent = entity_table(pred, truth, s1_ids)
    out = summarize(ent)
    if groups is not None:
        g = groups.reindex(ent.index)
        out["f05_by_group"] = ent.groupby(g.to_numpy())["f05"].mean().round(5).to_dict()
    return out


def candidate_metrics(cand: pd.DataFrame, truth: pd.DataFrame, s1_ids) -> dict:
    """Blocking quality: pair recall, per-entity full coverage, candidate counts, oracle F0.5."""
    idx = pd.Index(pd.unique(np.asarray(s1_ids)), name="s1")
    cand = cand[["s1", "cand"]].drop_duplicates()
    cand = cand[cand["s1"].isin(idx)]
    truth = truth[["s1", "cand"]].drop_duplicates()
    truth = truth[truth["s1"].isin(idx)]
    hit = cand.merge(truth, on=["s1", "cand"])
    n_c = _counts(cand, idx)
    n_t = _counts(truth, idx)
    n_h = _counts(hit, idx)
    nonsingle = n_t > 0
    oracle = summarize(entity_table(hit, truth, idx))
    return {
        "candidate_recall": float(len(hit) / max(1, len(truth))),
        "entity_full_coverage": float((n_h[nonsingle] == n_t[nonsingle]).mean()),
        "entity_any_coverage": float((n_h[nonsingle] > 0).mean()),
        "avg_candidates": float(n_c.mean()),
        "p50_candidates": float(n_c.median()),
        "p95_candidates": float(n_c.quantile(0.95)),
        "max_candidates": int(n_c.max()) if len(n_c) else 0,
        "zero_candidate_rate": float((n_c == 0).mean()),
        "n_candidate_pairs": int(len(cand)),
        "oracle_f05": oracle["f05"],
    }


def f05_from_sets(pred: set, truth: set) -> float:
    """Reference scalar implementation (used by tests)."""
    if not truth:
        return 1.0 if not pred else 0.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(truth)
    return (1 + BETA2) * p * r / (BETA2 * p + r)
