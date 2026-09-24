"""Fast macro-F0.5 evaluation of many decision rules over one scored candidate set.

``PreparedEval`` maps every candidate pair to its entity index and truth label once;
evaluating a keep-mask is then two ``bincount`` calls, so grids of hundreds of thresholds
are cheap. Results are identical to ``evaluation.metrics.evaluate`` restricted to the
candidate set (true matches outside the candidate set still count as misses because
``n_true`` comes from the full ground truth).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

BETA2 = 0.25


@dataclass
class PreparedEval:
    ent_idx: np.ndarray      # entity index per candidate pair
    is_true: np.ndarray      # bool per candidate pair
    n_true: np.ndarray       # true matches per entity (full ground truth)
    s1: np.ndarray           # entity codes, aligned with n_true


def prepare(cands: pd.DataFrame, truth: pd.DataFrame, s1_ids) -> PreparedEval:
    s1 = np.asarray(pd.unique(np.asarray(s1_ids, dtype=np.int64)))
    pos = pd.Index(s1)
    t = truth[truth["s1"].isin(pos)]
    n_true = np.bincount(pos.get_indexer(t["s1"]), minlength=len(s1))
    c = cands[cands["s1"].isin(pos)]
    ent_idx = pos.get_indexer(c["s1"])
    key = pd.MultiIndex.from_frame(c[["s1", "cand"]])
    tkey = pd.MultiIndex.from_frame(t[["s1", "cand"]])
    is_true = key.isin(tkey)
    return PreparedEval(ent_idx=ent_idx, is_true=np.asarray(is_true), n_true=n_true, s1=s1)


def per_entity_f05(n_pred: np.ndarray, tp: np.ndarray, n_true: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(n_pred > 0, tp / np.maximum(n_pred, 1), 0.0)
        r = np.where(n_true > 0, tp / np.maximum(n_true, 1), 0.0)
        f = np.where(tp > 0, (1 + BETA2) * p * r / (BETA2 * p + r), 0.0)
    return np.where(n_true == 0, (n_pred == 0).astype(float), f)


def evaluate_mask(prep: PreparedEval, keep: np.ndarray) -> dict:
    n = len(prep.n_true)
    n_pred = np.bincount(prep.ent_idx[keep], minlength=n)
    tp = np.bincount(prep.ent_idx[keep & prep.is_true], minlength=n)
    f = per_entity_f05(n_pred, tp, prep.n_true)
    single = prep.n_true == 0
    predicted = n_pred > 0
    return {
        "f05": float(f.mean()),
        "micro_precision": float(tp.sum() / max(1, n_pred.sum())),
        "micro_recall": float(tp.sum() / max(1, prep.n_true.sum())),
        "singleton_accuracy": float((n_pred[single] == 0).mean()) if single.any() else float("nan"),
        "false_merge_rate": float((n_pred > tp)[predicted].mean()) if predicted.any() else 0.0,
        "avg_predicted": float(n_pred.mean()),
    }


def grid_search(prep: PreparedEval, scores: np.ndarray, thresholds) -> pd.DataFrame:
    rows = []
    for t in thresholds:
        m = evaluate_mask(prep, scores >= t)
        m["threshold"] = float(t)
        rows.append(m)
    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def best(curve: pd.DataFrame) -> pd.Series:
    return curve.loc[curve["f05"].idxmax()]
