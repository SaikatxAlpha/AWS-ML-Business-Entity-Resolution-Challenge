"""Transparent, non-learned baseline matcher.

score = w · name_sim + (1 − w) · addr_sim, with rapidfuzz token-set similarity (0–100);
when either address is empty only the name similarity is used. A candidate is accepted
when score ≥ t. (w, t) are chosen by grid search for macro F0.5 on training-fold entities
and then applied unchanged to validation entities.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from ..decision.threshold_optimizer import PreparedEval, best, grid_search


def pair_texts(cands: pd.DataFrame, rec: pd.DataFrame, cols: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    r = rec.set_index("code")[cols]
    a = r.loc[cands["s1"].to_numpy()]
    b = r.loc[cands["cand"].to_numpy()]
    return {c: (a[c].to_numpy(), b[c].to_numpy()) for c in cols}


def fuzzy_scores(cands: pd.DataFrame, rec: pd.DataFrame, name_col: str, addr_col: str) -> pd.DataFrame:
    tx = pair_texts(cands, rec, [name_col, addr_col])
    (na, nb), (aa, ab) = tx[name_col], tx[addr_col]
    name_sim = process.cpdist(na, nb, scorer=fuzz.token_set_ratio, workers=-1).astype(np.float32)
    addr_sim = process.cpdist(aa, ab, scorer=fuzz.token_set_ratio, workers=-1).astype(np.float32)
    empty = (pd.Series(aa).str.len().to_numpy() == 0) | (pd.Series(ab).str.len().to_numpy() == 0)
    addr_sim[empty] = np.nan
    return cands.assign(name_sim=name_sim, addr_sim=addr_sim)


def combined(scored: pd.DataFrame, w: float) -> np.ndarray:
    n = scored["name_sim"].to_numpy()
    a = scored["addr_sim"].to_numpy()
    return np.where(np.isnan(a), n, w * n + (1 - w) * a)


def tune(scored: pd.DataFrame, prep: PreparedEval, weights=(0.3, 0.4, 0.5, 0.6, 0.7),
         thresholds=np.arange(50, 100.1, 1.0)) -> tuple[dict, pd.DataFrame]:
    curves = []
    for w in weights:
        c = grid_search(prep, combined(scored, w), thresholds)
        c["w"] = w
        curves.append(c)
    curves = pd.concat(curves, ignore_index=True)
    b = best(curves)
    return {"w": float(b["w"]), "t": float(b["threshold"])}, curves
