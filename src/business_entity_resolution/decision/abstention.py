"""Decision policies over scored candidates (variable cardinality, explicit abstention).

All policies return a boolean keep-mask over the candidate pairs; an S1 entity whose
candidates are all rejected gets an empty prediction (abstention / no-match).

* ``global``   : keep p ≥ t
* ``source``   : keep p ≥ t_S2 / t_S3 depending on the candidate's source
* ``relative`` : keep p ≥ t and p ≥ α · max_p(entity)       (drops weak tail candidates)
* ``abstain``  : ``relative`` + reject the whole entity when max_p < τ (singleton protection)
* ``stage2``   : a second model on entity-context features (see ``context_matrix``)
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .threshold_optimizer import PreparedEval, evaluate_mask


def entity_stats(ent_idx: np.ndarray, p: np.ndarray, n_ent: int) -> dict[str, np.ndarray]:
    """Per-pair context: entity max / second max / count above thresholds / rank of p."""
    order = np.lexsort((-p, ent_idx))
    e_sorted, p_sorted = ent_idx[order], p[order]
    first = np.r_[True, e_sorted[1:] != e_sorted[:-1]]
    start = np.maximum.accumulate(np.where(first, np.arange(len(order)), 0))
    rank_sorted = np.arange(len(order)) - start
    rank = np.empty(len(p), dtype=np.int32)
    rank[order] = rank_sorted
    pmax = np.zeros(n_ent, dtype=np.float32)
    np.maximum.at(pmax, ent_idx, p)
    p2 = np.zeros(n_ent, dtype=np.float32)
    sec = rank == 1
    p2[ent_idx[sec]] = p[sec]
    psum = np.bincount(ent_idx, weights=p, minlength=n_ent).astype(np.float32)
    n50 = np.bincount(ent_idx, weights=(p >= 0.5), minlength=n_ent).astype(np.float32)
    n20 = np.bincount(ent_idx, weights=(p >= 0.2), minlength=n_ent).astype(np.float32)
    return {"p": p, "rank_p": rank.astype(np.float32), "p_max": pmax[ent_idx], "p_2nd": p2[ent_idx],
            "p_margin": (pmax - p2)[ent_idx], "p_rel": p / np.maximum(pmax[ent_idx], 1e-6),
            "p_sum": psum[ent_idx], "n_p50": n50[ent_idx], "n_p20": n20[ent_idx]}


def policy_mask(policy: dict, p: np.ndarray, ent_idx: np.ndarray, n_ent: int, is_s3: np.ndarray | None = None) -> np.ndarray:
    kind = policy["type"]
    if kind == "global":
        return p >= policy["t"]
    if kind == "source":
        return np.where(is_s3 > 0, p >= policy["t3"], p >= policy["t2"])
    st = entity_stats(ent_idx, p, n_ent)
    keep = (p >= policy["t"]) & (st["p_rel"] >= policy["alpha"])
    if kind == "abstain":
        keep &= st["p_max"] >= policy["tau"]
    return keep


def tune_policies(prep: PreparedEval, p: np.ndarray, is_s3: np.ndarray) -> dict[str, tuple[dict, dict]]:
    """Grid-search every rule-based policy for macro F0.5 on ``prep`` (a tuning set)."""
    n = len(prep.n_true)
    out = {}
    grid_t = np.round(np.arange(0.05, 0.991, 0.01), 3)
    best = max(((evaluate_mask(prep, p >= t)["f05"], t) for t in grid_t))
    out["global"] = ({"type": "global", "t": float(best[1])}, evaluate_mask(prep, p >= best[1]))
    t0 = best[1]
    near = np.round(np.arange(max(0.05, t0 - 0.2), min(0.99, t0 + 0.2) + 1e-9, 0.02), 3)
    b = max(((evaluate_mask(prep, np.where(is_s3 > 0, p >= t3, p >= t2))["f05"], t2, t3)
             for t2, t3 in itertools.product(near, near)))
    pol = {"type": "source", "t2": float(b[1]), "t3": float(b[2])}
    out["source"] = (pol, evaluate_mask(prep, policy_mask(pol, p, prep.ent_idx, n, is_s3)))
    st = entity_stats(prep.ent_idx, p, n)
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    b = max(((evaluate_mask(prep, (p >= t) & (st["p_rel"] >= a))["f05"], t, a) for t in near for a in alphas))
    pol = {"type": "relative", "t": float(b[1]), "alpha": float(b[2])}
    out["relative"] = (pol, evaluate_mask(prep, policy_mask(pol, p, prep.ent_idx, n)))
    taus = np.round(np.arange(0.3, 0.991, 0.03), 3)
    base = (p >= b[1]) & (st["p_rel"] >= b[2])
    bt = max(((evaluate_mask(prep, base & (st["p_max"] >= tau))["f05"], tau) for tau in taus))
    pol = {"type": "abstain", "t": float(b[1]), "alpha": float(b[2]), "tau": float(bt[1])}
    out["abstain"] = (pol, evaluate_mask(prep, policy_mask(pol, p, prep.ent_idx, n)))
    return out


CONTEXT_EXTRA = ["fielded_cos", "name_tset", "name_char_cos", "addr_tset", "core_idf_cov_a", "house_number_match",
                 "num_conflict", "b_is_s3", "b_addr_empty", "n_cands", "n_blocks", "name_generic_a"]


def context_matrix(ent_idx: np.ndarray, p: np.ndarray, n_ent: int, feats: pd.DataFrame) -> pd.DataFrame:
    st = entity_stats(ent_idx, p, n_ent)
    m = pd.DataFrame(st)
    for c in CONTEXT_EXTRA:
        m[c] = feats[c].to_numpy()
    return m
