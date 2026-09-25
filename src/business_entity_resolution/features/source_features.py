"""Source, blocking-provenance and candidate-list context features."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..blocking.index import CountryIndex


def source_block_features(ix: CountryIndex, cands: pd.DataFrame, blocks: list[str]) -> dict[str, np.ndarray]:
    f = {"b_is_s3": (ix.source[cands["c"].to_numpy()] == 3).astype(np.float32),
         "n_blocks": cands["n_blocks"].to_numpy(np.float32)}
    for b in blocks:
        r = cands[f"rank_{b}"].to_numpy()
        f[f"rank_{b}"] = np.where(r >= 0, r, 99).astype(np.float32)
    return f


def context_features(q: np.ndarray, feats: pd.DataFrame, cols=("fielded_cos", "name_char_cos", "addr_cos", "name_tset")) -> dict[str, np.ndarray]:
    """Position of each candidate relative to the other candidates of the same S1 entity."""
    g = pd.Series(q)
    out = {"n_cands": g.map(g.value_counts()).to_numpy(np.float32)}
    for col in cols:
        v = pd.Series(feats[col].to_numpy(), index=g.index)
        mx = v.groupby(g.to_numpy()).transform("max").to_numpy()
        out[f"{col}_gap"] = (mx - v.to_numpy()).astype(np.float32)
        out[f"{col}_rank"] = v.groupby(g.to_numpy()).rank(ascending=False, method="min").to_numpy(np.float32)
    return out


def top_candidate_features(ix: CountryIndex, q: np.ndarray, c: np.ndarray, score: np.ndarray) -> dict[str, np.ndarray]:
    """Consistency with the entity's strongest candidate (by ``score``): S2/S3–S2/S3 graph edges.

    Records of one business are often duplicated inside S2/S3, so a candidate that closely
    resembles the best candidate is more likely a true match (and vice versa).
    """
    from ..blocking.index import rowdot

    order = np.lexsort((-score, q))
    first = np.r_[True, q[order][1:] != q[order][:-1]]
    top_of_q = pd.Series(c[order][first], index=q[order][first])
    t = top_of_q.reindex(q).to_numpy()

    def jacc(W, cnt):
        inter = rowdot(W, c, W, t)
        den = cnt[c] + cnt[t] - inter
        return np.where(den > 0, inter / np.maximum(den, 1e-9), 0.0).astype(np.float32)

    return {
        "is_top": (c == t).astype(np.float32),
        "top_name_jacc": jacc(ix.W_name, ix.name_ntok),
        "top_addr_jacc": jacc(ix.W_addr, ix.addr_ntok),
        "top_num_common": rowdot(ix.W_num, c, ix.W_num, t),
        "top_char_cos": rowdot(ix.C_name, c, ix.C_name, t),
    }
