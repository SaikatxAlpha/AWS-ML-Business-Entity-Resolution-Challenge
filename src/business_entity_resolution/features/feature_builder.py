"""Assemble the pairwise feature matrix for a candidate set of one country index."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ..blocking.index import CountryIndex
from .address_features import address_features
from .name_features import name_features
from .numeric_features import numeric_features
from .source_features import context_features, source_block_features, top_candidate_features

FEATURE_VERSION = "f1"


def build_features(ix: CountryIndex, cands: pd.DataFrame, blocks: list[str], chunk: int = 2_000_000,
                   verbose: bool = False) -> pd.DataFrame:
    """Return ``cands`` (q, c, rank_*, n_blocks) + s1/cand codes + all pair features (float32)."""
    t0 = time.time()
    parts = []
    for s in range(0, len(cands), chunk):
        cc = cands.iloc[s:s + chunk]
        q, c = cc["q"].to_numpy(), cc["c"].to_numpy()
        f = {}
        f.update(name_features(ix, q, c))
        f.update(address_features(ix, q, c))
        f.update(numeric_features(ix, q, c))
        f.update(source_block_features(ix, cc, blocks))
        parts.append(pd.DataFrame(f))
    feats = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    q_all = cands["q"].to_numpy()
    c_all = cands["c"].to_numpy()
    extra = context_features(q_all, feats)
    extra.update(top_candidate_features(ix, q_all, c_all, feats["fielded_cos"].to_numpy()))
    feats = pd.concat([feats, pd.DataFrame(extra)], axis=1)
    out = pd.concat([pd.DataFrame({"s1": ix.code[q_all], "cand": ix.code[cands["c"].to_numpy()]}), feats], axis=1)
    if verbose:
        print(f"[features] {len(out):,} pairs x {feats.shape[1]} features in {time.time() - t0:.0f}s", flush=True)
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("s1", "cand", "y", "set", "country")]
