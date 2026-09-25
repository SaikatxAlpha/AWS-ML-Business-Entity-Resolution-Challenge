"""Multi-block candidate generation: union of per-block top-k retrievals.

Each block is a TF-IDF representation in ``CountryIndex.R`` searched with exact sparse
top-k cosine. The union keeps, for every (S1, candidate) pair, each block's score and rank
(NaN / -1 when that block did not retrieve it) and the number of blocks that agree — these
become matcher features. The union is exactly the candidate set written to
candidate_pairs.tsv and scored by the model.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .index import CountryIndex

BLOCKING_VERSION = "mb1"
DEFAULT_BLOCKS = {"fielded": 30, "addr": 15, "name_char": 15, "name_word": 10}


def generate_candidates(ix: CountryIndex, q_rows: np.ndarray, blocks: dict[str, int] | None = None,
                        timings: dict | None = None) -> pd.DataFrame:
    blocks = blocks or DEFAULT_BLOCKS
    parts = {}
    for b, k in blocks.items():
        t = time.time()
        parts[b] = ix.retrieve(b, q_rows, k)
        ix.release_pool(b)
        if timings is not None:
            timings[b] = time.time() - t
    keys = [p["q"].to_numpy(np.int64) * ix.n + p["c"].to_numpy(np.int64) for p in parts.values()]
    uniq = pd.Index(np.unique(np.concatenate(keys))) if keys else pd.Index([], dtype=np.int64)
    out = pd.DataFrame({"q": (uniq.to_numpy() // ix.n).astype(np.int32), "c": (uniq.to_numpy() % ix.n).astype(np.int32)})
    n_blocks = np.zeros(len(out), dtype=np.int8)
    for (b, p), key in zip(parts.items(), keys):
        pos = uniq.get_indexer(key)
        rank = np.full(len(out), -1, dtype=np.int16)
        rank[pos] = p["rank"].to_numpy()
        out[f"rank_{b}"] = rank
        n_blocks[pos] += 1
    out["n_blocks"] = n_blocks
    return out


def block_membership(cands: pd.DataFrame, block: str, k: int | None = None) -> np.ndarray:
    r = cands[f"rank_{block}"].to_numpy()
    return (r >= 0) if k is None else ((r >= 0) & (r < k))
