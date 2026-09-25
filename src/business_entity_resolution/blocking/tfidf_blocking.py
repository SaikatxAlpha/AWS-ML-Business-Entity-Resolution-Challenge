"""Fielded TF-IDF nearest-neighbour retrieval.

Each record becomes a bag of prefixed tokens, e.g. ``n:apex n:healthcare a:andheri
a:west d:102``. IDF is fitted per country on the split's own records (S1 ∪ S2 ∪ S3; no
labels), so an unseen country gets its own statistics automatically. Very frequent
features (document frequency above ``max_df``) are dropped: they carry almost no
evidence and dominate the cost of the sparse product. Retrieval is an exact sparse top-k
cosine search (``sparse_dot_topn``) restricted to the same country value.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn


@dataclass
class TfidfBlockingConfig:
    fields: dict = field(default_factory=lambda: {"n": "name_core", "a": "addr_norm", "d": "nums"})
    weights: dict = field(default_factory=lambda: {"n": 1.0, "a": 1.0, "d": 1.0})
    top_k: int = 50
    max_df: float = 0.02
    min_df: int = 2
    min_score: float = 0.0
    chunk: int = 50_000
    n_threads: int = -1


def _split(s: str) -> list[str]:
    return s.split()


def build_docs(df: pd.DataFrame, cfg: TfidfBlockingConfig) -> list[str]:
    cols = [(p, df[c].to_numpy()) for p, c in cfg.fields.items()]
    return [" ".join(f"{p}:{t}" for p, vals in cols for t in vals[i].split()) for i in range(len(df))]


def _field_scaling(vocab: dict, cfg: TfidfBlockingConfig) -> sp.dia_matrix | None:
    if all(w == 1.0 for w in cfg.weights.values()):
        return None
    w = np.ones(len(vocab), dtype=np.float32)
    for tok, j in vocab.items():
        w[j] = cfg.weights.get(tok.split(":", 1)[0], 1.0)
    return sp.diags(w)


def _l2(m: sp.csr_matrix) -> sp.csr_matrix:
    n = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
    n[n == 0] = 1.0
    return sp.diags(1.0 / n).dot(m).tocsr()


def tfidf_candidates(rec: pd.DataFrame, query_codes, cfg: TfidfBlockingConfig, verbose: bool = True) -> pd.DataFrame:
    """Top-k S2/S3 candidates for each queried S1 record. Returns s1, cand, score, rank."""
    query_codes = pd.Index(np.asarray(query_codes, dtype=np.int64))
    out = []
    for country, idx in rec.groupby("country").indices.items():
        part = rec.iloc[idx]
        is_s1 = part["source"].to_numpy() == 1
        q = part[is_s1 & part["code"].isin(query_codes).to_numpy()]
        if q.empty:
            continue
        pool = part[~is_s1]
        t = time.time()
        vec = TfidfVectorizer(analyzer=_split, lowercase=False, sublinear_tf=True, max_df=cfg.max_df,
                              min_df=cfg.min_df, dtype=np.float32)
        # one pass: vectorise every record of the country once, then slice queries / pool
        x_all = vec.fit_transform(build_docs(part, cfg)).tocsr()
        scale = _field_scaling(vec.vocabulary_, cfg)
        q_mask = is_s1 & part["code"].isin(query_codes).to_numpy()
        xp = x_all[np.flatnonzero(~is_s1)]
        xq = x_all[np.flatnonzero(q_mask)]
        del x_all
        if scale is not None:
            xp, xq = _l2(xp @ scale), _l2(xq @ scale)
        xpt = xp.T.tocsr()
        pool_codes = pool["code"].to_numpy()
        q_codes = q["code"].to_numpy()
        for start in range(0, xq.shape[0], cfg.chunk):
            c = sp_matmul_topn(xq[start:start + cfg.chunk], xpt, top_n=cfg.top_k, threshold=cfg.min_score,
                               sort=True, n_threads=cfg.n_threads).tocsr()
            counts = np.diff(c.indptr)
            rows = np.repeat(np.arange(c.shape[0]), counts)
            rank = np.arange(c.nnz) - np.repeat(c.indptr[:-1], counts)
            out.append(pd.DataFrame({
                "s1": q_codes[start + rows], "cand": pool_codes[c.indices],
                "score": c.data.astype(np.float32), "rank": rank.astype(np.int16),
            }))
        if verbose:
            print(f"[tfidf] {country}: {len(q):,} queries x {len(pool):,} pool, vocab {len(vec.vocabulary_):,}, "
                  f"{time.time() - t:.0f}s", flush=True)
    if not out:
        return pd.DataFrame({"s1": pd.Series(dtype=np.int64), "cand": pd.Series(dtype=np.int64),
                             "score": pd.Series(dtype=np.float32), "rank": pd.Series(dtype=np.int16)})
    return pd.concat(out, ignore_index=True)
