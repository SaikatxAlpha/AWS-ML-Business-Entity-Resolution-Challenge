"""Per-country record index shared by blocking and pair-feature computation.

For one (split, country) it holds, for *all* records of that country (S1 ∪ S2 ∪ S3):

* binary token matrices (CSR): name words, address words, address numbers, address
  components (comma-separated parts, normalised);
* IDF vectors computed on the split's own records (label-free, so an unseen country gets
  its own statistics);
* a character 3-gram TF-IDF matrix of the normalised name;
* per-record attributes (norms, genericness counts, flags, integer-coded first number /
  first core token / initials / last address component).

Blocking representations (all L2-normalised, IDF-weighted, very frequent features pruned):
    ``fielded``  : core-name words + address words + numbers
    ``addr``     : address words + numbers only (finds matches with unrelated trade names)
    ``name_word``: all normalised name words
    ``name_char``: character 3-grams of the normalised name (typos, OCR, transliteration)
Word-based retrieval matrices are materialised only while a block is being searched; the
pairwise cosine for any representation is computed from the binary matrices with squared
IDF weights, which gives exactly the retrieval cosine at a fraction of the memory.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from ..preprocessing.normalization import to_ascii_series
from ..preprocessing.token_maps import apply_map

_DOMAIN = r"(?i)\.(?:com|net|org|in|co|fr|biz|info)\b|\bwww\b"
_ALIAS = r"(?i)\b(?:f/k/a|fka|d/b/a|dba|a/k/a|aka|formerly|t/a)\b"
_NON_ASCII = r"[^\x00-\x7f]"
STOPWORDS = ("the", "and", "of")


def _split(s: str) -> list[str]:
    return s.split()


def _split_bar(s: str) -> list[str]:
    return [t for t in s.split("|") if t]


def idf_of(binary: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    n = binary.shape[0]
    df = np.bincount(binary.indices, minlength=binary.shape[1])
    return (np.log((n + 1) / (df + 1)) + 1).astype(np.float32), df


def rowdot(a: sp.csr_matrix, ia: np.ndarray, b: sp.csr_matrix, ib: np.ndarray,
           w: np.ndarray | None = None, chunk: int = 1_000_000) -> np.ndarray:
    """Row-wise (optionally column-weighted) dot products: sum_j a[ia[k], j] * b[ib[k], j] * w[j]."""
    out = np.empty(len(ia), dtype=np.float32)
    for s in range(0, len(ia), chunk):
        e = min(len(ia), s + chunk)
        prod = a[ia[s:e]].multiply(b[ib[s:e]]).tocsr()
        out[s:e] = np.asarray(prod.sum(axis=1)).ravel() if w is None else prod @ w
    return out


def rowsum(a: sp.csr_matrix, w: np.ndarray) -> np.ndarray:
    return np.asarray(a @ w).ravel().astype(np.float32)


def _codes(values: pd.Series) -> np.ndarray:
    """Integer codes for equality tests; empty string -> -1."""
    codes, uniques = pd.factorize(values)
    codes = codes.astype(np.int32)
    empty = np.flatnonzero(np.asarray(uniques) == "")
    if len(empty):
        codes[codes == empty[0]] = -1
    return codes


@dataclass
class IndexConfig:
    max_df: dict = field(default_factory=lambda: {"fielded": 0.02, "addr": 0.02, "name_char": 0.01, "name_word": 0.02})
    char_ngram: tuple = (3, 3)


class CountryIndex:
    def __init__(self, rec: pd.DataFrame, weak_tokens: set[str], addr_map: dict[str, str],
                 cfg: IndexConfig | None = None, verbose: bool = True):
        t0 = time.time()
        self.cfg = cfg or IndexConfig()
        r = rec.reset_index(drop=True)
        self.n = len(r)
        self.source = r["source"].to_numpy()
        self.code = r["code"].to_numpy()
        self.row_of_code = pd.Index(self.code)

        # --- binary token matrices + IDF (label-free, on this split's records)
        def vec(col, analyzer=_split):
            v = CountVectorizer(analyzer=analyzer, lowercase=False, binary=True, dtype=np.float32)
            return v.fit_transform(col).tocsr(), v.vocabulary_

        names_l = r["name_norm"].tolist()          # plain lists: iterating Arrow arrays from Python is slow
        self.W_name, vocab_name = vec(names_l)
        self.idf_name, self.df_name = idf_of(self.W_name)
        self.core_mask = np.ones(self.W_name.shape[1], dtype=np.float32)
        self.core_mask[[j for t, j in vocab_name.items() if t in weak_tokens or t in STOPWORDS]] = 0.0
        self.W_addr, _ = vec(r["addr_norm"].tolist())
        self.idf_addr, self.df_addr = idf_of(self.W_addr)
        self.W_num, vocab_num = vec(r["nums"].tolist())
        self.idf_num, self.df_num = idf_of(self.W_num)
        self.postal_mask = np.array([1.0 if 5 <= len(t) <= 6 else 0.0 for t, _ in sorted(vocab_num.items(), key=lambda x: x[1])],
                                    dtype=np.float32)
        comps = self._components(r["addr_raw"], addr_map)
        self.W_comp, _ = vec(comps, analyzer=_split_bar)
        last_comp = pd.Series([c.rsplit("|", 1)[-1] if c else "" for c in comps])
        del comps

        # --- block weightings (IDF with frequent features pruned) and per-record norms
        md = self.cfg.max_df
        self.block_w = {
            "fielded": {"name": self._w(self.idf_name * self.core_mask, self.df_name, md["fielded"]),
                        "addr": self._w(self.idf_addr, self.df_addr, md["fielded"]),
                        "num": self._w(self.idf_num, self.df_num, md["fielded"])},
            "addr": {"addr": self._w(self.idf_addr, self.df_addr, md["addr"]),
                     "num": self._w(self.idf_num, self.df_num, md["addr"])},
            "name_word": {"name": self._w(self.idf_name, self.df_name, md["name_word"])},
        }
        self.block_norm = {b: np.sqrt(sum(rowsum(self._W(f), w * w) for f, w in parts.items())).astype(np.float32)
                           for b, parts in self.block_w.items()}
        cv = TfidfVectorizer(analyzer="char_wb", ngram_range=self.cfg.char_ngram, lowercase=False, sublinear_tf=True,
                             max_df=md["name_char"], min_df=2, dtype=np.float32)
        self.C_name = cv.fit_transform(names_l).tocsr()
        del names_l

        # --- per-record attributes
        self.name_norm = r["name_norm"]
        self.name_core = r["name_core"]
        self.addr_norm = r["addr_norm"]
        self.name_len = r["name_norm"].str.len().to_numpy(dtype=np.float32)
        self.name_ntok = np.diff(self.W_name.indptr).astype(np.float32)
        self.core_ntok = rowsum(self.W_name, self.core_mask)
        self.addr_ntok = np.diff(self.W_addr.indptr).astype(np.float32)
        self.num_cnt = np.diff(self.W_num.indptr).astype(np.float32)
        self.postal_cnt = rowsum(self.W_num, self.postal_mask)
        self.comp_cnt = np.diff(self.W_comp.indptr).astype(np.float32)
        self.name_idf_core_total = rowsum(self.W_name, self.idf_name * self.core_mask)
        self.addr_idf_total = rowsum(self.W_addr, self.idf_addr)
        first_num = r["nums"].str.split(" ", n=1).str[0].fillna("")
        self.first_num_code = _codes(first_num)
        self.first_num_col = np.array([vocab_num.get(t, -1) if t else -1 for t in first_num.tolist()], dtype=np.int64)
        self.first_core_code = _codes(r["name_core"].str.split(" ", n=1).str[0].fillna(""))
        initials = pd.Series(["".join(t[0] for t in c.split()) if len(c.split()) >= 2 else "" for c in r["name_core"].tolist()])
        both = _codes(pd.concat([initials, r["name_core"]], ignore_index=True))
        self.initials_code, self.core_code = both[:self.n], both[self.n:]
        self.last_comp_code = _codes(last_comp)
        nn, an = r["name_norm"], r["addr_norm"]
        self.name_generic = np.log1p(nn.map(nn.value_counts()).to_numpy(dtype=np.float32))
        self.addr_generic = np.log1p(an.map(an.value_counts()).to_numpy(dtype=np.float32))
        self.addr_empty = (an.str.len() == 0).to_numpy(dtype=np.float32)
        self.addr_generic[self.addr_empty > 0] = 0.0
        raw = r["name_raw"]
        self.name_nonascii = raw.str.contains(_NON_ASCII, regex=True).to_numpy(dtype=np.float32)
        self.name_domain = raw.str.contains(_DOMAIN, regex=True).to_numpy(dtype=np.float32)
        self.name_alias = raw.str.contains(_ALIAS, regex=True).to_numpy(dtype=np.float32)
        self.addr_nonascii = r["addr_raw"].str.contains(_NON_ASCII, regex=True).to_numpy(dtype=np.float32)
        if verbose:
            print(f"[index] {self.n:,} records; nnz name={self.W_name.nnz:,} addr={self.W_addr.nnz:,} "
                  f"num={self.W_num.nnz:,} comp={self.W_comp.nnz:,} char={self.C_name.nnz:,}; {time.time() - t0:.0f}s",
                  flush=True)

    # ------------------------------------------------------------------ helpers
    def _w(self, weights: np.ndarray, df: np.ndarray, max_df: float) -> np.ndarray:
        return np.where(df <= max_df * self.n, weights, 0.0).astype(np.float32)

    def _W(self, fieldname: str) -> sp.csr_matrix:
        return {"name": self.W_name, "addr": self.W_addr, "num": self.W_num}[fieldname]

    @staticmethod
    def _components(addr_raw: pd.Series, addr_map: dict) -> list[str]:
        s = to_ascii_series(addr_raw).str.lower().str.replace(r"[^0-9a-z,]+", " ", regex=True).tolist()
        out = []
        for a in s:
            parts = []
            for c in a.split(","):
                toks = [(t.lstrip("0") or "0") if t.isdigit() else t for t in c.split() if t != "null"]
                if toks:
                    parts.append(apply_map(" ".join(toks), addr_map))
            out.append("|".join(parts))
        return out

    def text(self, col: str, idx: np.ndarray) -> list:
        return getattr(self, col).take(idx).tolist()

    def cosine(self, block: str, q: np.ndarray, c: np.ndarray) -> np.ndarray:
        """Exact cosine of the block's IDF-weighted representation for each (q, c) pair."""
        if block == "name_char":
            return rowdot(self.C_name, q, self.C_name, c)
        dot = sum(rowdot(self._W(f), q, self._W(f), c, w * w) for f, w in self.block_w[block].items())
        den = self.block_norm[block][q] * self.block_norm[block][c]
        return np.where(den > 0, dot / np.maximum(den, 1e-12), 0.0).astype(np.float32)

    def retrieval_matrix(self, block: str) -> sp.csr_matrix:
        if block == "name_char":
            return self.C_name
        parts = [sp.csr_matrix(self._W(f).multiply(w.reshape(1, -1)), dtype=np.float32)
                 for f, w in self.block_w[block].items()]
        m = sp.hstack(parts, format="csr") if len(parts) > 1 else parts[0]
        m.eliminate_zeros()
        inv = np.where(self.block_norm[block] > 0, 1.0 / np.maximum(self.block_norm[block], 1e-12), 0.0).astype(np.float32)
        return sp.csr_matrix(sp.diags(inv) @ m, dtype=np.float32)

    def rowmax_shared(self, m: sp.csr_matrix, q: np.ndarray, c: np.ndarray, w: np.ndarray,
                      chunk: int = 1_000_000) -> np.ndarray:
        """Max column weight among tokens shared by m[q[k]] and m[c[k]] (0 if none)."""
        out = np.zeros(len(q), dtype=np.float32)
        wd = sp.diags(w.astype(np.float32))
        for s in range(0, len(q), chunk):
            e = min(len(q), s + chunk)
            prod = m[q[s:e]].multiply(m[c[s:e]]).tocsr() @ wd
            out[s:e] = prod.max(axis=1).toarray().ravel()
        return out

    def first_num_in(self, q: np.ndarray, c: np.ndarray) -> np.ndarray:
        """1 if the first address number of q appears among c's address numbers."""
        cols = self.first_num_col[q]
        out = np.zeros(len(q), dtype=np.float32)
        sel = np.flatnonzero(cols >= 0)
        for s in range(0, len(sel), 2_000_000):
            idx = sel[s:s + 2_000_000]
            out[idx] = np.asarray(self.W_num[c[idx], cols[idx]]).ravel()
        return out

    # ------------------------------------------------------------------ retrieval
    def pool_rows(self) -> np.ndarray:
        return np.flatnonzero(self.source != 1)

    def retrieve(self, block: str, q_rows: np.ndarray, k: int, min_score: float = 0.0,
                 chunk: int = 100_000) -> pd.DataFrame:
        pool = self.pool_rows()
        m = self.retrieval_matrix(block)
        pt = m[pool].T.tocsr()
        mq = m[q_rows]
        del m
        out = []
        for s in range(0, len(q_rows), chunk):
            c = sp_matmul_topn(mq[s:s + chunk], pt, top_n=k, threshold=min_score, sort=True, n_threads=-1).tocsr()
            counts = np.diff(c.indptr)
            rows = np.repeat(np.arange(c.shape[0]), counts)
            rank = np.arange(c.nnz) - np.repeat(c.indptr[:-1], counts)
            out.append(pd.DataFrame({"q": q_rows[s:s + chunk][rows].astype(np.int32), "c": pool[c.indices].astype(np.int32),
                                     "score": c.data.astype(np.float32), "rank": rank.astype(np.int16)}))
        return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["q", "c", "score", "rank"])

    def release_pool(self, block: str | None = None) -> None:
        """Kept for API compatibility: retrieval matrices are no longer cached."""
