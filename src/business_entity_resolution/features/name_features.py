"""Name similarity features (all computed on arrays of candidate pairs)."""
from __future__ import annotations

import numpy as np
from rapidfuzz import fuzz, process
from rapidfuzz.distance import DamerauLevenshtein, JaroWinkler

from ..blocking.index import CountryIndex, rowdot


def _safe_div(a, b):
    return np.where(b > 0, a / np.maximum(b, 1e-9), 0.0).astype(np.float32)


def _cp(a, b, scorer):
    return (process.cpdist(a, b, scorer=scorer, workers=-1) / 100.0).astype(np.float32)


def name_features(ix: CountryIndex, q: np.ndarray, c: np.ndarray) -> dict[str, np.ndarray]:
    an, bn = np.array(ix.text("name_norm", q), dtype=object), np.array(ix.text("name_norm", c), dtype=object)
    ac, bc = np.array(ix.text("name_core", q), dtype=object), np.array(ix.text("name_core", c), dtype=object)
    na, nb = ix.name_ntok[q], ix.name_ntok[c]
    ca, cb = ix.core_ntok[q], ix.core_ntok[c]
    inter = rowdot(ix.W_name, q, ix.W_name, c)
    core_inter = rowdot(ix.W_name, q, ix.W_name, c, ix.core_mask)
    core_w = ix.idf_name * ix.core_mask
    shared_idf = rowdot(ix.W_name, q, ix.W_name, c, core_w)
    tot_a, tot_b = ix.name_idf_core_total[q], ix.name_idf_core_total[c]
    f = {
        "name_exact": (an == bn).astype(np.float32),
        "core_exact": (ac == bc).astype(np.float32),
        "name_ratio": _cp(an, bn, fuzz.ratio),
        "core_ratio": _cp(ac, bc, fuzz.ratio),
        "name_jw": process.cpdist(an, bn, scorer=JaroWinkler.normalized_similarity, workers=-1).astype(np.float32),
        "core_dl": process.cpdist(ac, bc, scorer=DamerauLevenshtein.normalized_similarity, workers=-1).astype(np.float32),
        "name_tsort": _cp(an, bn, fuzz.token_sort_ratio),
        "name_tset": _cp(an, bn, fuzz.token_set_ratio),
        "core_partial": _cp(ac, bc, fuzz.partial_ratio),
        "name_common": inter,
        "name_jacc": _safe_div(inter, na + nb - inter),
        "name_dice": _safe_div(2 * inter, na + nb),
        "name_overlap": _safe_div(inter, np.minimum(na, nb)),
        "core_common": core_inter,
        "core_jacc": _safe_div(core_inter, ca + cb - core_inter),
        "core_idf_cov_a": _safe_div(shared_idf, tot_a),
        "core_idf_cov_b": _safe_div(shared_idf, tot_b),
        "core_idf_shared": shared_idf,
        "core_max_shared_idf": ix.rowmax_shared(ix.W_name, q, c, core_w),
        "name_word_cos": ix.cosine("name_word", q, c),
        "name_char_cos": ix.cosine("name_char", q, c),
        "name_len_ratio": _safe_div(np.minimum(ix.name_len[q], ix.name_len[c]), np.maximum(ix.name_len[q], ix.name_len[c])),
        "name_ntok_diff": np.abs(na - nb).astype(np.float32),
        "core_ntok_a": ca, "core_ntok_b": cb,
        "first_core_eq": ((ix.first_core_code[q] == ix.first_core_code[c]) & (ix.first_core_code[q] >= 0)).astype(np.float32),
        "acronym_match": (((ix.initials_code[q] == ix.core_code[c]) & (ix.initials_code[q] >= 0)) |
                          ((ix.initials_code[c] == ix.core_code[q]) & (ix.initials_code[c] >= 0))).astype(np.float32),
        "name_generic_a": ix.name_generic[q], "name_generic_b": ix.name_generic[c],
        "b_name_nonascii": ix.name_nonascii[c], "b_name_domain": ix.name_domain[c], "b_name_alias": ix.name_alias[c],
    }
    return f
