"""Address similarity features (word, component and TF-IDF level)."""
from __future__ import annotations

import numpy as np
from rapidfuzz import fuzz, process

from ..blocking.index import CountryIndex, rowdot
from .name_features import _cp, _safe_div


def address_features(ix: CountryIndex, q: np.ndarray, c: np.ndarray) -> dict[str, np.ndarray]:
    aa, ba = np.array(ix.text("addr_norm", q), dtype=object), np.array(ix.text("addr_norm", c), dtype=object)
    na, nb = ix.addr_ntok[q], ix.addr_ntok[c]
    inter = rowdot(ix.W_addr, q, ix.W_addr, c)
    shared_idf = rowdot(ix.W_addr, q, ix.W_addr, c, ix.idf_addr)
    comp_inter = rowdot(ix.W_comp, q, ix.W_comp, c)
    ka, kb = ix.comp_cnt[q], ix.comp_cnt[c]
    b_empty = ix.addr_empty[c]
    f = {
        "addr_exact": ((aa == ba) & (b_empty == 0)).astype(np.float32),
        "addr_ratio": _cp(aa, ba, fuzz.ratio),
        "addr_tset": _cp(aa, ba, fuzz.token_set_ratio),
        "addr_tsort": _cp(aa, ba, fuzz.token_sort_ratio),
        "addr_common": inter,
        "addr_jacc": _safe_div(inter, na + nb - inter),
        "addr_idf_cov_a": _safe_div(shared_idf, ix.addr_idf_total[q]),
        "addr_idf_cov_b": _safe_div(shared_idf, ix.addr_idf_total[c]),
        "addr_cos": ix.cosine("addr", q, c),
        "fielded_cos": ix.cosine("fielded", q, c),
        "comp_common": comp_inter,
        "comp_cov_a": _safe_div(comp_inter, ka),
        "comp_jacc": _safe_div(comp_inter, ka + kb - comp_inter),
        "last_comp_eq": ((ix.last_comp_code[q] == ix.last_comp_code[c]) & (ix.last_comp_code[q] >= 0)).astype(np.float32),
        "addr_ntok_a": na, "addr_ntok_b": nb,
        "b_addr_empty": b_empty,
        "b_addr_nonascii": ix.addr_nonascii[c],
        "addr_generic_a": ix.addr_generic[q], "addr_generic_b": ix.addr_generic[c],
    }
    return f
