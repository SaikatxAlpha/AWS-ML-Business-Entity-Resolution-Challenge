"""Numeric address evidence: house/building numbers, postal/PIN codes, number conflicts."""
from __future__ import annotations

import numpy as np

from ..blocking.index import CountryIndex, rowdot
from .name_features import _safe_div


def numeric_features(ix: CountryIndex, q: np.ndarray, c: np.ndarray) -> dict[str, np.ndarray]:
    na, nb = ix.num_cnt[q], ix.num_cnt[c]
    inter = rowdot(ix.W_num, q, ix.W_num, c)
    shared_idf = rowdot(ix.W_num, q, ix.W_num, c, ix.idf_num)
    postal_inter = rowdot(ix.W_num, q, ix.W_num, c, ix.postal_mask)
    pa, pb = ix.postal_cnt[q], ix.postal_cnt[c]
    both = (na > 0) & (nb > 0)
    postal_both = (pa > 0) & (pb > 0)
    first_in_b = ix.first_num_in(q, c)
    return {
        "num_common": inter,
        "num_jacc": _safe_div(inter, na + nb - inter),
        "num_cov_a": _safe_div(inter, na),
        "num_idf_shared": shared_idf,
        "num_conflict": (both & (inter == 0)).astype(np.float32),
        "num_cnt_a": na, "num_cnt_b": nb,
        "num_cnt_diff": np.abs(na - nb).astype(np.float32),
        "house_number_match": first_in_b,
        "first_num_eq": ((ix.first_num_code[q] == ix.first_num_code[c]) & (ix.first_num_code[q] >= 0)).astype(np.float32),
        "pin_both": postal_both.astype(np.float32),
        "pin_exact_match": (postal_inter > 0).astype(np.float32),
        "pin_conflict": (postal_both & (postal_inter == 0)).astype(np.float32),
    }
