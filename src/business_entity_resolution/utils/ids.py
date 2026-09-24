"""Compact integer encoding of entity IDs.

``S<k>-<n>`` -> ``k * 10**10 + n`` (int64). All IDs in the data have a numeric body well
below 10**10 (checked at encode time), so the mapping is lossless and joins on int64 are
much cheaper than joins on strings.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_BASE = 10**10


def encode_ids(ids: pd.Series) -> np.ndarray:
    ids = pd.Series(ids, copy=False).astype(str)
    src = ids.str[1].astype(np.int64).to_numpy()
    num = ids.str[3:].astype(np.int64).to_numpy()
    if not ids.str.match(r"^S[123]-\d+$").all() or (num >= _BASE).any():
        raise ValueError("unexpected entity id format")
    return src * _BASE + num


def decode_ids(codes) -> pd.Series:
    codes = np.asarray(codes, dtype=np.int64)
    src = (codes // _BASE).astype(str)
    num = (codes % _BASE).astype(str)
    return pd.Series(np.char.add(np.char.add(np.char.add("S", src), "-"), num), dtype=str)


def source_of(codes) -> np.ndarray:
    return (np.asarray(codes, dtype=np.int64) // _BASE).astype(np.int8)
