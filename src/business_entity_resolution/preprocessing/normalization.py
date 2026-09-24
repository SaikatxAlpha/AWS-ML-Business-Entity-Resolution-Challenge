"""Shared low-level text normalisation.

``basic_normalize`` is deliberately conservative: Unicode NFKC, offline transliteration to
ASCII (anyascii, ISC licence), lowercase, and punctuation -> single space. Field-specific
logic (legal suffixes, abbreviations, address components) lives in
``name_normalization`` / ``address_normalization``.
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd
from anyascii import anyascii

_NON_ALNUM = re.compile(r"[^0-9a-z]+")
_NON_ASCII_PATTERN = r"[^\x00-\x7f]"


def to_ascii(text: str) -> str:
    if text.isascii():
        return text
    return anyascii(unicodedata.normalize("NFKC", text))


def basic_normalize(text: str) -> str:
    return _NON_ALNUM.sub(" ", to_ascii(text).lower()).strip()


def to_ascii_series(s: pd.Series) -> pd.Series:
    """Vectorised ``to_ascii``: only non-ASCII rows go through Python."""
    s = s.astype(str)
    mask = s.str.contains(_NON_ASCII_PATTERN, regex=True)
    if mask.any():
        s = s.copy()
        s[mask] = [to_ascii(x) for x in s[mask]]
    return s


def basic_normalize_series(s: pd.Series) -> pd.Series:
    return (
        to_ascii_series(s)
        .str.lower()
        .str.replace(r"[^0-9a-z]+", " ", regex=True)
        .str.strip()
    )


def unicode_script(ch: str) -> str:
    """Coarse Unicode script of a character (first word of its Unicode name)."""
    name = unicodedata.name(ch, "")
    return name.split(" ", 1)[0] if name else "UNKNOWN"
