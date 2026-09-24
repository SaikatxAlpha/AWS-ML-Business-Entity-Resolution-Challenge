"""Tokenisation helpers shared by names and addresses."""
from __future__ import annotations

import re

_NUM = re.compile(r"\d+")
_OCR_TOKEN = re.compile(r"^(?=(?:.*[a-z]){2})[a-z0-9]*[0134568][a-z0-9]*$")
_ORDINAL = re.compile(r"^\d+(?:st|nd|rd|th)$")
_OCR_TABLE =str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "6": "g", "8": "b"})


def tokens(text: str) -> list[str]:
    return text.split()


def fix_ocr_token(tok: str) -> str:
    """Undo digit-for-letter OCR swaps inside alphabetic tokens (``nati0nal``, ``8rothers``).

    Only tokens with >= 2 letters whose digits are all OCR-confusable are changed, so
    alphanumeric identifiers such as ``b13`` or ``24hr`` are left untouched.
    """
    if tok.isalpha() or tok.isdigit() or _ORDINAL.match(tok) or not _OCR_TOKEN.match(tok):
        return tok
    if any(c.isdigit() and c not in "0134568" for c in tok):
        return tok
    return tok.translate(_OCR_TABLE)


def numbers(text: str) -> list[str]:
    """Digit runs with leading zeros removed (``003623`` -> ``3623``)."""
    return [m.lstrip("0") or "0" for m in _NUM.findall(text)]


def is_subsequence(short: str, long: str) -> bool:
    it = iter(long)
    return all(c in it for c in short)


def abbreviation_compatible(a: str, b: str) -> bool:
    """True if the shorter token is an in-order subsequence of the longer one with the same
    first letter (``rd``/``road``, ``mh``/``maharashtra``, ``av``/``avenue``)."""
    if not a or not b or a[0] != b[0] or a.isdigit() or b.isdigit():
        return False
    s, l = (a, b) if len(a) <= len(b) else (b, a)
    return len(s) >= 2 and is_subsequence(s, l)
