"""Record schema definitions for the raw competition files."""
from __future__ import annotations

import csv

from ..config import GT_COLUMNS, SOURCE_COLUMNS

# Every column is read as a string. The files contain literal `"` characters inside
# fields, so quoting MUST be disabled or rows get silently merged. `na_filter=False`
# keeps strings such as "NA" / "None" / "<NULL>" verbatim instead of turning them into NaN
# (they are noise we want to see and normalise ourselves).
READ_KWARGS = dict(
    sep="\t",
    quoting=csv.QUOTE_NONE,
    dtype=str,
    na_filter=False,
    keep_default_na=False,
    encoding="utf-8",
)

SOURCE_SCHEMA = list(SOURCE_COLUMNS)
GT_SCHEMA = list(GT_COLUMNS)
