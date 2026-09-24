"""Structural integrity checks for raw inputs.

These checks stream the raw TSV (independently of pandas) so that a parsing problem in
the loader cannot hide a malformed row.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import pandas as pd

from ..config import ID_PREFIX, source_path


@dataclass
class RawFileCheck:
    path: str
    n_lines: int = 0
    field_counts: Counter = field(default_factory=Counter)
    rows_with_quote: int = 0
    rows_with_cr: int = 0
    has_bom: bool = False
    bad_examples: list = field(default_factory=list)

    @property
    def malformed(self) -> int:
        return sum(v for k, v in self.field_counts.items() if k != 4)


def check_raw_file(split: str, source: int) -> RawFileCheck:
    path = source_path(split, source)
    res = RawFileCheck(str(path))
    with open(path, "rb") as fb:
        res.has_bom = fb.read(3) == b"\xef\xbb\xbf"
    with open(path, encoding="utf-8", newline="") as f:
        next(f)
        for line in f:
            res.n_lines += 1
            if "\r" in line:
                res.rows_with_cr += 1
            if '"' in line:
                res.rows_with_quote += 1
            n = line.rstrip("\r\n").count("\t") + 1
            res.field_counts[n] += 1
            if n != 4 and len(res.bad_examples) < 5:
                res.bad_examples.append(line[:200])
    return res


def check_frame(df: pd.DataFrame, source: int) -> dict:
    """Checks on a loaded source frame; returns a dict of counts (all should be 0 except n)."""
    ids = df["entity_id"]
    return {
        "n": len(df),
        "duplicate_ids": int(ids.duplicated().sum()),
        "bad_prefix": int((~ids.str.startswith(ID_PREFIX[source])).sum()),
        "empty_id": int((ids.str.strip() == "").sum()),
        "empty_country": int((df["country"].str.strip() == "").sum()),
    }


def check_ground_truth(gt: pd.DataFrame, s1_ids: pd.Series, s23_ids: set) -> dict:
    gt_ids = set(gt["source1_entity_id"])
    s1 = set(s1_ids)
    lists = gt["matched_entity_ids"]
    exploded = lists[lists != ""].str.split(",").explode()
    return {
        "gt_rows": len(gt),
        "duplicate_gt_rows": int(gt["source1_entity_id"].duplicated().sum()),
        "s1_missing_from_gt": len(s1 - gt_ids),
        "gt_not_in_s1": len(gt_ids - s1),
        "matched_ids_not_in_sources": int((~exploded.isin(s23_ids)).sum()),
        "matched_ids_bad_prefix": int((~exploded.str.match(r"^S[23]-")).sum()),
        "match_ids_linked_to_multiple_s1": int(exploded.duplicated().sum()),
    }
