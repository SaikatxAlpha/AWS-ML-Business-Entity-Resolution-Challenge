import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from business_entity_resolution.utils.ids import encode_ids
from business_entity_resolution.utils.io import read_id_list_file, write_submission

ROOT = Path(__file__).resolve().parents[1]


def _codes(ids):
    return encode_ids(pd.Series(ids))


def _pairs(rows):
    return pd.DataFrame({"s1": _codes([a for a, _, _ in rows]), "cand": _codes([b for _, b, _ in rows]),
                         "score": [s for _, _, s in rows]})


def test_write_read_and_official_validator(tmp_path):
    s1 = ["S1-1", "S1-2", "S1-3"]
    cands = _pairs([("S1-1", "S2-10", .9), ("S1-1", "S3-11", .5), ("S1-1", "S2-10", .9), ("S1-2", "S3-12", .7)])
    matches = cands[cands["score"] > 0.6]
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    hdr = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
    (test_dir / "test_source1.tsv").write_text(hdr + "".join(f"{i}\tn\ta\tX\n" for i in s1))
    (test_dir / "test_source2.tsv").write_text(hdr + "S2-10\tn\ta\tX\n")
    (test_dir / "test_source3.tsv").write_text(hdr + "S3-11\tn\ta\tX\nS3-12\tn\ta\tX\n")
    out = tmp_path / "output"
    stats = write_submission(out, _codes(s1), cands, matches)
    assert stats["matching_results.tsv"]["rows"] == 3
    lines = (out / "matching_results.tsv").read_text().splitlines()
    assert lines == ["source1_entity_id\tmatched_entity_ids", "S1-1\tS2-10", "S1-2\tS3-12", "S1-3\t"]
    cl = (out / "candidate_pairs.tsv").read_text().splitlines()
    assert cl[1] == "S1-1\tS2-10,S3-11"          # deduplicated, ordered by score
    back = read_id_list_file(out / "matching_results.tsv")
    assert set(zip(back.s1, back.cand)) == set(zip(matches.s1, matches.cand))
    res = subprocess.run([sys.executable, str(ROOT / "utils" / "validate_submission.py"),
                          "--matching", str(out / "matching_results.tsv"),
                          "--candidate", str(out / "candidate_pairs.tsv"),
                          "--test-dir", str(test_dir), "--check-ids"], capture_output=True, text=True, encoding="utf-8",
                         env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    assert res.returncode == 0, res.stdout
    assert "PASS" in res.stdout


def test_writer_rejects_match_outside_candidates(tmp_path):
    cands = _pairs([("S1-1", "S2-10", .9)])
    matches = _pairs([("S1-1", "S3-99", .9)])
    with pytest.raises(ValueError):
        write_submission(tmp_path, _codes(["S1-1"]), cands, matches)
