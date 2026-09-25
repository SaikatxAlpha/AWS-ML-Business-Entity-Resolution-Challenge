"""End-to-end inference: raw TSVs -> normalisation -> blocking -> candidate_pairs -> decision
-> matching_results -> official validator.

The normaliser used here is learned from **all** training labels (tag ``all``); retrieval
statistics are computed on the inference split's own records, so unseen countries are
handled from their own data. The decision rule/model is read from ``models/``.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from ..blocking.tfidf_blocking import TfidfBlockingConfig, tfidf_candidates
from ..config import DATA_DIR, MODELS_DIR, OUTPUT_DIR, ROOT, cache_path
from ..evaluation.validation import folds, truth_pairs
from ..models.baseline import apply_rule
from ..preprocessing.records import PREPROCESSING_VERSION, learn_normalizer, normalized
from ..utils.io import write_submission


def _add_raw_tokens(rec: pd.DataFrame, cols) -> pd.DataFrame:
    for src, dst in (("name_raw", "name_rawtok"), ("addr_raw", "addr_rawtok")):
        if dst in cols:
            rec[dst] = rec[src].str.lower().str.replace(r"[^0-9a-z]+", " ", regex=True).str.strip()
    return rec


def run_validator(out_dir=OUTPUT_DIR, test_dir=None) -> tuple[int, str]:
    """Official validator, two passes (as its own docs recommend for large candidate files):
    1. every rule on both files (incl. matches ⊆ candidates);
    2. ``--check-ids`` (IDs exist in test S2/S3) on the scored matching file only."""
    base = [sys.executable, str(ROOT / "utils" / "validate_submission.py"),
            "--matching", str(out_dir / "matching_results.tsv"), "--test-dir", str(test_dir or DATA_DIR / "test")]
    outs, code = [], 0
    for extra in (["--candidate", str(out_dir / "candidate_pairs.tsv")],
                  ["--candidate", str(out_dir / "__skip_candidate__.tsv"), "--check-ids"]):
        res = subprocess.run(base + extra, capture_output=True, text=True, encoding="utf-8", errors="replace",
                             env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        outs.append(f"$ validate_submission.py {' '.join(extra)}\n{res.stdout}{res.stderr}")
        code = max(code, res.returncode)
    return code, "\n".join(outs)


def run_inference(model_file: str = "baseline_rule.json", split: str = "test", out_dir=OUTPUT_DIR,
                  validate: bool = True) -> dict:
    t0 = time.time()
    spec = json.loads((MODELS_DIR / model_file).read_text(encoding="utf-8"))
    nz = learn_normalizer(folds()["s1"], truth_pairs(), tag=spec.get("normalizer_tag", "f0"))
    rec = normalized(split, nz)
    rec = _add_raw_tokens(rec, set(spec["blocking"]["fields"].values()) | {spec["rule"].get("name_col"), spec["rule"].get("addr_col")})
    s1_codes = rec.loc[rec["source"] == 1, "code"].to_numpy()
    cfg = TfidfBlockingConfig(fields=spec["blocking"]["fields"], top_k=spec["blocking"]["top_k"],
                              max_df=spec["blocking"]["max_df"])
    cpath = cache_path("candidates", f"{split}_{spec['name']}_{PREPROCESSING_VERSION}_k{cfg.top_k}.parquet")
    if cpath.exists():
        cands = pd.read_parquet(cpath)
    else:
        cands = tfidf_candidates(rec, s1_codes, cfg)
        cands.to_parquet(cpath, index=False)
    t_block = time.time() - t0
    matches = apply_rule(cands, rec, spec["rule"])
    stats = write_submission(out_dir, s1_codes, cands, matches)
    per_country = rec[rec["source"] == 1].groupby("country").size().to_dict()
    matched_s1 = rec[(rec["source"] == 1) & rec["code"].isin(matches["s1"])].groupby("country").size()
    report = {
        "model": spec["name"], "split": split, "s1_entities": int(len(s1_codes)),
        "s1_by_country": per_country,
        "predicted_nonempty_rate_by_country": (matched_s1 / pd.Series(per_country)).fillna(0).round(4).to_dict(),
        "avg_candidates": float(len(cands) / len(s1_codes)), "avg_matches": float(len(matches) / len(s1_codes)),
        "files": stats, "blocking_s": t_block, "total_s": time.time() - t0,
    }
    if validate and split == "test":
        code, out = run_validator(out_dir)
        report["validator_exit_code"] = code
        report["validator_output"] = out
    (out_dir / "run_report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    return report
