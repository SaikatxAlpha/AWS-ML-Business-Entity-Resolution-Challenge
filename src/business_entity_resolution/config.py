"""Central configuration: paths, seeds and dataset constants.

Paths can be overridden with environment variables so the pipeline runs from any
checkout location:  BER_ROOT (project root), BER_DATA_DIR, BER_CACHE_DIR.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("BER_ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR = Path(os.environ.get("BER_DATA_DIR", ROOT / "dataset"))
CACHE_DIR = Path(os.environ.get("BER_CACHE_DIR", ROOT / "cache"))
REPORTS_DIR = ROOT / "reports"
OUTPUT_DIR = ROOT / "output"
MODELS_DIR = ROOT / "models"
EXPERIMENTS_DIR = ROOT / "experiments"

SEED = 20260925

SPLITS = ("train", "test")
SOURCES = (1, 2, 3)
SOURCE_COLUMNS = ("entity_id", "business_name", "business_address", "country")
GT_COLUMNS = ("source1_entity_id", "matched_entity_ids")
ID_PREFIX = {1: "S1-", 2: "S2-", 3: "S3-"}


def source_path(split: str, source: int) -> Path:
    return DATA_DIR / split / f"{split}_source{source}.tsv"


def ground_truth_path() -> Path:
    return DATA_DIR / "train" / "train_ground_truth.tsv"


def cache_path(*parts: str) -> Path:
    p = CACHE_DIR.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
