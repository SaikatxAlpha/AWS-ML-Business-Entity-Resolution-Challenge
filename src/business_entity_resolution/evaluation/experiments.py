"""Experiment tracking: every run appends one row to experiments/results.csv and writes the
full metric dict to experiments/runs/<experiment_id>.json. Failed / worse experiments are
logged too — nothing is overwritten."""
from __future__ import annotations

import csv
import datetime as dt
import json

import psutil

from ..config import EXPERIMENTS_DIR

COLUMNS = [
    "experiment_id", "date", "preprocessing_version", "blocking_version", "feature_version", "model",
    "hyperparameters", "threshold", "candidate_recall", "precision", "recall", "f05", "singleton_accuracy",
    "average_candidates", "runtime_s", "memory_gb", "notes",
]


def _num(v, nd=5):
    return "" if v is None else (round(v, nd) if isinstance(v, float) else v)


def log_experiment(experiment_id: str, *, metrics: dict, cand_metrics: dict | None = None,
                   preprocessing_version: str = "", blocking_version: str = "", feature_version: str = "",
                   model: str = "", hyperparameters: dict | None = None, threshold=None,
                   runtime_s: float | None = None, notes: str = "", extra: dict | None = None) -> None:
    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    runs = EXPERIMENTS_DIR / "runs"
    runs.mkdir(exist_ok=True)
    cand_metrics = cand_metrics or {}
    row = {
        "experiment_id": experiment_id,
        "date": dt.datetime.now().isoformat(timespec="seconds"),
        "preprocessing_version": preprocessing_version,
        "blocking_version": blocking_version,
        "feature_version": feature_version,
        "model": model,
        "hyperparameters": json.dumps(hyperparameters or {}, sort_keys=True),
        "threshold": json.dumps(threshold) if isinstance(threshold, (dict, list)) else _num(threshold),
        "candidate_recall": _num(cand_metrics.get("candidate_recall")),
        "precision": _num(metrics.get("micro_precision")),
        "recall": _num(metrics.get("micro_recall")),
        "f05": _num(metrics.get("f05")),
        "singleton_accuracy": _num(metrics.get("singleton_accuracy")),
        "average_candidates": _num(cand_metrics.get("avg_candidates"), 2),
        "runtime_s": _num(runtime_s, 1),
        "memory_gb": round(psutil.Process().memory_info().rss / 1e9, 2),
        "notes": notes,
    }
    path = EXPERIMENTS_DIR / "results.csv"
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)
    full = {"row": row, "metrics": metrics, "candidate_metrics": cand_metrics, "extra": extra or {}}
    (runs / f"{experiment_id}.json").write_text(json.dumps(full, indent=2, default=str), encoding="utf-8")
