"""Probability calibration (Platt / isotonic) and reliability measurement.

Calibrators are fitted on one half of the calibration entities and assessed on the other
half (cross-fitting), so reliability numbers are out-of-sample.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def ece(p: np.ndarray, y: np.ndarray, bins: int = 20) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(tot)


def reliability(p: np.ndarray, y: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return [{"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": int((idx == b).sum()),
             "mean_p": float(p[idx == b].mean()) if (idx == b).any() else None,
             "frac_pos": float(y[idx == b].mean()) if (idx == b).any() else None} for b in range(bins)]


class Calibrator:
    def __init__(self, kind: str):
        self.kind = kind
        self.model = None

    def fit(self, p: np.ndarray, y: np.ndarray) -> "Calibrator":
        if self.kind == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)
        elif self.kind == "platt":
            z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1)).reshape(-1, 1)
            self.model = LogisticRegression(C=1e6).fit(z, y)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        if self.kind == "isotonic":
            return self.model.predict(p).astype(np.float32)
        if self.kind == "platt":
            z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1)).reshape(-1, 1)
            return self.model.predict_proba(z)[:, 1].astype(np.float32)
        return p
