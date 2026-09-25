"""Pairwise matcher training: P(same entity | pair features).

Both learners use histogram gradient boosting on the same features / split:
* XGBoost (Apache-2.0)  — primary
* LightGBM (MIT)        — comparison
Early stopping uses the calibration set (training-fold entities not used for fitting).
"""
from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

XGB_PARAMS = {
    "objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist",
    "max_depth": 8, "eta": 0.1, "subsample": 0.8, "colsample_bytree": 0.8,
    "min_child_weight": 5, "lambda": 1.0, "alpha": 0.0, "gamma": 0.0, "nthread": -1, "seed": 0,
}
LGB_PARAMS = {
    "objective": "binary", "metric": "binary_logloss", "learning_rate": 0.1, "num_leaves": 127,
    "min_data_in_leaf": 50, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
    "lambda_l2": 1.0, "num_threads": -1, "seed": 0, "verbose": -1,
}


def train_xgb(X: pd.DataFrame, y: np.ndarray, Xv: pd.DataFrame, yv: np.ndarray, params: dict | None = None,
              rounds: int = 1500, early: int = 50, weight: np.ndarray | None = None):
    p = {**XGB_PARAMS, **(params or {})}
    t = time.time()
    dtr = xgb.DMatrix(X, label=y, weight=weight)
    dva = xgb.DMatrix(Xv, label=yv)
    bst = xgb.train(p, dtr, num_boost_round=rounds, evals=[(dva, "calib")], early_stopping_rounds=early,
                    verbose_eval=False)
    return bst, {"best_iteration": int(bst.best_iteration), "train_s": time.time() - t, "params": p}


def predict_xgb(bst, X: pd.DataFrame) -> np.ndarray:
    return bst.predict(xgb.DMatrix(X), iteration_range=(0, bst.best_iteration + 1)).astype(np.float32)


def train_lgb(X: pd.DataFrame, y: np.ndarray, Xv: pd.DataFrame, yv: np.ndarray, params: dict | None = None,
              rounds: int = 1500, early: int = 50, weight: np.ndarray | None = None):
    p = {**LGB_PARAMS, **(params or {})}
    t = time.time()
    dtr = lgb.Dataset(X, label=y, weight=weight, free_raw_data=True)
    dva = lgb.Dataset(Xv, label=yv, reference=dtr)
    bst = lgb.train(p, dtr, num_boost_round=rounds, valid_sets=[dva],
                    callbacks=[lgb.early_stopping(early, verbose=False)])
    return bst, {"best_iteration": int(bst.best_iteration), "train_s": time.time() - t, "params": p}


def predict_lgb(bst, X: pd.DataFrame) -> np.ndarray:
    return bst.predict(X, num_iteration=bst.best_iteration).astype(np.float32)
