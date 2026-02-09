"""Simple ML model for directional prediction."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class MLResult:
    label: str
    probability_up: float
    features: list[str]


class MLSignalModel:
    """Train a simple classifier to predict next-day direction."""

    def __init__(self) -> None:
        self.model = None
        self.feature_names: list[str] = []
        self.is_trained = False
        self.last_metrics: dict = {}

    def fit(self, data: pd.DataFrame) -> None:
        df = data.copy()
        df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

        candidates = [
            "sma_fast",
            "sma_slow",
            "ema_fast",
            "ema_slow",
            "rsi",
            "macd",
            "macd_signal",
            "bb_upper",
            "bb_lower",
            "atr",
            "volume_ratio",
        ]
        self.feature_names = [c for c in candidates if c in df.columns]

        df = df.dropna(subset=self.feature_names + ["target"])
        if len(df) < 50 or not self.feature_names:
            self.is_trained = False
            return

        try:
            from sklearn.ensemble import RandomForestClassifier

            X = df[self.feature_names]
            y = df["target"]
            model = RandomForestClassifier(n_estimators=200, random_state=42)
            model.fit(X, y)
            self.model = model
            self.is_trained = True
        except Exception:
            self.is_trained = False
            self.last_metrics = {}

    def predict_latest(self, data: pd.DataFrame) -> MLResult | None:
        if not self.is_trained or self.model is None:
            return None
        latest = data.dropna(subset=self.feature_names).iloc[-1:]
        if latest.empty:
            return None
        proba = float(self.model.predict_proba(latest[self.feature_names])[0][1])
        label = "UP" if proba >= 0.5 else "DOWN"
        return MLResult(label=label, probability_up=proba, features=self.feature_names)

    def feature_importances(self) -> dict:
        if not self.is_trained or self.model is None:
            return {}
        if hasattr(self.model, "feature_importances_"):
            return dict(zip(self.feature_names, self.model.feature_importances_))
        return {}

    def evaluate_walk_forward(self, data: pd.DataFrame, splits: int = 5) -> dict:
        df = data.copy()
        df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
        df = df.dropna(subset=self.feature_names + ["target"])
        if len(df) < 200:
            return {"status": "insufficient_data"}

        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import TimeSeriesSplit
            from sklearn.metrics import accuracy_score
        except Exception:
            return {"status": "sklearn_unavailable"}

        X = df[self.feature_names].values
        y = df["target"].values
        tscv = TimeSeriesSplit(n_splits=splits)
        scores = []
        for train_idx, test_idx in tscv.split(X):
            model = RandomForestClassifier(n_estimators=200, random_state=42)
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[test_idx])
            scores.append(float(accuracy_score(y[test_idx], pred)))

        metrics = {
            "status": "ok",
            "splits": splits,
            "accuracy_mean": float(sum(scores) / len(scores)) if scores else 0.0,
            "accuracy_per_split": scores,
        }
        self.last_metrics = metrics
        return metrics
