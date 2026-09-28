"""Train an optional calibrated supervised ranker for structured scanner telemetry.

The ranker is a triage signal only. Confirmed vulnerabilities still require the
category-specific proof verifiers in detection.signature.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import DATASETS_DIR, MODELS_DIR
from detection.ml_model import MLAnomalyDetector

FEATURE_COLUMNS = MLAnomalyDetector.FEATURE_KEYS


def load_training_frame(dataset_path: Path) -> pd.DataFrame:
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. Provide a proof-verified telemetry dataset first."
        )
    frame = pd.read_csv(dataset_path).fillna(0.0)
    missing = [column for column in FEATURE_COLUMNS + ["label"] if column not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    frame["label"] = frame["label"].astype(int)
    labels = set(frame["label"].unique())
    if not labels.issubset({0, 1}) or len(labels) < 2:
        raise ValueError("Training labels must contain both normal (0) and confirmed attack (1) rows.")
    return frame


def build_model() -> CalibratedClassifierCV:
    base = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )
    try:
        return CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)
    except TypeError:  # scikit-learn compatibility with older releases
        return CalibratedClassifierCV(base_estimator=base, method="sigmoid", cv=3)


def train(dataset_path: Path, model_path: Path, metrics_path: Path | None = None) -> dict:
    frame = load_training_frame(dataset_path)
    x_train, x_test, y_train, y_test = train_test_split(
        frame[FEATURE_COLUMNS],
        frame["label"],
        test_size=0.2,
        random_state=42,
        stratify=frame["label"],
    )
    model = build_model()
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "samples": int(len(frame)),
        "train_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        "positive_rate": float(frame["label"].mean()),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "average_precision": float(average_precision_score(y_test, probabilities)),
        "brier_score": float(brier_score_loss(y_test, probabilities)),
        "evaluation_warning": "Random row split; use target/endpoint/payload-family grouped splits for production claims.",
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_keys": FEATURE_COLUMNS, "metrics": metrics}, model_path)
    if metrics_path:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the optional calibrated tabular triage ranker.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(DATASETS_DIR) / "processed" / "features.csv",
        help="Proof-verified telemetry CSV with scanner feature columns and a binary label.",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path(MODELS_DIR) / "tabular_ranker.pkl",
    )
    parser.add_argument("--metrics-output", type=Path, default=None)
    args = parser.parse_args()
    train(args.dataset, args.model_output, args.metrics_output)


if __name__ == "__main__":
    main()
