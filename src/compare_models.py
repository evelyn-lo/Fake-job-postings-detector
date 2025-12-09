"""
Benchmark multiple classifiers on the fake job dataset and report accuracy plus
log losses on train/test splits.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC

from train_model import (
    CATEGORICAL_FIELDS,
    NUMERIC_FIELDS,
    TEXT_FIELDS,
    TrainingConfig,
    load_dataframe,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = PROJECT_ROOT / "data" / "fake_job_postings.csv"
OUTPUT_JSON = PROJECT_ROOT / "models" / "model_comparison.json"
OUTPUT_MD = PROJECT_ROOT / "models" / "model_comparison.md"


def make_preprocessor(config: TrainingConfig) -> ColumnTransformer:
    """Reuse the same feature engineering for every estimator."""
    text_vectorizer = TfidfVectorizer(
        max_features=config.vectorizer_max_features,
        min_df=config.vectorizer_min_df,
        ngram_range=config.vectorizer_ngram_range,
    )
    categorical_encoder = OneHotEncoder(handle_unknown="ignore")
    return ColumnTransformer(
        transformers=[
            ("text", text_vectorizer, "text"),
            ("cat", categorical_encoder, CATEGORICAL_FIELDS),
            ("num", "passthrough", NUMERIC_FIELDS),
        ]
    )


def model_suite(random_state: int) -> List[Tuple[str, object]]:
    """Return estimators to benchmark."""
    return [
        (
            "Logistic Regression",
            LogisticRegression(
                solver="liblinear",
                class_weight="balanced",
                max_iter=2000,
                C=1.0,
            ),
        ),
        (
            "Linear SVM (Calibrated)",
            CalibratedClassifierCV(
                LinearSVC(
                    class_weight="balanced",
                    max_iter=5000,
                ),
                method="sigmoid",
                cv=3,
            ),
        ),
        (
            "Random Forest",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                class_weight="balanced",
                n_jobs=-1,
                random_state=random_state,
            ),
        ),
        (
            "Gradient Boosting",
            GradientBoostingClassifier(
                learning_rate=0.05,
                n_estimators=120,
                max_depth=3,
                random_state=random_state,
            ),
        ),
    ]


def evaluate(pipeline: Pipeline, X, y) -> Dict[str, float]:
    """Compute accuracy, precision/recall/f1, roc auc, and log loss."""
    probs = pipeline.predict_proba(X)
    preds = np.argmax(probs, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, preds, average="binary", zero_division=0
    )

    return {
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(y, probs[:, 1])),
        "log_loss": float(log_loss(y, probs)),
    }


def main() -> None:
    config = TrainingConfig(
        test_size=0.2,
        random_state=42,
        text_features=TEXT_FIELDS,
        categorical_features=CATEGORICAL_FIELDS,
        numeric_features=NUMERIC_FIELDS,
        vectorizer_max_features=5000,
        vectorizer_min_df=5,
        vectorizer_ngram_range=(1, 2),
        model_type="",
        model_params={},
    )

    df = load_dataframe(DATA_CSV)
    X = df[["text"] + CATEGORICAL_FIELDS + NUMERIC_FIELDS]
    y = df["fraudulent"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )

    results: List[Dict[str, object]] = []
    for name, estimator in model_suite(config.random_state):
        print(f"[compare_models] Training {name}...")
        preprocessor = make_preprocessor(config)
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        print(f"[compare_models] Evaluating {name}...")
        train_metrics = evaluate(pipeline, X_train, y_train)
        test_metrics = evaluate(pipeline, X_test, y_test)

        results.append(
            {
                "model": name,
                "train_accuracy": train_metrics["accuracy"],
                "val_accuracy": test_metrics["accuracy"],
                "train_log_loss": train_metrics["log_loss"],
                "val_log_loss": test_metrics["log_loss"],
                "val_precision": test_metrics["precision"],
                "val_recall": test_metrics["recall"],
                "val_f1": test_metrics["f1"],
                "val_roc_auc": test_metrics["roc_auc"],
            }
        )

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Also render a markdown table for quick reference.
    header = (
        "| Model | Train Acc | Val Acc | Train Log Loss | Val Log Loss | "
        "Val Precision | Val Recall | Val F1 | Val ROC AUC |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
    )
    rows = []
    for row in results:
        rows.append(
            "| {model} | {train_accuracy:.3f} | {val_accuracy:.3f} | "
            "{train_log_loss:.3f} | {val_log_loss:.3f} | "
            "{val_precision:.3f} | {val_recall:.3f} | {val_f1:.3f} | "
            "{val_roc_auc:.3f} |".format(**row)
        )

    with OUTPUT_MD.open("w", encoding="utf-8") as f:
        f.write(header + "\n".join(rows) + "\n")

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

