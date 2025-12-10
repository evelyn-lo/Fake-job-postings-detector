"""
Train the fake job classifier and export artifacts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from feature_defs import (
    BASE_NUMERIC_FIELDS,
    CATEGORICAL_FIELDS,
    DERIVED_NUMERIC_FIELDS,
    NUMERIC_FIELDS,
    TEXT_FIELDS,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = PROJECT_ROOT / "data" / "fake_job_postings.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "fake_job_classifier.joblib"
REPORT_PATH = PROJECT_ROOT / "models" / "model_report.json"

SUSPECT_PHRASES = [
    "signing bonus",
    "wire transfer",
    "no experience needed",
    "quick cash",
    "investment required",
    "data entry",
    "aptitude staffing",
    "oil & energy",
    "apply via gmail",
    "training fee",
    "earn money",
    "work from home immediately",
]


@dataclass
class TrainingConfig:
    test_size: float
    random_state: int
    text_features: List[str]
    categorical_features: List[str]
    numeric_features: List[str]
    vectorizer_max_features: int
    vectorizer_min_df: int
    vectorizer_ngram_range: Tuple[int, int]
    model_type: str
    model_params: Dict[str, object]


def load_dataframe(path: Path | str) -> pd.DataFrame:
    """Load the CSV and build engineered features matching inference."""
    df = pd.read_csv(path)

    # Normalize text fields and concatenate.
    for field in TEXT_FIELDS:
        df[field] = df[field].fillna("").astype(str)
    df["text"] = (
        df[TEXT_FIELDS]
        .agg(lambda row: " ".join(value for value in row if value), axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # Base numeric fields with missing flags.
    for field in BASE_NUMERIC_FIELDS:
        missing_mask = df[field].isna()
        df[field] = df[field].fillna(0).astype(int)
        df[f"{field}_missing"] = missing_mask.astype(int)

    text_lower = df["text"].fillna("").str.lower()
    df["text_length"] = df["text"].fillna("").str.split().apply(len)
    df["scam_phrase_hit"] = text_lower.apply(
        lambda x: 1 if any(phrase in x for phrase in SUSPECT_PHRASES) else 0
    )
    df["has_url"] = text_lower.apply(
        lambda x: 1 if ("http://" in x or "https://" in x or "www." in x) else 0
    )
    df["has_email"] = text_lower.apply(lambda x: 1 if "@" in x else 0)
    df["has_phone"] = text_lower.apply(
        lambda x: 1 if any(token in x for token in ["+1", "tel", "phone", "call", "cell"]) else 0
    )
    df["has_freemail"] = text_lower.apply(
        lambda x: 1 if any(domain in x for domain in ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com"]) else 0
    )

    # Ensure all derived numeric columns exist.
    for col in DERIVED_NUMERIC_FIELDS:
        if col not in df.columns:
            df[col] = 0

    return df


def make_preprocessor(config: TrainingConfig) -> ColumnTransformer:
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


def humanize_feature(raw_name: str) -> str:
    if raw_name.startswith("text__"):
        return f"text token '{raw_name.split('__', 1)[1]}'"
    if raw_name.startswith("cat__"):
        stripped = raw_name.split("__", 1)[1]
        column, _, value = stripped.partition("_")
        if value:
            return f"{column.replace('_', ' ')} = {value}"
        return column
    if raw_name.startswith("num__"):
        return raw_name.split("__", 1)[1].replace("_", " ")
    return raw_name


def top_coefficients(feature_names: np.ndarray, coef: np.ndarray, k: int = 15):
    top_pos = np.argsort(coef)[-k:][::-1]
    top_neg = np.argsort(coef)[:k]
    return (
        [
            {
                "feature": humanize_feature(feature_names[i]),
                "raw_feature_name": feature_names[i],
                "weight": float(coef[i]),
            }
            for i in top_pos
        ],
        [
            {
                "feature": humanize_feature(feature_names[i]),
                "raw_feature_name": feature_names[i],
                "weight": float(coef[i]),
            }
            for i in top_neg
        ],
    )


def train_and_evaluate(config: TrainingConfig):
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

    preprocessor = make_preprocessor(config)
    classifier = LogisticRegression(**config.model_params)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
    pipeline.fit(X_train, y_train)

    probs = pipeline.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, preds, average="binary", zero_division=0
    )
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(y_test, probs)),
    }

    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    coef = pipeline.named_steps["classifier"].coef_[0]
    top_fake, top_real = top_coefficients(feature_names, coef)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    report = {
        "dataset": {
            "path": str(DATA_CSV),
            "num_samples": int(len(df)),
            "fraudulent_rate": float(y.mean()),
        },
        "training": {
            "config": {
                "test_size": config.test_size,
                "random_state": config.random_state,
                "text_features": config.text_features,
                "categorical_features": config.categorical_features,
                "numeric_features": config.numeric_features,
                "vectorizer_max_features": config.vectorizer_max_features,
                "vectorizer_min_df": config.vectorizer_min_df,
                "vectorizer_ngram_range": list(config.vectorizer_ngram_range),
                "model_type": config.model_type,
                "model_params": config.model_params,
            },
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
        },
        "metrics": metrics,
        "top_features": {
            "fake": top_fake,
            "real": top_real,
        },
    }

    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Saved model to", MODEL_PATH)
    print("Saved report to", REPORT_PATH)
    print(json.dumps(metrics, indent=2))


def main() -> None:
    default_config = TrainingConfig(
        test_size=0.2,
        random_state=42,
        text_features=TEXT_FIELDS,
        categorical_features=CATEGORICAL_FIELDS,
        numeric_features=NUMERIC_FIELDS,
        vectorizer_max_features=20000,
        vectorizer_min_df=5,
        vectorizer_ngram_range=(1, 3),
        model_type="LogisticRegression",
        model_params={
            "solver": "liblinear",
            "class_weight": "balanced",
            "max_iter": 2000,
            "C": 1.0,
            "penalty": "l2",
        },
    )
    train_and_evaluate(default_config)


if __name__ == "__main__":
    main()

