from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from feature_defs import (  # noqa: E402
    BASE_NUMERIC_FIELDS,
    CATEGORICAL_FIELDS,
    DERIVED_NUMERIC_FIELDS,
    LONG_TEXT_FIELDS,
    NUMERIC_FIELDS,
    TEXT_FIELDS,
)

MODEL_PATH = BASE_DIR / "models" / "fake_job_classifier.joblib"
REPORT_PATH = BASE_DIR / "models" / "model_report.json"
FEEDBACK_PATH = BASE_DIR / "reports" / "user_feedback.json"

app = Flask(__name__)
pipeline = joblib.load(MODEL_PATH)
with REPORT_PATH.open("r", encoding="utf-8") as f:
    MODEL_REPORT = json.load(f)

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


def humanize_feature(raw_name: str) -> str:
    """Convert engineering-style names into concise explanations."""
    if raw_name.startswith("text__"):
        return f"Text token '{raw_name.split('__', 1)[1]}'"
    if raw_name.startswith("cat__"):
        stripped = raw_name.split("__", 1)[1]
        column, _, value = stripped.partition("_")
        if value:
            return f"{column.replace('_', ' ')} = {value}"
        return column
    if raw_name.startswith("num__"):
        return raw_name.split("__", 1)[1].replace("_", " ")
    return raw_name


def sanitize_form_value(value: str | None, fallback: str = "") -> str:
    return (value or fallback).strip()


def parse_flag(value: str | None, missing_value: int = -1) -> int:
    """Parse form flag values; treat 'none' or blank as missing_value."""
    cleaned = sanitize_form_value(value, fallback="none").lower()
    if cleaned in ("none", ""):
        return missing_value
    try:
        return int(cleaned)
    except (TypeError, ValueError):
        return missing_value


def build_model_frame(form_data: Dict[str, str]) -> pd.DataFrame:
    """Turn raw form input into the dataframe the pipeline expects."""
    record: Dict[str, object] = {}
    for field in TEXT_FIELDS:
        record[field] = sanitize_form_value(form_data.get(field, ""))
    for field in CATEGORICAL_FIELDS:
        record[field] = sanitize_form_value(form_data.get(field, "Unknown")) or "Unknown"

    # Handle base numeric fields: allow "none" to mean missing -> -1
    base_values: Dict[str, int] = {}
    for field in BASE_NUMERIC_FIELDS:
        parsed = parse_flag(form_data.get(field), missing_value=-1)
        is_missing = 1 if parsed == -1 else 0
        # For the main numeric feature, neutralize missing values to 0 so they don't drive weights.
        record[field] = 0 if is_missing else parsed
        base_values[field] = record[field]
        record[f"{field}_missing"] = is_missing

    df = pd.DataFrame([record])
    df["text"] = (
        df[TEXT_FIELDS]
        .agg(lambda row: " ".join(value for value in row if value), axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # Derived numeric cues
    text_lower = df["text"].fillna("").str.lower()
    df["text_length"] = df["text"].fillna("").str.split().apply(len)
    df["scam_phrase_hit"] = text_lower.apply(
        lambda x: 1 if any(phrase in x for phrase in SUSPECT_PHRASES) else 0
    )
    df["has_url"] = text_lower.apply(
        lambda x: 1 if ("http://" in x or "https://" in x or "www." in x) else 0
    )
    df["has_email"] = text_lower.apply(lambda x: 1 if "@" in x else 0)
    df["has_phone"] = text_lower.apply(lambda x: 1 if any(token in x for token in ["+1", "tel", "phone", "call", "cell"]) else 0)
    df["has_freemail"] = text_lower.apply(
        lambda x: 1 if any(domain in x for domain in ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com"]) else 0
    )

    # Ensure all expected numeric columns exist, fill absent with 0
    for col in DERIVED_NUMERIC_FIELDS:
        if col not in df.columns:
            df[col] = 0

    return df[["text"] + CATEGORICAL_FIELDS + NUMERIC_FIELDS]


def explain_prediction(
    model_frame: pd.DataFrame, top_k: int = 5
) -> Dict[str, object]:
    """Return probability, label, and top local feature contributions."""
    probabilities = pipeline.predict_proba(model_frame)[0]
    pred_class = int(probabilities[1] >= 0.5)

    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    transformed = preprocessor.transform(model_frame)
    feature_names = preprocessor.get_feature_names_out()
    contributions = transformed.multiply(classifier.coef_[0]).toarray()[0]

    if pred_class == 1:
        target_indices = np.where(contributions > 0)[0]
        sort_order = contributions
        default_indices = np.argsort(contributions)[-top_k:]
    else:
        target_indices = np.where(contributions < 0)[0]
        sort_order = -contributions
        default_indices = np.argsort(contributions)[:top_k]

    if target_indices.size == 0:
        chosen = default_indices
    else:
        chosen = target_indices[np.argsort(sort_order[target_indices])][-top_k:]

    chosen = chosen[::-1]  # highest impact first

    row = model_frame.iloc[0]
    reasons: List[Dict[str, object]] = []
    for idx in chosen:
        raw_name = feature_names[idx]
        # Suppress base numeric signals when the user marked the field as missing (-1).
        if raw_name.startswith("num__"):
            col = raw_name.split("__", 1)[1]
            if col in BASE_NUMERIC_FIELDS and row.get(col, 0) == -1:
                continue
        reasons.append(
            {
                "feature": humanize_feature(raw_name),
                "impact": float(contributions[idx]),
            }
        )
    # If everything was suppressed, fall back to the top candidates (unsuppressed).
    if not reasons:
        for idx in chosen[:top_k]:
            reasons.append(
                {
                    "feature": humanize_feature(feature_names[idx]),
                    "impact": float(contributions[idx]),
                }
            )

    confidence = float(abs(probabilities[1] - 0.5) * 2)
    is_uncertain = 0.45 <= probabilities[1] <= 0.65

    return {
        "label": "Likely Fake" if pred_class == 1 else "Likely Legitimate",
        "prob_fake": float(probabilities[1]),
        "prob_real": float(probabilities[0]),
        "reasons": reasons,
        "confidence": confidence,
        "is_uncertain": is_uncertain,
    }


def find_suspicious_phrases(text: str) -> List[str]:
    lower = text.lower()
    return [phrase for phrase in SUSPECT_PHRASES if phrase in lower][:6]


def generate_counterfactuals(form_values: Dict[str, str], frame_row: pd.Series) -> List[str]:
    suggestions = []
    if frame_row.get("text_length", 0) < 60:
        suggestions.append("Add a detailed description of responsibilities.")
    if not form_values.get("company_profile", "").strip():
        suggestions.append("Include a company profile to establish legitimacy.")
    if not form_values.get("description", "").strip():
        suggestions.append("Provide a complete job description instead of leaving it blank.")
    if not form_values.get("requirements", "").strip():
        suggestions.append("List concrete requirements or qualifications.")
    if not form_values.get("benefits", "").strip():
        suggestions.append("Describe salary or benefits to reduce suspicion.")
    if parse_flag(form_values.get("has_company_logo"), missing_value=0) == 0:
        suggestions.append("Add a verified company logo to increase trust.")
    if parse_flag(form_values.get("has_questions"), missing_value=0) == 0:
        suggestions.append("Add screening questions to show due diligence.")
    if not form_values.get("salary_range", "").strip():
        suggestions.append("Share a realistic salary range.")
    if not suggestions:
        suggestions.append("Job already covers key details; verify contact info uses company domains.")
    return suggestions[:5]


def append_feedback(entry: Dict[str, str]) -> None:
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    data: List[Dict[str, str]] = []
    if FEEDBACK_PATH.exists():
        try:
            with FEEDBACK_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = []
    data.append(entry)
    with FEEDBACK_PATH.open("w", encoding="utf-8") as f:
        json.dump(data[-500:], f, indent=2)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    form_defaults = {field: "" for field in TEXT_FIELDS + CATEGORICAL_FIELDS}
    form_defaults.update({field: "none" for field in BASE_NUMERIC_FIELDS})
    suspicious_phrases: List[str] = []
    counterfactuals: List[str] = []
    confidence_percent = None
    uncertainty_note = None
    feedback_message = None

    if request.method == "POST":
        action = request.form.get("form_type", "score")
        payload_data: Dict[str, str] = {}
        if action == "feedback":
            try:
                payload_data = json.loads(request.form.get("form_payload", "{}"))
            except json.JSONDecodeError:
                payload_data = {}
            form_defaults.update(payload_data)
            feedback_value = request.form.get("feedback_value")
            if feedback_value:
                append_feedback(
                    {
                        "feedback": feedback_value,
                        "timestamp": datetime.utcnow().isoformat(),
                        "label": request.form.get("label"),
                        "prob_fake": request.form.get("prob_fake"),
                    }
                )
                feedback_message = "Thanks for the feedback! It helps us tune the detector."
        else:
            payload_data = {**request.form}
            form_defaults.update(payload_data)

        if payload_data:
            try:
                frame = build_model_frame(payload_data)
                result = explain_prediction(frame)
                text_blob = frame["text"].iloc[0]
                suspicious_phrases = find_suspicious_phrases(text_blob)
                counterfactuals = generate_counterfactuals(payload_data, frame.iloc[0])
                confidence_percent = round(result["confidence"] * 100, 1)
                if result["is_uncertain"]:
                    uncertainty_note = (
                        "The model is uncertain about this posting - consider manual review."
                    )
            except Exception as exc:  # noqa: BLE001
                error = f"Unable to score posting: {exc}"

    return render_template(
        "index.html",
        result=result,
        error=error,
        form_values=form_defaults,
        metrics=MODEL_REPORT["metrics"],
        top_features=MODEL_REPORT["top_features"],
        training=MODEL_REPORT["training"],
        dataset=MODEL_REPORT["dataset"],
        text_fields=TEXT_FIELDS,
        categorical_fields=CATEGORICAL_FIELDS,
        base_numeric_fields=BASE_NUMERIC_FIELDS,
        suspicious_phrases=suspicious_phrases,
        counterfactuals=counterfactuals,
        confidence_percent=confidence_percent,
        uncertainty_note=uncertainty_note,
        feedback_message=feedback_message,
        fairness_statement="",
    )


if __name__ == "__main__":
    app.run(debug=True)

