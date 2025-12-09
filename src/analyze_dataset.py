"""
Generate exploratory data analysis outputs for the fake jobs dataset.

Outputs:
- reports/dataset_analysis.json (structured metrics)
- reports/dataset_analysis.md (human-readable summary)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from train_model import (
    CATEGORICAL_FIELDS,
    NUMERIC_FIELDS,
    TEXT_FIELDS,
    load_dataframe,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = PROJECT_ROOT / "data" / "fake_job_postings.csv"
REPORT_JSON = PROJECT_ROOT / "reports" / "dataset_analysis.json"
REPORT_MD = PROJECT_ROOT / "reports" / "dataset_analysis.md"


def compute_lengths(df: pd.DataFrame) -> pd.DataFrame:
    """Add helper columns with length statistics."""
    length_cols = {}
    for field in ["description", "requirements", "benefits", "company_profile"]:
        length_cols[f"{field}_words"] = (
            df[field]
            .fillna("")
            .str.split()
            .apply(len)
        )
    length_cols["text_words"] = (
        df["text"].fillna("").str.split().apply(len)
    )
    return df.assign(**length_cols)


def summarize_numeric(df: pd.DataFrame, label: int) -> Dict[str, float]:
    subset = df[df["fraudulent"] == label]
    stats = {"count": int(len(subset))}
    for column in NUMERIC_FIELDS:
        stats[column] = float(subset[column].mean())
    for column in [
        "text_words",
        "description_words",
        "requirements_words",
        "benefits_words",
        "company_profile_words",
    ]:
        stats[f"avg_{column}"] = float(subset[column].mean())
    return stats


def top_categories(df: pd.DataFrame, column: str, label: int, n: int = 10) -> List[Dict[str, object]]:
    subset = df[df["fraudulent"] == label]
    value_counts = subset[column].value_counts().head(n)
    return [
        {"value": idx, "count": int(count), "share": float(count / len(subset))}
        for idx, count in value_counts.items()
    ]


def top_tokens(df: pd.DataFrame, y: pd.Series, n: int = 20) -> Dict[str, List[Dict[str, object]]]:
    vectorizer = CountVectorizer(
        min_df=25, ngram_range=(1, 2), stop_words="english"
    )
    X = vectorizer.fit_transform(df["text"])
    vocab = np.array(vectorizer.get_feature_names_out())
    totals = {
        0: np.asarray(X[y == 0].sum(axis=0)).flatten() + 1,
        1: np.asarray(X[y == 1].sum(axis=0)).flatten() + 1,
    }
    probs = {label: counts / counts.sum() for label, counts in totals.items()}
    log_odds = np.log(probs[1] / probs[0])
    top_fraud = np.argsort(log_odds)[-n:][::-1]
    top_real = np.argsort(log_odds)[:n]

    def serialize(indices):
        return [
            {
                "token": vocab[idx],
                "fraud_share": float(probs[1][idx]),
                "real_share": float(probs[0][idx]),
                "log_odds": float(log_odds[idx]),
            }
            for idx in indices
        ]

    return {
        "fraudulent_tokens": serialize(top_fraud),
        "legitimate_tokens": serialize(top_real),
    }


def generate_markdown(summary: Dict[str, object]) -> str:
    """Render a narrative summary."""
    overall = summary["overall"]
    fraud = summary["fraudulent"]
    legit = summary["legitimate"]
    tokens = summary["top_tokens"]

    def format_cat(cat_list):
        rows = []
        for item in cat_list[:8]:
            rows.append(f"- {item['value']} ({item['share']*100:.1f}%)")
        return "\n".join(rows)

    fraud_inds = format_cat(summary["fraudulent_top_industries"])
    legit_inds = format_cat(summary["legitimate_top_industries"])
    fraud_funcs = format_cat(summary["fraudulent_top_functions"])
    legit_funcs = format_cat(summary["legitimate_top_functions"])

    def token_table(token_list):
        lines = ["| Phrase | Log-Odds (Fraud vs Legit) |", "| --- | --- |"]
        for item in token_list[:10]:
            lines.append(f"| {item['token']} | {item['log_odds']:.2f} |")
        return "\n".join(lines)

    template = f"""# Fake Job Dataset Exploration

**Rows:** {overall['count']}  
**Fraud rate:** {overall['fraud_rate']*100:.2f}%  
**Median text length:** {overall['median_text_words']} words

## Segment comparison

| Metric | Legitimate | Fraudulent |
| --- | --- | --- |
| Avg word count (full text) | {legit['avg_text_words']:.1f} | {fraud['avg_text_words']:.1f} |
| Avg description words | {legit['avg_description_words']:.1f} | {fraud['avg_description_words']:.1f} |
| Share with company logo | {legit['has_company_logo']*100:.1f}% | {fraud['has_company_logo']*100:.1f}% |
| Share with screening questions | {legit['has_questions']*100:.1f}% | {fraud['has_questions']*100:.1f}% |
| Share telecommuting | {legit['telecommuting']*100:.1f}% | {fraud['telecommuting']*100:.1f}% |

## Top industries

**Fraudulent postings concentrate in:**  
{fraud_inds}

**Legitimate postings concentrate in:**  
{legit_inds}

## Top job functions

**Fraudulent:**  
{fraud_funcs}

**Legitimate:**  
{legit_funcs}

## Fraud-heavy phrases

{token_table(tokens['fraudulent_tokens'])}

## Legitimate-heavy phrases

{token_table(tokens['legitimate_tokens'])}
"""
    return template


def main() -> None:
    df = load_dataframe(DATA_CSV)
    df = compute_lengths(df)
    y = df["fraudulent"]

    overall_stats = {
        "count": int(len(df)),
        "fraud_rate": float(y.mean()),
        "median_text_words": float(df["text_words"].median()),
    }
    summary = {
        "overall": overall_stats,
        "fraudulent": summarize_numeric(df, 1),
        "legitimate": summarize_numeric(df, 0),
        "fraudulent_top_industries": top_categories(df, "industry", 1),
        "legitimate_top_industries": top_categories(df, "industry", 0),
        "fraudulent_top_functions": top_categories(df, "function", 1),
        "legitimate_top_functions": top_categories(df, "function", 0),
        "fraudulent_top_employment_type": top_categories(df, "employment_type", 1),
        "legitimate_top_employment_type": top_categories(df, "employment_type", 0),
        "top_tokens": top_tokens(df, y),
    }

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    markdown = generate_markdown(summary)
    with REPORT_MD.open("w", encoding="utf-8") as f:
        f.write(markdown.strip() + "\n")

    print(f"Saved reports to {REPORT_JSON} and {REPORT_MD}")


if __name__ == "__main__":
    main()


