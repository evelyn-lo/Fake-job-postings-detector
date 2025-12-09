"""
Shared feature configuration for the fake job detection project.
"""

from __future__ import annotations

from typing import List

# Long-form text blobs concatenated into a single field.
LONG_TEXT_FIELDS: List[str] = ["company_profile", "description", "requirements", "benefits"]

TEXT_FIELDS: List[str] = [
    "title",
    "location",
    "department",
    "salary_range",
    *LONG_TEXT_FIELDS,
]

# No categorical columns are fed into the baseline model per the reference tutorial.
CATEGORICAL_FIELDS: List[str] = []

BASE_NUMERIC_FIELDS: List[str] = ["telecommuting", "has_company_logo", "has_questions"]

DERIVED_NUMERIC_FIELDS: List[str] = [
    "text_length",
    "scam_phrase_hit",
    "has_url",
    "has_email",
    "has_phone",
    "has_freemail",
    *[f"{col}_missing" for col in BASE_NUMERIC_FIELDS],
]

NUMERIC_FIELDS: List[str] = BASE_NUMERIC_FIELDS + DERIVED_NUMERIC_FIELDS


__all__ = [
    "TEXT_FIELDS",
    "CATEGORICAL_FIELDS",
    "NUMERIC_FIELDS",
    "LONG_TEXT_FIELDS",
    "DERIVED_NUMERIC_FIELDS",
    "BASE_NUMERIC_FIELDS",
]

