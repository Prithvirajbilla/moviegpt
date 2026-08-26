"""Shared record schema and deterministic splits for MovieGPT."""

from __future__ import annotations

import hashlib
from typing import Any


SPLITS = ("train", "validation", "test")
DOCUMENT_FIELDS = (
    "document_id",
    "source",
    "source_id",
    "title",
    "language",
    "license",
    "text",
    "text_sha256",
    "character_count",
    "byte_count",
    "line_count",
)


def document_split(document_id: str) -> str:
    """Assign a stable 90/5/5 split without depending on corpus order."""

    bucket = int(hashlib.sha256(document_id.encode("utf-8")).hexdigest()[:8], 16)
    percentile = bucket % 10_000
    if percentile < 9_000:
        return "train"
    if percentile < 9_500:
        return "validation"
    return "test"


def parquet_schema() -> Any:
    """Return the PyArrow schema without requiring PyArrow for non-Parquet tasks."""

    import pyarrow as pa

    return pa.schema(
        [
            pa.field("document_id", pa.string(), nullable=False),
            pa.field("source", pa.string(), nullable=False),
            pa.field("source_id", pa.string(), nullable=False),
            pa.field("title", pa.string(), nullable=False),
            pa.field("language", pa.string(), nullable=False),
            pa.field("license", pa.string(), nullable=False),
            pa.field("text", pa.large_string(), nullable=False),
            pa.field("text_sha256", pa.string(), nullable=False),
            pa.field("character_count", pa.int64(), nullable=False),
            pa.field("byte_count", pa.int64(), nullable=False),
            pa.field("line_count", pa.int64(), nullable=False),
        ]
    )
