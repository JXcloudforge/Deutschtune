"""Strict JSONL reader and contract validation for A4 supervised messages."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterator


class DataContractError(ValueError):
    """Raised when an input row violates the declared dataset contract."""


def normalized_input(text: str) -> str:
    """Normalize exact-comparison whitespace while preserving German casing."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def validate_row(row: Any, *, line_number: int = 1) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise DataContractError(f"line {line_number}: row must be a JSON object")
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise DataContractError(f"line {line_number}: 'messages' must be an array")
    if not messages:
        raise DataContractError(f"line {line_number}: 'messages' must not be empty")

    roles: list[str] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise DataContractError(f"line {line_number}: messages[{index}] must be an object")
        role, content = message.get("role"), message.get("content")
        if not isinstance(role, str) or role not in {"system", "user", "assistant"}:
            raise DataContractError(f"line {line_number}: messages[{index}].role is unsupported")
        if not isinstance(content, str) or not content.strip():
            raise DataContractError(f"line {line_number}: messages[{index}].content must be non-empty text")
        roles.append(role)

    allowed = (["system", "user", "assistant"], ["user", "assistant"])
    if roles not in allowed:
        raise DataContractError(
            f"line {line_number}: expected optional system, one user, then one assistant; got {roles}"
        )
    group_id = row.get("group_id")
    if group_id is not None and (not isinstance(group_id, str) or not group_id.strip()):
        raise DataContractError(f"line {line_number}: group_id must be a non-empty string when supplied")
    return row


def validate_input_record(row: Any, *, line_number: int = 1) -> dict[str, Any]:
    """Validate frozen input-only records; reference fields are explicitly forbidden."""
    if not isinstance(row, dict):
        raise DataContractError(f"line {line_number}: row must be a JSON object")
    reference_fields = {
        "expected", "reference", "answer", "gold", "messages", "corrected_sentence",
        "corrected_text", "is_correct", "error_types", "error_categories",
        "explanation_zh", "practice_example_de",
    }
    if row.keys() & reference_fields:
        raise DataContractError(f"line {line_number}: input-only records must not contain reference fields")
    identifier, text, level = row.get("id"), row.get("input"), row.get("level")
    if not isinstance(identifier, str) or not identifier.strip():
        raise DataContractError(f"line {line_number}: id must be non-empty text")
    if not isinstance(text, str) or not text.strip():
        raise DataContractError(f"line {line_number}: input must be non-empty text")
    if not isinstance(level, str) or not level.strip():
        raise DataContractError(f"line {line_number}: level must be non-empty text")
    return row


def iter_jsonl(path: str | Path) -> Iterator[tuple[int, dict[str, Any]]]:
    source = Path(path)
    with source.open("r", encoding="utf-8-sig") as stream:
        for line_number, raw in enumerate(stream, start=1):
            if not raw.strip():
                raise DataContractError(f"line {line_number}: blank lines are not allowed")
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise DataContractError(f"line {line_number}: invalid JSON: {exc.msg}") from exc
            yield line_number, validate_row(row, line_number=line_number)


def iter_input_jsonl(path: str | Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Read input-only JSONL while preventing accidental reference leakage."""
    source = Path(path)
    with source.open("r", encoding="utf-8-sig") as stream:
        for line_number, raw in enumerate(stream, start=1):
            if not raw.strip():
                raise DataContractError(f"line {line_number}: blank lines are not allowed")
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise DataContractError(f"line {line_number}: invalid JSON: {exc.msg}") from exc
            yield line_number, validate_input_record(row, line_number=line_number)


def iter_evaluation_jsonl(path: str | Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Read either frozen input-only rows or legacy supervised fixtures for dry-run compatibility."""
    source = Path(path)
    with source.open("r", encoding="utf-8-sig") as stream:
        for line_number, raw in enumerate(stream, start=1):
            if not raw.strip():
                raise DataContractError(f"line {line_number}: blank lines are not allowed")
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise DataContractError(f"line {line_number}: invalid JSON: {exc.msg}") from exc
            yield line_number, validate_row(row, line_number=line_number) if isinstance(row, dict) and "messages" in row else validate_input_record(row, line_number=line_number)


def input_text(row: dict[str, Any]) -> str:
    """Get a user sentence from either an input-only row or a supervised messages row."""
    if isinstance(row.get("input"), str):
        return row["input"]
    messages = row.get("messages", [])
    return next(message["content"] for message in messages if message.get("role") == "user")
