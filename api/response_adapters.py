"""Independent JSON and NDJSON response serialization adapters."""

from __future__ import annotations

import json
from typing import Any


def serialize(payload: dict[str, Any], response_format: str) -> tuple[str, str]:
    if response_format == "json":
        return json.dumps(payload, ensure_ascii=False), "application/json; charset=utf-8"
    if response_format == "ndjson":
        return json.dumps(payload, ensure_ascii=False) + "\n", "application/x-ndjson; charset=utf-8"
    raise ValueError("response_format must be json or ndjson")
