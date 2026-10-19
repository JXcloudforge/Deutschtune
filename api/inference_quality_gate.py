"""Local I01 quality-gate orchestration with injected model and rule Oracle.

This module does not load a model, call HTTP services, or implement grammar
rules. Callers inject those boundaries; tests use a deterministic mock only.
"""
from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from typing import Any, Callable, Protocol


GATE_SCHEMA = "i01-quality-gate.v1"
REQUIRED_COMPONENTS = {
    "fragment_preservation": "fragment_status",
    "task_constraints": "task_status",
    "grammar": "grammar_status",
    "explanation_facts": "explanation_status",
    "punctuation": "punctuation_status",
}
FORBIDDEN_REFERENCE_KEYS = {
    "reference", "references", "silver", "silverreference", "testreference",
    "testreferences", "gold", "goldreference", "answerkey", "expectedanswer",
    "expected", "referencetext",
}


class MockModel(Protocol):
    model_id: str

    def generate(self, task_spec: dict[str, Any]) -> str: ...


OracleCallable = Callable[[dict[str, Any], dict[str, str]], dict[str, Any]]


def _contains_reference_fields(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            re.sub(r"[^a-z0-9]", "", key.casefold()) in FORBIDDEN_REFERENCE_KEYS
            or _contains_reference_fields(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_reference_fields(child) for child in value)
    return False


def _digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _envelope(request_id: str, task_type: str, status: str, reason_codes: list[str],
              oracle: dict[str, Any] | None, evidence: dict[str, Any], response: str | None = None) -> dict[str, Any]:
    components = {}
    coverage = {}
    oracle_version = None
    if oracle is not None:
        oracle_version = oracle.get("oracle_version")
        coverage = oracle.get("coverage", {})
        components = {name: oracle.get(field, "UNKNOWN") for name, field in REQUIRED_COMPONENTS.items()}
    result = {
        "schema_version": GATE_SCHEMA,
        "request_id": request_id,
        "task_type": task_type,
        "status": status,
        "reason_codes": sorted(set(reason_codes)),
        "response": response,
        "validation": {
            "oracle_version": oracle_version,
            "verdict": oracle.get("verdict", "NOT_RUN") if oracle else "NOT_RUN",
            "components": components,
            "coverage": coverage,
            "oracle_reason_codes": oracle.get("reason_codes", []) if oracle else [],
        },
        "evidence": evidence,
    }
    return result


def run_mock_inference(request: dict[str, Any], mock_model: MockModel,
                       oracle: OracleCallable) -> dict[str, Any]:
    """Run task envelope -> mock candidate -> Oracle -> fail-closed response."""
    request_id = request.get("request_id") if isinstance(request, dict) else None
    request_id = request_id if isinstance(request_id, str) and request_id else "invalid-request"
    task = request.get("task_spec") if isinstance(request, dict) else None
    task_type = task.get("task_type", "UNKNOWN") if isinstance(task, dict) else "UNKNOWN"
    if not isinstance(task, dict) or not isinstance(task.get("task_type"), str):
        return _envelope(request_id, task_type, "SAFE_REFUSAL", ["INVALID_TASK_ENVELOPE"], None, {})
    if _contains_reference_fields(request):
        return _envelope(request_id, task_type, "SAFE_REFUSAL", ["REFERENCE_FIELDS_FORBIDDEN"], None,
                         {"sample_id": task.get("sample_id")})
    try:
        candidate_text = mock_model.generate(deepcopy(task))
    except Exception:
        return _envelope(request_id, task_type, "SAFE_REFUSAL", ["MOCK_UPSTREAM_FAILURE"], None,
                         {"sample_id": task.get("sample_id")})
    if not isinstance(candidate_text, str) or not candidate_text.strip():
        return _envelope(request_id, task_type, "SAFE_REFUSAL", ["EMPTY_CANDIDATE"], None,
                         {"sample_id": task.get("sample_id")})
    candidate = {"response": candidate_text}
    try:
        result = oracle(task, candidate)
    except Exception:
        return _envelope(request_id, task_type, "PENDING_REVIEW", ["ORACLE_EXECUTION_FAILED"], None,
                         {"sample_id": task.get("sample_id"), "candidate_sha256": _digest(candidate_text)})
    evidence = {
        "sample_id": task.get("sample_id"),
        "task_sha256": _digest(task),
        "candidate_sha256": _digest(candidate_text),
    }
    machine_verdict = result.get("verdict")
    if machine_verdict == "FAIL":
        return _envelope(request_id, task_type, "SAFE_REFUSAL", ["ORACLE_REJECTED_CANDIDATE"], result, evidence)
    if machine_verdict != "PASS":
        code = "ORACLE_REVIEW_REQUIRED" if machine_verdict == "REVIEW_REQUIRED" else "ORACLE_UNKNOWN"
        return _envelope(request_id, task_type, "PENDING_REVIEW", [code], result, evidence)

    component_statuses = {name: result.get(field, "UNKNOWN") for name, field in REQUIRED_COMPONENTS.items()}
    coverage = result.get("coverage", {})
    uncovered = [
        name for name in REQUIRED_COMPONENTS
        if not str(coverage.get(name, "")).startswith("COMPLETE")
    ]
    nonpassing = [name for name, status in component_statuses.items() if status != "PASS"]
    if uncovered or nonpassing:
        codes = ["INSUFFICIENT_ORACLE_COVERAGE"] if uncovered else []
        codes.extend(f"COMPONENT_NOT_PASS:{name}" for name in nonpassing)
        return _envelope(request_id, task_type, "PENDING_REVIEW", codes, result, evidence)

    evidence["model_id"] = getattr(mock_model, "model_id", "MOCK")
    return _envelope(request_id, task_type, "VALIDATED", [], result, evidence,
                     response=candidate_text.strip())
