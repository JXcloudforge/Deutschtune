"""I01 gated Tutor request handling using the existing quality-gate boundary."""
from __future__ import annotations

import json
from typing import Any

from api.inference_quality_gate import run_mock_inference
from api.tutor_fallback import build_verified_rule_fallback


class OpenAICompatibleTutor:
    """Adapt the existing OpenAI-compatible client to I01's structured task input."""

    def __init__(self, client: Any, *, model_id: str, system_prompt: str,
                 temperature: float = 0.0, top_p: float = 1.0, max_tokens: int = 256):
        self.client = client
        self.model_id = model_id
        self.system_prompt = system_prompt
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_tokens = int(max_tokens)

    def generate(self, task_spec: dict[str, Any]) -> str:
        allowed = (
            "task_type", "grammar_id", "template_family", "level", "subject",
            "subject_person_number", "required_lemma", "required_time_phrase",
            "complement", "time_required",
        )
        safe_task = {key: task_spec[key] for key in allowed if key in task_spec}
        return self.client.chat_completion({
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": json.dumps(safe_task, ensure_ascii=False)},
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "stream": False,
        })


class _CaptureModel:
    def __init__(self, model: Any):
        self.model = model
        self.model_id = getattr(model, "model_id", "UNKNOWN_MODEL")
        self.candidate: str | None = None

    def generate(self, task_spec: dict[str, Any]) -> str:
        value = self.model.generate(task_spec)
        if isinstance(value, str):
            self.candidate = value
        return value


def run_tutor_request(request: dict[str, Any], model: Any, oracle: Any,
                      *, internal_diagnostics: bool = False) -> dict[str, Any]:
    captured = _CaptureModel(model)
    gated = run_mock_inference(request, captured, oracle)
    task = request.get("task_spec") if isinstance(request, dict) else None
    if any(code in {"REFERENCE_FIELDS_FORBIDDEN", "INVALID_TASK_ENVELOPE"}
           for code in gated.get("reason_codes", [])):
        result = {**gated, "response": None, "fallback": {
            "status": "NOT_VERIFIED", "response": None,
            "reason_codes": ["FALLBACK_BLOCKED_BY_INVALID_OR_REFERENCE_INPUT"],
        }}
        if internal_diagnostics:
            result["diagnostics"] = {"model_candidate": None, "model_candidate_released": False,
                                      "model_gate_status": gated.get("status")}
        return result
    fallback = build_verified_rule_fallback(task) if isinstance(task, dict) else {
        "status": "NOT_VERIFIED", "response": None, "reason_codes": ["INVALID_TASK_ENVELOPE"]
    }
    if fallback["status"] == "VERIFIED_RULE_FALLBACK":
        result = {
            **gated,
            "model_validation": gated.get("validation"),
            "status": "VERIFIED_RULE_FALLBACK",
            "response": fallback["response"],
            "validation": fallback["validation"],
            "reason_codes": sorted(set(gated["reason_codes"] + ["MODEL_OUTPUT_NOT_RELEASED"])),
            "fallback": {key: value for key, value in fallback.items() if key != "response"},
        }
    else:
        result = {**gated, "response": gated.get("response") if gated.get("status") == "VALIDATED" else None,
                  "fallback": {key: value for key, value in fallback.items() if key != "response"}}
    if internal_diagnostics:
        result["diagnostics"] = {
            "model_candidate": captured.candidate,
            "model_candidate_released": gated.get("status") == "VALIDATED",
            "model_gate_status": gated.get("status"),
        }
    return result
