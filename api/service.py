"""Flask /chat endpoint with request validation and injectable model client."""

from __future__ import annotations

import logging
import re
import time
import uuid
from typing import Any, Protocol

from flask import Flask, Response, request

from api.response_adapters import serialize
from scripts.openai_compatible import InferenceHTTPError, InferenceTimeout


class ChatClient(Protocol):
    def chat_completion(self, payload: dict[str, Any], *, request_id: str | None = None) -> str: ...


def create_app(config: dict[str, Any], client: ChatClient, *, tutor_model: Any = None,
               tutor_oracle: Any = None) -> Flask:
    app = Flask(__name__)
    max_request_bytes = config.get("max_request_bytes", 262144)
    if (not isinstance(max_request_bytes, int) or isinstance(max_request_bytes, bool)
            or not 1 <= max_request_bytes <= 1048576):
        raise ValueError("max_request_bytes must be an integer between 1 and 1048576")
    app.config["MAX_CONTENT_LENGTH"] = max_request_bytes
    app.config["A4_CONFIG"] = config
    logger = logging.getLogger("a4.api")

    def response(payload: dict[str, Any], code: int, fmt: str) -> Response:
        body, mimetype = serialize(payload, fmt)
        return Response(body, status=code, mimetype=mimetype)

    @app.errorhandler(413)
    def request_too_large(_error: Any) -> Response:
        return response({"error": "REQUEST_TOO_LARGE"}, 413, "json")

    @app.get("/healthz")
    def health() -> Response:
        return response({"status": "ok"}, 200, "json")

    @app.post("/tatoeba/restore")
    def tatoeba_restore() -> Response:
        started = time.perf_counter()
        if not request.is_json:
            return response({"error": "Content-Type must be application/json"}, 400, "json")
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return response({"error": "request body must be a JSON object"}, 400, "json")

        request_id = body.get("request_id")
        if (not isinstance(request_id, str) or not request_id.strip() or len(request_id) > 128
                or not re.fullmatch(r"[A-Za-z0-9._:-]+", request_id)):
            return response({"error": "request_id is required and must be 1-128 safe characters"}, 400, "json")
        text = body.get("text")
        if not isinstance(text, str) or not text.strip() or len(text) > int(config.get("max_input_chars", 12000)):
            return response({"request_id": request_id,
                             "error": "text is required, non-empty, and within the configured size limit"}, 400, "json")

        tatoeba_config = config.get("tatoeba", {})
        model_name = tatoeba_config.get("model_name", config.get("served_model_name"))
        if not isinstance(model_name, str) or not model_name.strip():
            return response({"request_id": request_id, "error": "TATOEBA_MODEL_NOT_CONFIGURED"}, 503, "json")
        payload = {
            "model": model_name,
            # G5 uses a single user prompt; do not prepend Tutor/correction instructions.
            "messages": [{"role": "user", "content": text}],
            "temperature": float(tatoeba_config.get("temperature", 0.0)),
            "top_p": float(tatoeba_config.get("top_p", 1.0)),
            "max_tokens": int(tatoeba_config.get("max_tokens", 128)),
            "stream": False,
        }
        try:
            answer = client.chat_completion(payload, request_id=request_id)
        except InferenceTimeout:
            logger.warning("request_id=%s route=tatoeba_restore status=upstream_timeout", request_id)
            return response({"request_id": request_id, "error": "model service timed out"}, 504, "json")
        except InferenceHTTPError:
            logger.warning("request_id=%s route=tatoeba_restore status=upstream_error", request_id)
            return response({"request_id": request_id, "error": "model service unavailable"}, 502, "json")
        except Exception:
            logger.error("request_id=%s route=tatoeba_restore status=internal_error", request_id)
            return response({"request_id": request_id, "error": "model service unavailable"}, 502, "json")

        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        logger.info("request_id=%s route=tatoeba_restore status=ok elapsed_ms=%s", request_id, elapsed_ms)
        return response({"request_id": request_id, "model": model_name, "response": answer}, 200, "json")

    @app.post("/chat")
    def chat() -> Response:
        started = time.perf_counter()
        fmt = "json"
        request_id = ""
        if not request.is_json:
            return response({"error": "Content-Type must be application/json"}, 400, fmt)
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return response({"error": "request body must be a JSON object"}, 400, fmt)
        if config.get("require_tutor_gate", False):
            return response({"error": "UNSTRUCTURED_CHAT_DISABLED_QUALITY_GATE_REQUIRED"}, 409, "json")
        fmt = body.get("response_format", "json")
        if not isinstance(fmt, str) or fmt not in {"json", "ndjson"}:
            return response({"error": "response_format must be json or ndjson"}, 400, "json")
        request_id = body.get("request_id")
        text = body.get("text")
        if not isinstance(request_id, str) or not request_id.strip() or len(request_id) > 128 or not re.fullmatch(r"[A-Za-z0-9._:-]+", request_id):
            request_id = str(uuid.uuid4())
            return response({"request_id": request_id, "error": "request_id is required and must be 1-128 safe characters"}, 400, fmt)
        if not isinstance(text, str) or not text.strip() or len(text) > int(config.get("max_input_chars", 12000)):
            return response({"request_id": request_id, "error": "text is required, non-empty, and within the configured size limit"}, 400, fmt)

        payload = {
            "model": config["served_model_name"],
            "messages": [
                {"role": "system", "content": config["system_prompt"]},
                {"role": "user", "content": text},
            ],
            "temperature": float(config.get("temperature", 0.0)),
            "top_p": float(config.get("top_p", 1.0)),
            "max_tokens": int(config.get("max_tokens", 256)),
            "stream": False,
        }
        try:
            answer = client.chat_completion(payload, request_id=request_id)
        except InferenceTimeout:
            logger.warning("request_id=%s status=upstream_timeout", request_id)
            return response({"request_id": request_id, "error": "model service timed out"}, 504, fmt)
        except InferenceHTTPError:
            logger.warning("request_id=%s status=upstream_error", request_id)
            return response({"request_id": request_id, "error": "model service unavailable"}, 502, fmt)
        except Exception:
            logger.error("request_id=%s status=internal_error", request_id)
            return response({"request_id": request_id, "error": "model service unavailable"}, 502, fmt)

        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        logger.info("request_id=%s status=ok elapsed_ms=%s", request_id, elapsed_ms)
        return response({"request_id": request_id, "model": config["served_model_name"], "response": answer}, 200, fmt)

    @app.post("/tutor")
    def tutor() -> Response:
        if not request.is_json:
            return response({"error": "Content-Type must be application/json"}, 400, "json")
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return response({"error": "request body must be a JSON object"}, 400, "json")
        request_id = body.get("request_id")
        if not isinstance(request_id, str) or not request_id.strip() or len(request_id) > 128 or not re.fullmatch(r"[A-Za-z0-9._:-]+", request_id):
            return response({"error": "request_id is required and must be 1-128 safe characters"}, 400, "json")
        if not isinstance(body.get("task_spec"), dict):
            return response({"request_id": request_id, "error": "task_spec must be an object"}, 400, "json")
        if tutor_model is None:
            return response({"request_id": request_id, "error": "TUTOR_MODEL_NOT_CONFIGURED"}, 503, "json")
        if tutor_oracle is None:
            from scripts.e01_oracle import assess as tutor_oracle_fn
        else:
            tutor_oracle_fn = tutor_oracle
        from api.tutor_gateway import run_tutor_request
        result = run_tutor_request(
            body, tutor_model, tutor_oracle_fn,
            internal_diagnostics=bool(config.get("internal_diagnostics", False)),
        )
        result.setdefault("request_id", request_id)
        status_code = {"VALIDATED": 200, "VERIFIED_RULE_FALLBACK": 200,
                       "PENDING_REVIEW": 202, "SAFE_REFUSAL": 422}.get(result.get("status"), 500)
        logger.info("request_id=%s route=tutor gate_status=%s", request_id, result.get("status"))
        return response(result, status_code, "json")

    return app
