"""Local-only Flask server entry point; never binds publicly by default."""

from __future__ import annotations

import json
import ipaddress
import os
from pathlib import Path

from api.service import create_app
from api.tutor_gateway import OpenAICompatibleTutor
from scripts.openai_compatible import OpenAICompatibleClient
from scripts.e01_oracle import assess


def validate_loopback_host(host: str) -> str:
    """Reject configuration that would expose the unauthenticated demo API."""
    if not isinstance(host, str) or not host.strip():
        raise ValueError("serving host must be localhost or a loopback IP address")
    normalized = host.strip().lower()
    if normalized == "localhost":
        return host
    try:
        address = ipaddress.ip_address(normalized)
    except ValueError as exc:
        raise ValueError("serving host must be localhost or a loopback IP address") from exc
    if not address.is_loopback:
        raise ValueError("unauthenticated demo server may bind only to loopback")
    return host


def main() -> None:
    config_path = Path(os.environ.get("A4_SERVING_CONFIG", "configs/serving.example.json"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    host = validate_loopback_host(config.get("host", "127.0.0.1"))
    key_env = config.get("api_key_env")
    api_key = os.environ.get(key_env) if key_env else None
    client = OpenAICompatibleClient(
        config["upstream_base_url"], timeout_seconds=config["timeout_seconds"], api_key=api_key
    )
    tutor_config = config.get("tutor", {})
    tutor_model = OpenAICompatibleTutor(
        client,
        model_id=config["served_model_name"],
        system_prompt=tutor_config.get(
            "system_prompt",
            "Complete only the supplied supported A1 German task. Return a complete German sentence on the first line and a concise fact-based Chinese explanation on the second line. Do not add, omit, or replace required content.",
        ),
        temperature=tutor_config.get("temperature", 0.0),
        top_p=tutor_config.get("top_p", 1.0),
        max_tokens=tutor_config.get("max_tokens", 256),
    )
    config["require_tutor_gate"] = True
    app = create_app(config, client, tutor_model=tutor_model, tutor_oracle=assess)
    app.run(host=host, port=int(config["port"]), threaded=True)


if __name__ == "__main__":
    main()
