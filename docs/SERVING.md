# DeutschTune serving boundary (I01/A01)

## Included interfaces

- `scripts/openai_compatible.py` implements an OpenAI-compatible Chat Completions client.
- `api/service.py` implements `GET /healthz` and `POST /tatoeba/restore`. The POST request is JSON with `request_id` and `text`; a successful response contains `request_id`, the configured model ID, and the upstream model's actual text. Timeout and upstream errors map to 504/502.
- `api/server.py` reads `configs/serving.example.json`, creates the client, preserves the legacy Tutor `/chat` quality gate (409), and binds only to `localhost` or a loopback IP. Existing `/tutor` semantics are unchanged.
- This repository does **not** include the historical OJAC adapter or its Gunicorn WSGI entrypoint. The single external-system demonstration is a bounded historical integration claim, not a runnable feature in this repository.

## Local run contract

The example config points to `http://127.0.0.1:8000/v1` and model ID `tatoeba-lora`. A separately authorized OpenAI-compatible model endpoint is required. From the repository root, after installing `requirements-api.txt`, use `python -m api.server`; the default Flask port is 20995 on loopback. This R3 review did not start the service or connect to a model.

The Tatoeba route has **no inbound authentication**. The entrypoint rejects non-loopback bind addresses, Flask limits bodies to 256 KiB by default (maximum configurable value 1 MiB), and oversized requests return JSON 413 without calling the upstream client. These safeguards do not authorize public exposure. Do not publish this route through a reverse proxy without a separately reviewed authentication boundary.

## Historical deployment evidence

Private archived records identify vLLM 0.31.0 and historical HTTP 200 responses for model and chat routes. A local Flask-through-tunnel test recorded a `tatoeba-lora` response. A separate single-question external-system demonstration was observed, but its adapter source and screenshots are excluded pending rights review. Archived HTTP results do not establish current service uptime or production readiness. The archived vLLM PyTorch version is not established as the G4 training PyTorch version.

`requirements-serving.txt` lists Gunicorn 23.0.0 from historical deployment notes. This repository does not provide a public Gunicorn deployment command for the local Flask route; its documented executable entrypoint is the loopback-only development server.
