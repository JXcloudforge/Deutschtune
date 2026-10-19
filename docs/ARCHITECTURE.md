# DeutschTune architecture

**Snapshot status:** R3 publication candidate, prepared 2026-10-19. This describes selected code and historical evidence; current service uptime is not verified.

## Components and contracts

| Layer | Candidate files | Responsibility | Boundary |
|---|---|---|---|
| D01 | `scripts/data_io.py`, `normalize_dataset.py`, `validate_dataset.py`, `prepare_llamafactory.py` | Validate ShareGPT JSONL, distinguish input-only records, normalize into a new version, detect exact input/group overlap, prepare train/validation registry | Does not establish German quality, legal rights, or semantic non-leakage; test references are not included. |
| T01 | `configs/lora_tatoeba.example.yaml` | Sanitized description of historical LoRA settings | Derived template only; paths are placeholders, not verified runnable config. |
| E01 | `scripts/g5_tatoeba_eval.py` | Deterministic paired inference interface and separate offline scoring | Frozen test inputs/references, predictions and GPU weights are withheld; this source alone cannot reproduce scores. |
| I01 | `scripts/openai_compatible.py` | Small synchronous and streaming client for OpenAI-compatible Chat Completions | Calls a configured endpoint; credentials must come from environment/config outside source control. |
| A01 | `api/service.py`, `api/server.py` | Flask health/Tatoeba routes and guarded Tutor routes | The legacy Tutor `/chat` gate remains enforced. |
| External integration | Source intentionally excluded | A single historical external-system response is recorded privately | Platform adapter redistribution rights and current service state are not verified in this candidate. |
| R01 | `docs/`, `results/` | Public-safe method/results summaries and limitations | No private raw receipts or original paths are copied. |

## Data flow

Approved source sentence → D01 exercise generation/admission (the original generation/admission implementation is not included) → train/validation JSONL → T01 LoRA training. Frozen test inputs are sent to Base and LoRA without references; predictions are scored offline against machine Silver references. Runtime requests pass from Flask to an OpenAI-compatible model endpoint. The historical external-system adapter was separate from the legacy Tutor route; its source is not included in this repository.

## No invented infrastructure

The candidate has no database, queue, user account system, orchestration cluster, or production monitoring layer. The historical GPU service used loopback vLLM and a separate Flask/Gunicorn process; the public copy contains no deployment secrets or live endpoints.
