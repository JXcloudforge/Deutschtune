# DeutschTune public-safe evidence index

This index references only files present in this candidate. Private original logs, data, model artifacts, prediction rows and platform screenshots are withheld.

| Claim | Included evidence | Evidence level and limit |
|---|---|---|
| D01 validation and train/validation preparation code | `scripts/data_io.py`, `scripts/validate_dataset.py`, `scripts/prepare_llamafactory.py` | Source and local synthetic tests; original generation/admission implementation and data are not included. |
| Historical G4 configuration and metrics | `docs/TRAINING.md`, sanitized `configs/lora_tatoeba.example.yaml` | Curated from private archive. Original YAML/log and Adapter weight are excluded; exact training Python/PyTorch runtime is partly unverified. |
| Frozen 1,000-row Base/LoRA comparison | `results/G5_PUBLIC_SUMMARY.md`, `docs/EVALUATION.md` | Aggregate from locally archived machine Silver scores; not rerun for this repository publication and not human Gold. |
| I01/A01 service implementation | `scripts/openai_compatible.py`, `api/service.py`, `api/server.py`, `docs/SERVING.md` | Source and mock-client tests. Historical vLLM HTTP 200 does not prove current uptime; no model endpoint was contacted in R3. |
| External-system single-question demonstration | `README.md`, `docs/SERVING.md` | Historical bounded statement only. Adapter source and original screenshot are excluded pending rights clearance. |
| Third-party terms | `docs/THIRD_PARTY_NOTICES.md` | References only; no third-party corpus, model weights, framework source, or license grant included. |

A GPU-side recorded Adapter weight SHA256 exists in the private archive, but the weight bytes were not downloaded and rehashed locally. The limited remote evidence package corroborates G4 logs/configuration and historical G6 HTTP routes; it does not include a full G5 remote inference receipt or a current live service check. Full public-repository training and deployment reproduction remains unverified.
