# DeutschTune reproduction guide

## What this candidate can verify

This repository has no Tatoeba corpus, train/validation/test rows, frozen references, Qwen weights, LoRA adapter, original GPU logs, or platform credentials. Its local tests cover synthetic data contracts, Flask input/error boundaries, and the loopback-only startup guard. They cannot reproduce the historical model scores.

In a separately created Python environment, install the declared local dependencies and run:

```powershell
python -m pip install -r requirements-dev.txt -r requirements-api.txt
python -m pytest -q -p no:cacheprovider
```

These commands were **not installed or executed in a fresh environment** during R3. The R3 test used an existing local virtual environment; the exact command and exit code are in the private release review. Tests create minimal synthetic rows at runtime. No historical fixture or frozen question is included.

## Optional local API example

After obtaining a lawful model and starting an OpenAI-compatible endpoint separately, review `configs/serving.example.json` and run from the repository root:

```powershell
python -m api.server
```

The example expects `http://127.0.0.1:8000/v1` and `tatoeba-lora`. `GET http://127.0.0.1:20995/healthz` checks the Flask process only; it does not prove the model is available. `POST /tatoeba/restore` accepts a JSON object with `request_id` and `text`. The route is unauthenticated and enforced as loopback-only; it is not a public API recipe. R3 did not execute live model inference or start a service.

`A4_SERVING_CONFIG` and related `A4_*` names are historical compatibility keys in the included code. Renaming them would change the interface and requires coordinated migration and regression testing. They are not the public project name.

## Historical workflow and missing inputs

`configs/lora_tatoeba.example.yaml` is a sanitized, non-runnable outline of the historical training parameters. `configs/dataset_info.example.json` illustrates train/validation registration only. Their `a4_*` dataset keys match the included preparation code and are retained for compatibility. The original YAML contained private paths and is excluded. A new training run requires a separately authorized dataset, pinned model revision, reviewed rights, compatible GPU environment, and fresh output paths.

The included G5 scorer separates frozen test inputs from offline Silver references, but this candidate intentionally omits the required test files and predictions. The historical 1,000-row metrics in `results/G5_PUBLIC_SUMMARY.md` are a curated aggregate, not a result produced by running this public candidate. The external-system adapter and original platform screenshot are also excluded.
