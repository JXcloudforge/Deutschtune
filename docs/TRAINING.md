# DeutschTune training (T01)

## Historical run summary

- Base: Qwen2.5-7B-Instruct.
- Framework: LLaMA-Factory 0.9.5.
- Hardware/runtime: NVIDIA A800 80GB; BF16 LoRA. The GPU evidence package records host Python 3.10.12 and an activated environment named `a4_py311`, but no post-activation `python --version` receipt. The launch script checks LLaMA-Factory 0.9.5; the training log records Transformers 4.57.1. The exact training interpreter and PyTorch version are not verified by the archived package.
- Data: 7,500 train and 1,000 validation rows; the independent 1,000-row test was not registered for training.
- LoRA: rank 8, alpha 16, dropout 0.05, target `all`; Qwen template; cutoff 512; micro-batch 1; gradient accumulation 8; learning rate 1e-4; two epochs; seed 20261017.
- Recorded run: 1,876 optimization steps, train loss 0.0196, eval loss 0.0011, exit code 0.

These are historical records in the private project archive and are not rerun or independently executed for this repository publication. The full original YAML contains private filesystem locations and was not copied. `configs/lora_tatoeba.example.yaml` is a sanitized derived template; it uses placeholders and is not a runnable historical snapshot. Model revision/weight hash evidence and adapter weights are not distributed.

## Reproduction boundary

A new run requires separately authorized data, a pinned model revision, reviewed model/data terms, compatible GPU libraries, and a newly reviewed output directory. Do not infer quality from loss alone. The repository is not a complete training package and does not bundle LLaMA-Factory or model dependencies.
