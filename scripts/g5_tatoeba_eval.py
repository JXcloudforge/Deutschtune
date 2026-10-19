"""Paired Base/LoRA inference and separate offline scoring for frozen Tatoeba Test.

The infer command never accepts or reads test references. The score command is
offline and never loads a model. Outputs are written only to new paths.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any
import unicodedata


EXPECTED_INPUT_SHA256 = "4fffe332e04621c9c33916f16e0057a498bc2b3338cc095b67154a43541ffa99"
EXPECTED_REFERENCE_SHA256 = "cd650acba3fe33ba583b240a535df13894ab823d5a0c4bc084c9a3c3b3b8e2a5"
PROMPT_VERSION = "tatoeba_g5_qwen_user_only_v1"
DECODING = {"do_sample": False, "max_new_tokens": 128, "num_beams": 1}
INFERENCE_PROTOCOL_V2 = "explicit_attention_mask_greedy_v2"
NEUTRAL_SAMPLING_CONTROLS = {"temperature": 1.0, "top_p": 1.0, "top_k": 50}
ANSWER_RE = re.compile(r"^\s*应选[“\"](?P<word>[^”\"\r\n]+)[”\"]。完整原句：(?P<sentence>[^\r\n]+)\s*$")
CHOICE_RE = re.compile(r"(?:应选|答案(?:是|为)?|选择)\s*[“\"]?(?P<word>[A-Za-zÄÖÜäöüß]+)")
SENTENCE_RE = re.compile(r"完整原句\s*[:：]\s*(?P<sentence>[^\r\n]+)")
WORD_ONLY_RE = re.compile(r"^[A-Za-zÄÖÜäöüß]+[。.!]?$")
SCORE_PARSER_VERSION = "tatoeba_choice_sentence_parser_v2"
WORD_PATTERN = r"[A-Za-zÄÖÜäöüß]+"
CHOICE_PATTERNS_V2 = tuple(re.compile(pattern) for pattern in (
    rf"(?:正确的?选项(?:应该)?是|正确答案(?:应该)?是)\s*[“\"]?(?P<word>{WORD_PATTERN})",
    rf"(?:应选|答案(?:是|为)?|选择)\s*[“\"]?(?P<word>{WORD_PATTERN})",
    rf"(?:符合|满足)[^。\n]{{0,100}}的词是\s*[“\"]?(?P<word>{WORD_PATTERN})",
    rf"只有\s*[“\"](?P<word>{WORD_PATTERN})[”\"]\s*(?:符合|满足)",
    rf"[“\"](?P<word>{WORD_PATTERN})[”\"]\s*(?:符合|满足)",
))
SENTENCE_RE_V2 = re.compile(
    r"完整(?:的)?(?:原句|句子)(?:应该)?(?:是|为)?\s*[:：]\s*[“\"]?(?P<sentence>[^\r\n”\"]+)"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"JSONL must contain objects: {path}")
    return rows


def frozen_inputs(path: Path, limit: int) -> list[dict[str, Any]]:
    if sha256_file(path) != EXPECTED_INPUT_SHA256:
        raise ValueError("frozen input SHA256 mismatch")
    rows = _read_jsonl(path)
    ids = [row.get("sample_id") for row in rows]
    if len(rows) != 1000 or len(set(ids)) != 1000 or any(not isinstance(i, str) for i in ids):
        raise ValueError("frozen Test must have exactly 1000 unique sample IDs")
    if any(set(row) != {"sample_id", "instruction", "input"}
           or not isinstance(row["instruction"], str) or not row["instruction"].strip()
           or row["input"] != "" for row in rows):
        raise ValueError("input-only Test schema violated or answer field present")
    if limit not in (100, 1000):
        raise ValueError("only the predeclared 100 and 1000 row runs are allowed")
    return sorted(rows, key=lambda row: (hashlib.sha256(row["sample_id"].encode()).hexdigest(), row["sample_id"]))[:limit]


def _norm(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())


def _parse_answer(value: str) -> dict[str, str] | None:
    match = ANSWER_RE.fullmatch(value)
    if not match:
        return None
    return {key: _norm(text) for key, text in match.groupdict().items()}


def _extract_choice(value: str) -> str | None:
    strict = _parse_answer(value)
    if strict:
        return strict["word"]
    first = next((line.strip() for line in value.splitlines() if line.strip()), "")
    if WORD_ONLY_RE.fullmatch(first):
        return _norm(first.rstrip("。.!"))
    for pattern in CHOICE_PATTERNS_V2:
        match = pattern.search(value)
        if match:
            return _norm(match.group("word"))
    match = CHOICE_RE.search(value)
    return _norm(match.group("word")) if match else None


def _extract_sentence(value: str) -> str | None:
    strict = _parse_answer(value)
    if strict:
        return strict["sentence"]
    match = SENTENCE_RE_V2.search(value)
    if match:
        return _norm(match.group("sentence").strip().strip('“”"'))
    match = SENTENCE_RE.search(value)
    if match:
        return _norm(match.group("sentence"))
    first = next((line.strip() for line in value.splitlines() if line.strip()), "")
    if first.endswith((".", "!", "?")) and not re.search(r"[\u3400-\u9fff]", first):
        return _norm(first)
    return None


def _greedy_generation_inputs(model: Any, tokenizer: Any, input_ids: Any, torch_module: Any) -> dict[str, Any]:
    """Preserve the Qwen token IDs while explicitly masking every unpadded token."""
    generation_config = copy.deepcopy(model.generation_config)
    for name, value in {**DECODING, **NEUTRAL_SAMPLING_CONTROLS}.items():
        setattr(generation_config, name, value)
    generation_config.pad_token_id = tokenizer.eos_token_id
    return {"input_ids": input_ids, "attention_mask": torch_module.ones_like(input_ids),
            "generation_config": generation_config}


def infer(test_path: Path, model_dir: Path, adapter_dir: Path | None, condition: str,
          limit: int, output: Path, expected_adapter_sha256: str | None = None) -> dict[str, Any]:
    """Run one model condition. Adapter identity is checked before loading it."""
    rows = frozen_inputs(test_path, limit)
    if output.exists() or output.with_suffix(output.suffix + ".partial").exists():
        raise FileExistsError("prediction output or partial file already exists")
    if condition == "lora":
        if not adapter_dir or not expected_adapter_sha256:
            raise ValueError("LoRA requires adapter path and audited adapter SHA256")
        adapter_hash = sha256_file(adapter_dir / "adapter_model.safetensors")
        if adapter_hash != expected_adapter_sha256.lower():
            raise ValueError("Adapter SHA256 differs from G4 audit")
    elif condition == "base":
        if adapter_dir or expected_adapter_sha256:
            raise ValueError("Base run must not load an Adapter")
        adapter_hash = None
    else:
        raise ValueError("condition must be base or lora")
    model_config_hash = sha256_file(model_dir / "config.json")
    tokenizer_hash = sha256_file(model_dir / "tokenizer.json")

    # Import only after input-only schema and identity checks. No reference path exists here.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16,
                                                device_map="auto", trust_remote_code=True)
    if condition == "lora":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
    model.eval()
    device = next(model.parameters()).device
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    with partial.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            prompt = row["instruction"]
            input_ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                      tokenize=True, add_generation_prompt=True,
                                                      return_tensors="pt").to(device)
            generation_inputs = _greedy_generation_inputs(model, tokenizer, input_ids, torch)
            with torch.inference_mode():
                generated = model.generate(**generation_inputs)
            response = tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True)
            record = {"sample_id": row["sample_id"], "condition": condition,
                      "response": response, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                      "test_inputs_sha256": EXPECTED_INPUT_SHA256, "prompt_version": PROMPT_VERSION,
                      "decoding": DECODING, "model_config_sha256": model_config_hash,
                      "tokenizer_sha256": tokenizer_hash, "adapter_sha256": adapter_hash,
                      "inference_protocol": INFERENCE_PROTOCOL_V2,
                      "attention_mask_explicit": True,
                      "inactive_sampling_controls": NEUTRAL_SAMPLING_CONTROLS}
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
    partial.rename(output)
    return {"condition": condition, "count": len(rows), "output": str(output),
            "predictions_sha256": sha256_file(output), "model_config_sha256": model_config_hash,
            "tokenizer_sha256": tokenizer_hash, "adapter_sha256": adapter_hash}


def score(test_path: Path, reference_path: Path, base_path: Path, lora_path: Path,
          limit: int, summary_path: Path, details_path: Path) -> dict[str, Any]:
    """Join complete paired predictions, then read references for offline scoring only."""
    rows = frozen_inputs(test_path, limit)
    expected_ids = {row["sample_id"] for row in rows}
    if summary_path.exists() or details_path.exists():
        raise FileExistsError("score output already exists")
    predictions: dict[str, dict[str, dict[str, Any]]] = {}
    protocols: dict[str, str] = {}
    for condition, path in (("base", base_path), ("lora", lora_path)):
        items = _read_jsonl(path)
        by_id = {item.get("sample_id"): item for item in items}
        if len(items) != limit or len(by_id) != limit or set(by_id) != expected_ids:
            raise ValueError(f"{condition} predictions are incomplete, duplicated, or use different IDs")
        for row in rows:
            item = by_id[row["sample_id"]]
            if (item.get("condition") != condition or item.get("test_inputs_sha256") != EXPECTED_INPUT_SHA256
                    or item.get("prompt_sha256") != hashlib.sha256(row["instruction"].encode()).hexdigest()
                    or item.get("prompt_version") != PROMPT_VERSION or item.get("decoding") != DECODING
                    or not isinstance(item.get("response"), str)):
                raise ValueError(f"{condition} prediction contract mismatch: {row['sample_id']}")
        identities = {(x.get("model_config_sha256"), x.get("tokenizer_sha256"), x.get("adapter_sha256"))
                      for x in items}
        if len(identities) != 1:
            raise ValueError(f"{condition} model identity varies within run")
        protocol_values = {x.get("inference_protocol", "legacy_v1_attention_mask_missing") for x in items}
        if len(protocol_values) != 1:
            raise ValueError(f"{condition} inference protocol varies within run")
        protocol = protocol_values.pop()
        if protocol not in {"legacy_v1_attention_mask_missing", INFERENCE_PROTOCOL_V2}:
            raise ValueError(f"{condition} inference protocol unknown")
        if protocol == INFERENCE_PROTOCOL_V2 and any(
            x.get("attention_mask_explicit") is not True
            or x.get("inactive_sampling_controls") != NEUTRAL_SAMPLING_CONTROLS for x in items
        ):
            raise ValueError(f"{condition} explicit mask or neutral sampling evidence missing")
        protocols[condition] = protocol
        predictions[condition] = by_id
    if protocols["base"] != protocols["lora"]:
        raise ValueError("Base/LoRA inference protocols differ")
    base_identity = next(iter(predictions["base"].values()))
    lora_identity = next(iter(predictions["lora"].values()))
    if (base_identity["model_config_sha256"] != lora_identity["model_config_sha256"]
            or base_identity["tokenizer_sha256"] != lora_identity["tokenizer_sha256"]
            or base_identity["adapter_sha256"] is not None
            or not lora_identity["adapter_sha256"]):
        raise ValueError("Base/LoRA identity or tokenizer mismatch")
    if sha256_file(reference_path) != EXPECTED_REFERENCE_SHA256:
        raise ValueError("frozen offline reference SHA256 mismatch")
    references = _read_jsonl(reference_path)
    by_ref = {row.get("sample_id"): row for row in references}
    if len(references) != 1000 or len(by_ref) != 1000 or not expected_ids.issubset(by_ref):
        raise ValueError("offline reference ID coverage mismatch")
    details = []
    counts = {condition: {"word_correct": 0, "sentence_strict": 0, "format_compliant": 0}
              for condition in ("base", "lora")}
    extraction_coverage = {condition: {"word": 0, "sentence": 0} for condition in ("base", "lora")}
    paired = {"improved_word": 0, "regressed_word": 0, "improved_sentence": 0, "regressed_sentence": 0}
    for row in rows:
        sid = row["sample_id"]
        reference = _parse_answer(by_ref[sid].get("answer", ""))
        if reference is None:
            raise ValueError(f"reference answer format invalid: {sid}")
        result = {"sample_id": sid, "instruction": row["instruction"],
                  "reference_word": reference["word"], "reference_sentence": reference["sentence"]}
        for condition in ("base", "lora"):
            response = predictions[condition][sid]["response"]
            parsed = _parse_answer(response)
            choice = _extract_choice(response)
            sentence = _extract_sentence(response)
            extraction_coverage[condition]["word"] += int(choice is not None)
            extraction_coverage[condition]["sentence"] += int(sentence is not None)
            flags = {"format_compliant": parsed is not None,
                     "word_correct": choice == reference["word"],
                     "sentence_strict": sentence == reference["sentence"]}
            for key, value in flags.items():
                counts[condition][key] += int(value)
                result[f"{condition}_{key}"] = value
            result[f"{condition}_response"] = response
        for metric in ("word_correct", "sentence_strict"):
            base_ok, lora_ok = result[f"base_{metric}"], result[f"lora_{metric}"]
            paired[f"improved_{'word' if metric == 'word_correct' else 'sentence'}"] += int(lora_ok and not base_ok)
            paired[f"regressed_{'word' if metric == 'word_correct' else 'sentence'}"] += int(base_ok and not lora_ok)
        details.append(result)
    summary = {"status": "SCORED", "scope": f"{limit}_OF_FROZEN_1000", "count": limit,
               "input_sha256": EXPECTED_INPUT_SHA256, "reference_sha256": EXPECTED_REFERENCE_SHA256,
               "base_predictions_sha256": sha256_file(base_path), "lora_predictions_sha256": sha256_file(lora_path),
               "model_config_sha256": base_identity["model_config_sha256"],
               "tokenizer_sha256": base_identity["tokenizer_sha256"],
               "adapter_sha256": lora_identity["adapter_sha256"], "prompt_version": PROMPT_VERSION,
               "decoding": DECODING, "inference_protocol": protocols["base"],
               "score_parser_version": SCORE_PARSER_VERSION,
               "counts": counts, "extraction_coverage": extraction_coverage, "paired": paired,
               "metric_definition": "NFKC plus whitespace collapse; case and punctuation retained. "
                                    "Word and sentence extraction use fixed, reference-independent patterns; "
                                    "format compliance requires the exact declared response envelope. "
                                    "References are machine-generated Silver, not human GOLD."}
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    details_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2); stream.write("\n")
    with details_path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(details[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(details)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_infer = sub.add_parser("infer")
    p_infer.add_argument("--test", type=Path, required=True)
    p_infer.add_argument("--model-dir", type=Path, required=True)
    p_infer.add_argument("--adapter-dir", type=Path)
    p_infer.add_argument("--expected-adapter-sha256")
    p_infer.add_argument("--condition", choices=("base", "lora"), required=True)
    p_infer.add_argument("--limit", type=int, choices=(100, 1000), required=True)
    p_infer.add_argument("--output", type=Path, required=True)
    p_score = sub.add_parser("score")
    p_score.add_argument("--test", type=Path, required=True)
    p_score.add_argument("--references", type=Path, required=True)
    p_score.add_argument("--base", type=Path, required=True)
    p_score.add_argument("--lora", type=Path, required=True)
    p_score.add_argument("--limit", type=int, choices=(100, 1000), required=True)
    p_score.add_argument("--summary", type=Path, required=True)
    p_score.add_argument("--details", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "infer":
        result = infer(args.test, args.model_dir, args.adapter_dir, args.condition,
                       args.limit, args.output, args.expected_adapter_sha256)
    else:
        result = score(args.test, args.references, args.base, args.lora,
                       args.limit, args.summary, args.details)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
