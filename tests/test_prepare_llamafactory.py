import json

import pytest

from scripts.prepare_llamafactory import prepare


def write(path, text):
    answer = {"corrected_text": text, "error_categories": ["grammar"], "explanation_zh": "解释"}
    row = {"group_id": path.stem, "messages": [{"role": "user", "content": "Ich bin hier."}, {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)}]}
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")


def test_prepares_versioned_sharegpt_without_test_registry(tmp_path):
    train, validation = tmp_path / "train.jsonl", tmp_path / "validation.jsonl"
    write(train, "Ich bin hier.")
    write(validation, "Ich bin hier.")
    target = tmp_path / "prepared" / "v1"
    manifest = prepare(train, validation, target, "a4-test-v1")
    registry = json.loads((target / "dataset_info.json").read_text(encoding="utf-8"))
    assert set(registry) == {"a4_train", "a4_validation"}
    assert manifest["test_registered"] is False
    assert manifest["prepared_rows"] == {"train": 1, "validation": 1}
    assert len(manifest["prepared_sha256"]["a4_train.jsonl"]) == 64


def test_tutor_provenance_fields_survive_sharegpt_preparation(tmp_path):
    train = tmp_path / "tutor_train.jsonl"
    validation = tmp_path / "tutor_validation.jsonl"
    write(train, "Ich lerne Deutsch.")
    write(validation, "Heute lerne ich Deutsch.")
    for path in (train, validation):
        row = json.loads(path.read_text(encoding="utf-8"))
        row.update({"sample_id":"s1", "grammar_id":"G007", "task_type":"grammar_explanation", "cefr_level":"A1", "source_ids":["GRIMM_GRAMMAR"], "generation_method":"GENERATED", "review_status":"NEEDS_REVIEW"})
        path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    output = tmp_path / "lf" / "tutor"
    prepare(train, validation, output, "tutor-v0.1")
    prepared = json.loads((output / "a4_train.jsonl").read_text(encoding="utf-8"))
    assert prepared["task_type"] == "grammar_explanation"
    assert prepared["source_ids"] == ["GRIMM_GRAMMAR"]


def test_test_input_cannot_be_prepared(tmp_path):
    test_path = tmp_path / "test.jsonl"
    test_path.write_text("{}\n", encoding="utf-8")
    validation = tmp_path / "validation.jsonl"
    write(validation, "Ich bin hier.")
    with pytest.raises(ValueError, match="test split input is forbidden"):
        prepare(test_path, validation, tmp_path / "out", "x")
