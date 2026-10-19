import json

import pytest

from scripts.data_io import DataContractError, iter_jsonl, validate_row


def test_reads_runtime_constructed_jsonl(tmp_path):
    path = tmp_path / "sample.jsonl"
    row = {"group_id": "synthetic-001", "messages": [
        {"role": "user", "content": "Setze ein Verb ein."},
        {"role": "assistant", "content": "Wir lernen."},
    ]}
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    rows = list(iter_jsonl(path))
    assert len(rows) == 1
    assert rows[0][1] == row


def test_rejects_malformed_json(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text('{"messages":\n', encoding="utf-8")
    with pytest.raises(DataContractError, match="invalid JSON"):
        list(iter_jsonl(path))


@pytest.mark.parametrize("row", [
    {"messages": [{"role": "user", "content": "Hi"}]},
    {"messages": [{"role": "user", "content": " "}, {"role": "assistant", "content": "Hi"}]},
    {"messages": [{"role": "assistant", "content": "Hi"}, {"role": "user", "content": "Hi"}]},
])
def test_rejects_invalid_input_output_contract(row):
    with pytest.raises(DataContractError):
        validate_row(row)


def test_runtime_row_contains_only_contract_fields():
    row = {"group_id": "synthetic-002", "messages": [
        {"role": "user", "content": "Ergänze den Satz."},
        {"role": "assistant", "content": "Sie spielt."},
    ]}
    assert set(row) == {"group_id", "messages"}
    assert validate_row(row) == row
