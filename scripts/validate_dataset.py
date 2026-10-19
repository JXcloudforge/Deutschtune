"""Read-only audit of A4 train/validation/test JSONL files."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

try:  # support both `python scripts/validate_dataset.py` and package imports
    from .data_io import DataContractError, input_text, iter_input_jsonl, iter_jsonl, normalized_input
except ImportError:  # pragma: no cover - direct script execution
    from data_io import DataContractError, input_text, iter_input_jsonl, iter_jsonl, normalized_input


def audit(paths: dict[str, Path], expected_counts: dict[str, int] | None = None) -> dict[str, Any]:
    report: dict[str, Any] = {"status": "PASS", "splits": {}, "issues": [], "duplicate_inputs": [], "cross_split_input_duplicates": [], "cross_split_group_overlaps": []}
    all_inputs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_groups: dict[str, set[str]] = {}
    all_group_rows_have_id = True

    for split, path in paths.items():
        if not path.is_file():
            report["splits"][split] = {"status": "NOT RUN", "path": str(path), "count": None, "sha256": None}
            report["issues"].append(f"{split}: file not found")
            report["status"] = "NOT RUN"
            continue
        source_hash = None
        rows: list[dict[str, Any]] = []
        try:
            source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            reader = iter_input_jsonl(path) if split == "test" else iter_jsonl(path)
            for line_number, row in reader:
                rows.append(row)
                user_text = input_text(row)
                all_inputs[normalized_input(user_text)].append({"split": split, "line": line_number})
            if not rows:
                raise DataContractError("dataset split must contain at least one row")
            groups = {row["group_id"] for row in rows if "group_id" in row}
            missing_group_rows = sum("group_id" not in row for row in rows)
            if missing_group_rows:
                all_group_rows_have_id = False
            all_groups[split] = groups
            report["splits"][split] = {
                "status": "PASS", "path": str(path), "count": len(rows),
                "sha256": source_hash,
                "group_ids_present": len(groups), "group_ids_missing": missing_group_rows,
            }
            if expected_counts and split in expected_counts:
                report["splits"][split]["expected_count"] = expected_counts[split]
                report["splits"][split]["count_matches_expected"] = len(rows) == expected_counts[split]
                if len(rows) != expected_counts[split]:
                    report["splits"][split]["status"] = "FAIL"
                    report["issues"].append(f"{split}: expected {expected_counts[split]} rows, found {len(rows)}")
                    report["status"] = "FAIL"
        except (OSError, DataContractError) as exc:
            report["splits"][split] = {"status": "FAIL", "path": str(path), "count": None, "sha256": source_hash}
            report["issues"].append(f"{split}: {exc}")
            report["status"] = "FAIL"

    for normalized, occurrences in all_inputs.items():
        if len(occurrences) > 1:
            report["duplicate_inputs"].append({"occurrences": occurrences})
        splits = sorted({item["split"] for item in occurrences})
        if len(splits) > 1:
            report["cross_split_input_duplicates"].append({"splits": splits, "occurrences": occurrences})
    if report["duplicate_inputs"]:
        report["issues"].append("exact normalized user-input duplicates found within or across splits")
        report["status"] = "FAIL"

    if all_group_rows_have_id and len(all_groups) == 3:
        split_names = list(all_groups)
        for index, left in enumerate(split_names):
            for right in split_names[index + 1:]:
                overlap = sorted(all_groups[left] & all_groups[right])
                if overlap:
                    report["cross_split_group_overlaps"].append({"splits": [left, right], "group_ids": overlap})
        if report["cross_split_group_overlaps"]:
            report["issues"].append("group_id values overlap across splits")
            report["status"] = "FAIL"
        report["group_check"] = "PASS" if not report["cross_split_group_overlaps"] else "FAIL"
    else:
        report["group_check"] = "NOT CHECKED: group_id missing or one or more split files unavailable"

    if any(item["status"] == "FAIL" for item in report["splits"].values()):
        report["status"] = "FAIL"
    elif any(item["status"] == "NOT RUN" for item in report["splits"].values()) and report["status"] != "FAIL":
        report["status"] = "NOT RUN"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--expected-train", type=int, default=18)
    parser.add_argument("--expected-validation", type=int, default=4)
    parser.add_argument("--expected-test", type=int, default=10)
    parser.add_argument("--no-count-check", action="store_true", help="Disable the R2 18/4/10 source-package count expectation")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    expected = None if args.no_count_check else {"train": args.expected_train, "validation": args.expected_validation, "test": args.expected_test}
    report = audit({"train": args.train, "validation": args.validation, "test": args.test}, expected_counts=expected)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["status"] == "PASS" else 2 if report["status"] == "NOT RUN" else 1


if __name__ == "__main__":
    sys.exit(main())
