"""Validate and re-serialize source splits into a separate versioned directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

try:
    from .data_io import iter_input_jsonl, iter_jsonl
except ImportError:  # pragma: no cover
    from data_io import iter_input_jsonl, iter_jsonl


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_splits(inputs: dict[str, Path], output_dir: Path, version: str) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".a4_normalized_", dir=output_dir.parent))
    split_meta: dict[str, Any] = {}
    try:
        for split, source in inputs.items():
            count = 0
            dest = staging / f"{split}.jsonl"
            reader = iter_input_jsonl(source) if split == "test" else iter_jsonl(source)
            with dest.open("x", encoding="utf-8", newline="\n") as stream:
                for _, row in reader:
                    stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    count += 1
            if not count:
                raise ValueError(f"{split} split is empty")
            split_meta[split] = {
                "count": count, "source_path": str(source),
                "source_sha256": file_hash(source), "normalized_sha256": file_hash(dest),
            }
        manifest = {
            "status": "NORMALIZED", "dataset_version": version,
            "schema_version": "a4-de-correction-v0.2", "format": "sharegpt_train_validation_and_input_only_test_jsonl",
            "splits": split_meta, "test_is_separate": True, "test_contains_reference_answers": False,
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        staging.rename(output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    manifest["output_dir"] = str(output_dir)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-version", required=True)
    args = parser.parse_args()
    try:
        result = normalize_splits({"train": args.train, "validation": args.validation, "test": args.test}, args.output_dir, args.dataset_version)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
