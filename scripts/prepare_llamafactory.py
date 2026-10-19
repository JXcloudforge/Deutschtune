"""Prepare separate train/validation ShareGPT files and dataset_info.json."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

try:
    from .data_io import iter_jsonl
except ImportError:  # pragma: no cover
    from data_io import iter_jsonl


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_test_path(path: Path) -> bool:
    return any(part.casefold() in {"test", "a4_test", "test.jsonl"} for part in path.parts)


def prepare(train: Path, validation: Path, output_dir: Path, dataset_version: str) -> dict[str, Any]:
    if is_test_path(train) or is_test_path(validation):
        raise ValueError("test split input is forbidden")
    if train.resolve() == validation.resolve():
        raise ValueError("train and validation inputs must be different files")
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".a4_lf_", dir=output_dir.parent))
    counts: dict[str, int] = {}
    try:
        for split, source, dest_name in (
            ("train", train, "a4_train.jsonl"),
            ("validation", validation, "a4_validation.jsonl"),
        ):
            count = 0
            dest = staging / dest_name
            with dest.open("x", encoding="utf-8", newline="\n") as stream:
                for _, row in iter_jsonl(source):
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    count += 1
            if count == 0:
                raise ValueError(f"{split} split must not be empty")
            counts[split] = count

        tags = {
            "role_tag": "role", "content_tag": "content",
            "user_tag": "user", "assistant_tag": "assistant", "system_tag": "system",
        }
        registry = {
            "a4_train": {"file_name": "a4_train.jsonl", "formatting": "sharegpt", "columns": {"messages": "messages"}, "tags": tags},
            "a4_validation": {"file_name": "a4_validation.jsonl", "formatting": "sharegpt", "columns": {"messages": "messages"}, "tags": tags},
        }
        (staging / "dataset_info.json").write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        manifest = {
            "status": "PREPARED",
            "dataset_version": dataset_version,
            "format": "llama_factory_sharegpt_jsonl",
            "source": {"train": str(train), "validation": str(validation)},
            "source_sha256": {"train": sha256(train), "validation": sha256(validation)},
            "prepared_rows": counts,
            "prepared_sha256": {name: sha256(staging / name) for name in ("a4_train.jsonl", "a4_validation.jsonl")},
            "test_registered": False,
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        staging.rename(output_dir)
    except Exception:
        # Remove only this invocation's newly-created staging directory.
        import shutil
        shutil.rmtree(staging, ignore_errors=True)
        raise
    manifest["output_dir"] = str(output_dir)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-version", required=True)
    args = parser.parse_args()
    try:
        result = prepare(args.train, args.validation, args.output_dir, args.dataset_version)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
