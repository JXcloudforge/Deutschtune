# DeutschTune data pipeline (D01)

## Historical task and split

The constrained task is German word restoration: a source sentence is transformed into a masked exercise with options/letter clues and a mechanically reconstructed answer. The admitted historical batch was 7,500 train, 1,000 validation, and 1,000 independent test rows. This repository contains none of those rows, source manifests, original sentences, or references.

The historical attribution file records 9,499 rows marked CC BY 2.0 FR and one marked CC0 1.0. The recorded data-use decision was limited to internal SFT, offline evaluation and non-public demonstration. It did not authorize public redistribution. Usernames remain source metadata and are not independently verified legal authorship.

## Included D01 code

- `data_io.py`: ShareGPT and input-only JSONL contract parsing; NFKC and whitespace normalization while preserving German case.
- `normalize_dataset.py`: writes a new normalized version and records hashes; refuses to overwrite an existing output directory.
- `validate_dataset.py`: checks declared split structure, exact normalized duplicates and group overlap when group IDs exist. It does not detect semantic or translation-graph leakage.
- `prepare_llamafactory.py`: prepares only train and validation and does not register a test split.

The original generator and 9,500-row export/admission ledgers are omitted because the repository does not redistribute corpus text and those records bind private batch identifiers/evidence. The code subset is not a replacement for the original admission decision.

## Safe use

A historical fixture lacked independent provenance evidence and is excluded. Tests construct minimal synthetic rows at runtime; they are not part of the admitted dataset. Do not place source datasets or test references in this repository. Obtain an appropriate data-use basis, preserve row-level attribution outside the public repository, and rerun D01 checks in a separately controlled copy before any new training.
