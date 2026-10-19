# DeutschTune evaluation (E01)

## Frozen paired test

Historical local acceptance records report 1,000 paired Base/LoRA rows under the same Mask v2 input and deterministic decoding protocol. Machine-generated Silver references were used only by the offline scorer; no reference answer was part of model input. Human Gold scoring was not completed.

| Metric | Base | LoRA |
|---|---:|---:|
| Choice match | 912/1,000 (91.2%) | 1,000/1,000 (100%) |
| Full source sentence exact match | 893/1,000 (89.3%) | 1,000/1,000 (100%) |
| Required response format | 0/1,000 (0%) | 1,000/1,000 (100%) |

These numbers are copied from the verified local frozen evaluation archive and were not recalculated for this repository publication. The denominator is the fixed 1,000-row test set under the archived parser and Silver references; results do not establish general German proficiency or correctness on open-ended tasks.

## Metric meaning and limitations

- Choice match compares the extracted selected word to the Silver reference.
- Full sentence exact match is a strict string comparison after the scorer's declared normalization; legal alternate wording can be marked unmatched.
- Format compliance checks the required output envelope, not grammatical correctness by itself.
- Silver is a mechanically generated reference, not human Gold. Translation relationships and all semantic leakage risks were not fully verified.
- The 1,000 source rows, raw predictions, detailed score table, manifests, per-row hashes, and screenshots are intentionally absent.

The included `scripts/g5_tatoeba_eval.py` expects explicitly supplied frozen inputs and references with pinned hashes. It is included as actual source, but cannot run meaningfully with this repository alone.
