# DeutschTune historical G5 aggregate

**Evidence status:** summarized from the source project's locally verified frozen 1,000-row acceptance archive; not recomputed for this repository publication. The R1-B remote evidence package did not contain a second set of raw inference logs or predictions.

| Metric (machine Silver) | Base | LoRA |
|---|---:|---:|
| Selected word match | 912 / 1,000 | 1,000 / 1,000 |
| Full original sentence strict match | 893 / 1,000 | 1,000 / 1,000 |
| Required response format | 0 / 1,000 | 1,000 / 1,000 |

## Interpretation

Base and LoRA were compared on the same frozen test inputs and decoding protocol; answer references were used only by the offline scorer. “Strict match” is exact according to the archived normalization/parser, not a complete assessment of linguistic correctness. Machine Silver references are not human Gold. Results are specific to this dataset, task, prompt, parser and run; they do not support claims of general German proficiency.

The source evidence includes raw model predictions, test inputs and references, detailed row scores, and private manifests. None are included here. The aggregate numbers are historical local evidence and were not independently recalculated for this repository publication.
