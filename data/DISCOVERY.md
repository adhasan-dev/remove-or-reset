# Guided discovery: sources, processing and limits

This collection helped **choose channels**, not train LoRA or measure independent
benchmark accuracy. It consists of benchmark-derived original questions plus
LLM-created variants. **Neither the benchmark content nor its variants is bundled.**
A rewritten or translated downloaded question is still source-derived content.

## Sources and how to obtain them

[`recipes/discovery_sources.json`](recipes/discovery_sources.json) contains the
exact revision, URL, relative filename, size and SHA-256 for every acquired
source file, plus historical notices. Read the later permission qualifications
below **before** using those historical license labels.

| Source | Used for | Exact source revision |
|---|---|---|
| [MMLU](https://huggingface.co/datasets/cais/mmlu) | General subject questions; 20 named validation subjects. | `c30699e8356da336a370243923dbaf21066bb9fe` |
| [ARC](https://huggingface.co/datasets/allenai/ai2_arc) | School-science questions from Easy/Challenge training splits. | `210d026faf9955653af8916fad021475a3f00453` |
| [GSM8K](https://huggingface.co/datasets/openai/gsm8k) | Arithmetic word problems; training rows 0 and 1 are demonstrations, not candidates. | `740312add88f781978c0658806c59bc2815b9866` |
| [BIG-Bench Hard](https://github.com/suzgunmirac/BIG-Bench-Hard) | Ten tasks spanning logic, ordering, counting, language and tracking. | `9ee07bd481feebf959a6b59d61ea57bdcf30964d` |
| [Natural Instructions](https://github.com/allenai/natural-instructions) | Quoref, SQuAD 1.1/2, BoolQ, WinoGrande and two WIQA tasks. | `55a365637381ce7f3748fa2eac7aef1a113bbb82` |
| [MBPP](https://huggingface.co/datasets/google-research-datasets/mbpp) and NI SNLI | Acquired and screened; **zero retained families**. | MBPP `4bb6404fdc6cacfda99d4ac4205087b89d32030c`; NI revision above. |

If lawfully entitled to use them, obtain each file from its recorded publisher
URL and save it under the recorded relative path in `work\discovery-raw`.
For example, the MMLU Parquet file belongs at
`work\discovery-raw\mmlu\all\validation-00000-of-00001.parquet`.
Do not replace a failed URL with an unverified source or treat this file list as
an automatic permission grant.

## Deterministic preparation and filtering

```powershell
python data\scripts\prepare_discovery.py --raw work\discovery-raw `
  --output work\discovery-candidates
```

This script verifies every source-content hash and replays the historical
deterministic preparation, **without model inference**:

1. Reads only the specified subjects/tasks; preserves source IDs and provenance.
2. Renders each question in its task-specific prompt and answer-label format.
3. Normalizes Unicode, case and whitespace to reject exact-input duplicates;
   rejects rendered inputs longer than 6,500 characters.
4. Groups related passages/problems, assigns groups to pilot/score/audit by
   SHA-256, and takes a fixed seed-20261024 reservoir under task quotas.
5. Checks the complete candidate-file SHA-256 against the historical pool.

The expected pool is **1,161 candidates**, not 1,161 accepted families.
Preparation excluded 243 duplicates and 15 overlength inputs.
All 227 pilot-group candidates were excluded from the 934-case main screening.
The offline helper also retains `select_supported(candidates, outcomes)` for
quota selection from externally supplied, exhaustive baseline outcomes; it
does not manufacture those outcomes.

Historical baseline screening used the untouched Gemma 3 4B PT model. Choice
tasks required agreement between label likelihood and the start of greedy
generation; other tasks used task-specific exact-answer checks. Inputs could
not exceed 2,048 tokens. Of 934 screened cases, 441 passed; quotas retained
316 and left 125 correct surplus. These are **baseline-selected calibration**
examples, not an unbiased benchmark. The selection/screening history is not
recreated merely by downloading the source datasets.

## LLM variants and later exclusions

The retained seed families were paraphrased and translated into English,
Spanish and Japanese using **Qwen3.5-9B Q5_K_M**, with separate semantic-review
calls, deterministic guards and inspection. Earlier Qwen3-4B/MADLAD pilots were
superseded. There were no newly invented seed tasks in this discovery collection.

Variants retained their family, source and group partition. The original task,
answer requirements and supplied evidence had to remain equivalent. Certain
translation categories were excluded or required further review. The final
inspection held 43 translations, leaving **1,768 retained prompts from 316
families**. Pilot/development exposure means the discovery "audit" partition
must not be described as a pristine final holdout.

The final **filtering** step is reproducible without LLM calls if the exact
1,811 pre-inspection variants are separately available. The included
`recipes/variant_filter.json` records only the 43 held IDs and input/output
hashes, not the questions or translations:

```powershell
python data\scripts\filter_variants.py --input "<local pre-inspection variants.jsonl>" `
  --output work\inspected-variants
```

It preserves accepted records unchanged and requires the exact 1,768-row output
hash. The recipe replays recorded decisions; it does not independently certify
translation quality or re-run the original semantic review.

**Variant-generation scripts are deliberately not included.** Exact frozen
variants are not obtainable simply by rerunning an LLM: generation, review and
human inspection are part of their provenance. The generated variants inherit
source-rights constraints; Qwen's Apache license does not clear benchmark rights.

If the exact artifacts become lawfully available, check them with:

```powershell
python data\scripts\validate_discovery.py --families "<local families.jsonl>" `
  --variants "<local variants.jsonl>"
```

Expected SHA-256:

| Artifact | SHA-256 |
|---|---|
| 316 families | `f1fb7b13e1cd278a185ded24228b49b3bf722f76365cac3a346c8e974cf52096` |
| 1,768 variants | `729780f59b7e3f6d67ea94364e15a986209c111d526acf961cfe500a75b2e288` |

Guided scoring additionally uses frozen calibration/protection and numerical
prior inputs described in [`../docs/METHODS.md`](../docs/METHODS.md).
Those inputs are not reconstructed by candidate preparation. The existing
selector-regeneration code is not shipped. Exact guided-selector regeneration is
**not self-contained in this release**; frozen-mask retraining remains available.

## Permission qualifications

The following qualifications apply in addition to the license labels in the
original acquisition metadata:

- ARC's publisher archive specifies noncommercial research/no redistribution,
  conflicting with a later CC BY-SA mirror label. Clarification is needed.
- MMLU's MIT notice did not establish rights to republish every originating
  exam/course/book question.
- Natural Instructions' Apache grant covers definitions, **not all instances**.
  Quoref/SQuAD/BoolQ passage attribution and originating rights matter.
- WinoGrande's original data license is **CC BY 2.0**, not the inherited Apache
  label. WIQA's original data-distribution permission was not established.
- GSM8K has MIT support; BBH requires its MIT notice plus underlying BIG-bench
  and task-specific notices.

No subset is silently substituted for the frozen collection. Hash verification
confirms identity, not permission. This documentation describes historical use
and conservative release boundaries, not legal advice or a blanket finding that
all research use was prohibited.
