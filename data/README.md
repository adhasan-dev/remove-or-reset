# Data for *Remove or Reset?*

**Only project-created datasets are included here.** Anything downloaded from
another publisher stays outside this folder, including filtered records, rewritten
questions, translations, and tokenized copies. Readers obtain those sources
themselves and rebuild the required inputs in an ignored `work` directory.
The numerical masks and recipes are included; they contain no downloaded questions
or model weights.

## Which data were used for what?

There are three different jobs: **choose channels**, **train the model**, and
**evaluate the trained model**. Data used to choose channels are not independent
evaluation data. A "panel" just means a fixed, smaller set of examples.

| Dataset or panel | Purpose in the study | Included here? |
|---|---|---|
| Gretel + Spider prepared **training** split: 62,817 examples | Teach every LoRA arm text-to-SQL using the same full epoch. | No; rebuild from upstream sources. |
| Prepared **selection** split: 8,389 SQL examples | Supply SQL examples for channel-selection calibration, not LoRA training. | No; rebuild. |
| Prepared **validation** split: 7,553 SQL examples | Separate SQL validation pool; a fixed 256-example panel measures target-token prediction loss. | No; rebuild. |
| Spider development: all 1,034 examples; **128 selected** across 20 databases | Primary evaluation: compare generated SQL with reference-query execution results. | No questions, queries, schemas, or databases bundled. |
| Original relation-graph panel: **128 questions over 64 graphs** | Diagnostic evaluation: follow one or two supplied relations. | Yes: [`original/reasoning.jsonl`](original/reasoning.jsonl). |
| Taylor calibration: **614 rows** across pilot/score/audit panels | Choose channels by local loss sensitivity. Includes 264 SQL rows plus 350 original auxiliary rows. Only the score panel determines the ranking; recall probes are excluded from that ranking. | Only original auxiliary rows: [`original/taylor_auxiliary.json`](original/taylor_auxiliary.json). SQL is rebuilt locally. |
| Guided discovery collection: **316 families / 1,768 variants** | Identify broadly responsive channels for protection. Broader guided calibration then scores candidate interventions. | No; all source-derived text, variants and reversible token plans excluded. See [DISCOVERY.md](DISCOVERY.md). |
| Original broad evaluation: **120 questions, ten domains** | Secondary short-answer screen of capabilities after adaptation. Not a training or channel-selection dataset. | Yes: [`heldout`](heldout). AI-assisted original authoring, not copied benchmark questions. |
| Four frozen numerical masks | Specify exactly which channels each intervention changes. Removal and reset share a mask. | Yes: [`masks`](masks). These are experiment specifications, not datasets of questions. |

The original Taylor auxiliary probes cover relation following, supplied
chemical-element facts, factual recall, and simple copying/formatting.
Related questions stay together in pilot, score, or audit partitions. They do
**not** reuse the final relation-graph evaluation panel. The score panel contains
128 SQL, 21 direct-lookup, 21 two-step, 42 supplied-fact, 21 formatting, and
42 recall rows; excluding recall leaves 233 ranking examples.

The 120-question panel contains prompts, separate references, a judge rubric,
and 23 factual-source citation bindings. References are for evaluation, never
fed to the model as part of the prompt. The citations and short original
support summaries are included, not the cited webpages. Source/family grouping
is preserved for analysis. "Held-out" does not guarantee freedom from pretraining
exposure or judge error.

**Mask naming:** this checkout contains `random`, `taylor`, `guided`, and
`quota_free_guided`. Each mask selects 69,632 channel coordinates across 34
layers. The first three select 2,048 per layer. Quota-free guided globally
selects the lowest equal-stratum raw content sensitivities outside the same
1,100 protected coordinates, with frozen per-layer counts ranging from 0 to
7,905.

## Folder map

| Path | Meaning |
|---|---|
| `original/` | Original graph evaluation and original Taylor auxiliary calibration. |
| `heldout/` | Original AI-assisted broad evaluation inputs, references, rubric and citations. |
| `masks/` | Exact frozen numerical intervention masks. |
| `recipes/evaluation.json` | Ordered source-record IDs for the SQL/loss panels; database hashes and archive members. No source text. |
| `recipes/taylor.json` | Ordered SQL calibration IDs, metadata, and expected token-entry checksum. No SQL text or token IDs. |
| `recipes/discovery_sources.json` | Pinned discovery source URLs, filenames, hashes and historical acquisition metadata. License labels must be read with the later corrections in DISCOVERY.md. |
| `scripts/` | Offline SQL preparation, discovery candidate filtering, runtime assembly, tokenization and validation. No LLM variant-generation scripts. |
| `manifest.json` | Checksums for shipped original data and numerical metadata. |
| `external_manifest.json` | Expected checksums/counts for downloaded-derived inputs rebuilt locally; **not** a list of bundled files. |
| `acquisition.json` | SQL acquisition pins, official archive identity, and excluded guided-artifact hashes. |

Machine-readable paths use `/` for portability; the PowerShell commands below
use Windows path separators. All commands run from the repository root.

## 1. Check the included originals

Requires Python 3.11 and no model, GPU, downloads, or optional packages:

```powershell
python data\scripts\validate.py
```

Success prints `"status": "verified"`. It checks manifest hashes, graph answers,
original Taylor rows, the broad-panel contract, numerical masks, and unexpected
JSON data files. Checksums confirm identity, not legal permission or scientific
correctness of every question.

## 2. Obtain the SQL sources yourself

Review the publishers' terms before acquisition. Save these **five files** in
`work\raw`, with exactly the filenames below:

| Local filename | Publisher and pinned revision |
|---|---|
| `gretel-train.parquet` | [Gretel synthetic_text_to_sql](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql/tree/740ab236e64503fba51be1101df7a1be83bf455d), `synthetic_text_to_sql_train.snappy.parquet` |
| `gretel-test.parquet` | Same revision, `synthetic_text_to_sql_test.snappy.parquet` |
| `spider-train.parquet` | [Spider mirror](https://huggingface.co/datasets/xlangai/spider/tree/0c350918f3f29ec754f1181c65cdce76cd6c133c), `spider/train-00000-of-00001.parquet` |
| `spider-validation.parquet` | Same revision, `spider/validation-00000-of-00001.parquet` |
| `spider-official-archive.zip` | Obtain the original database archive through the [official Spider page](https://yale-lily.github.io/spider); use the archive identified in `acquisition.json`. |

Exact download URLs, byte sizes, revisions and SHA-256 values are in
[`acquisition.json`](acquisition.json). The official archive SHA-256 is
`00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b`.
If the publisher changes it, do not bypass the check or substitute a mirror.
Keep the original archive and its notices, even though processing extracts only
schemas/database files. SQL preparation needs the archive's **full database
collection**, not just the 20 final evaluation databases.

For the four revision-pinned Parquet files, this explicit optional PowerShell
command downloads the URLs recorded in the acquisition metadata. It refuses to
replace existing files:

```powershell
New-Item -ItemType Directory -Force work\raw | Out-Null
$sources = (Get-Content data\acquisition.json -Raw | ConvertFrom-Json).source_pins.prepared_source_files
foreach ($source in $sources) {
    $path = "work\raw\$($source.id).parquet"
    if (Test-Path $path) { throw "Already exists: $path" }
    Invoke-WebRequest -Uri $source.url -OutFile $path
}
# Acquire spider-official-archive.zip separately from the official publisher.
python data\scripts\validate.py --raw work\raw
```

Gretel data use Apache-2.0; Spider records use CC BY-SA 4.0. SQLite source-rights
questions remain unresolved: reader acquisition is **not** permission to
redistribute database payloads. Nothing in our original-data license overrides
upstream terms.

## 3. Rebuild and validate the SQL splits

Use the project environment or a dedicated Python 3.11 environment:

```powershell
python -m pip install -r data\scripts\requirements.txt
python data\scripts\prepare_sql.py --raw work\raw --output work\prepared-sql
python data\scripts\validate.py --prepared work\prepared-sql
```

The output directory must be new. No processing command downloads a model or
dataset, and no existing prepared output is overwritten. Failed/partial outputs
are not certified; use a different directory after investigating a failure.

Preparation does the following, in order:

1. Verifies all five acquired files against pinned hashes and sizes.
2. Extracts Spider schemas and databases locally. It reads databases without
   modifying them.
3. Parses read-only SQL with SQLGlot; tries SQLite, MySQL, PostgreSQL, T-SQL,
   then BigQuery dialects. Gretel queries are converted to SQLite, and permitted
   schema/fixture statements are evaluated in isolated in-memory databases.
   Queries that cannot be executed are rejected from training candidates.
4. Serializes canonical schemas and fixed `Schema / Question / SQL:` prompts.
   Spider reference SQL is preserved rather than rewritten in the output.
   All 1,034 Spider development rows are retained, with explicit processing
   failure flags where necessary.
5. Removes exact question/schema/SQL duplicates, with Spider training taking
   priority over Gretel; excludes schemas overlapping Spider development.
6. Assigns whole schemas to 80/10/10 train/selection/validation pools using the
   frozen SHA-256 namespace `remove-or-reset-v1-seed-20261011`. Sorts rows by ID.
7. Requires **exact historical uncompressed hashes**, expected counts, unique
   IDs and no schema overlap across the four splits.

Expected counts are 62,817 / 8,389 / 7,553 / 1,034. The original runtime used
Python 3.11.9, SQLite 3.45.1, PyArrow 25.0.1 and SQLGlot 27.29.0. The optional
requirements pin the latter two; Python's SQLite version comes with Python.
Other versions/platforms are not promised byte-identical. An exact-hash failure
is an explicit mismatch, not a successful reproduction.

## 4. Assemble the local runtime inputs and tokenize

Obtain Gemma 3 4B PT's tokenizer at revision
`cc012e0a6d0787b4adcc0fa2c4da74402494554d` yourself under its terms.
Install the repository's documented computation environment for Transformers;
the SQL-only requirements do not install it. These commands load **only the local
tokenizer**, not model weights or a GPU:

```powershell
$Tokenizer = "<local snapshot directory containing tokenizer files>"
python data\scripts\build_runtime.py --prepared work\prepared-sql `
  --tokenizer-dir $Tokenizer --output work\data
python data\scripts\validate.py --runtime work\data
python data\scripts\tokenize_training.py --data-root work\data `
  --tokenizer-dir $Tokenizer --workspace work
```

Runtime assembly copies the included originals, creates deterministic gzip SQL
files, restores the exact evaluation subsets by their recorded IDs, and
reconstructs Taylor SQL token entries. It checks all six rebuilt payloads against
their historical release hashes. The local `work\data\manifest.json` declares
both original and external-derived inputs; the published manifest declares only
the original inputs.

Tokenization uses the actual training serializer in
[`src/remove_reset/data.py`](../src/remove_reset/data.py):

```text
prompt = BOS + encode(input, no automatic special tokens)
target = encode(newline + target_text, no automatic special tokens) + EOS
sequence = prompt + target
loss labels = ignore prompt positions; score target positions
```

There is **no packing, truncation or silent dropping**. The maximum length is
2,048 tokens. The token-cache command requires 62,817 examples, 7,628,998 total
nonpadding tokens and 2,087,265 target tokens, plus the seed-42 training order
and 19-target-token final example. It binds the cache to source bytes, tokenizer
signature and serializer source. Re-running verifies and reuses a matching
cache; changed/corrupt caches fail rather than being silently replaced.
Taylor tokens are checked separately during runtime assembly.

To materialize the 20 evaluation databases:

```powershell
python -m remove_reset materialize-databases --data-root work\data `
  --archive work\raw\spider-official-archive.zip --output work\databases
```

**Pass `--data-root work\data` to data-dependent CLI commands**, including
`preflight`, `train`, and `evaluate`. Keep `--database-root
work\databases` where needed. A clone alone supports original-data checks and
mask inspection, not SQL training/evaluation without acquisition.

## Discovery is a separate workflow

See [DISCOVERY.md](DISCOVERY.md) for all original sources, revision-pinned
acquisition metadata, deterministic filtering, LLM-assisted variant creation,
later exclusions, and checks for separately supplied frozen artifacts.
**No scripts to generate LLM variants are included.** Downloading the seed
benchmarks does not recreate the exact accepted variants. The frozen masks let
readers retrain without regenerating discovery.

## License and attribution

Original project datasets, references, rubric, original source summaries,
numerical masks and numerical recipe metadata are licensed under
**[Creative Commons Attribution 4.0 International](../LICENSES/CC-BY-4.0.txt)**.
See [LICENSE.md](LICENSE.md) for the grant and suggested attribution.
This permits sharing, adaptation and commercial reuse with attribution and
change notices, without forcing a downstream share-alike license.

This grant covers only the original contributions, not cited factual sources,
downloaded benchmarks, their derivatives, Gemma assets or software scripts.
Original authorship includes disclosed AI assistance; no exclusive copyright,
legal clearance of external sources, or absence of errors is guaranteed.
Source-code licensing is a separate repository-wide decision.
