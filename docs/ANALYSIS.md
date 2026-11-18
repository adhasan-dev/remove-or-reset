# Publication analysis

`scripts\analyze.py` regenerates publication tables and dependency-free SVG
figures from new, reader-generated artifacts. The repository does not bundle
historical outputs, judgments, bootstrap values, or selection derivatives.
Analysis is offline and never invokes a model or judge.

## Run

Use a new output directory:

```powershell
.\.venv\Scripts\python.exe scripts\analyze.py `
  --config C:\work\analysis-config.json `
  --output C:\work\analysis-2026-11-15
```

Python callers may use:

```python
from remove_reset.analysis import analyze, main

analyze("analysis-config.json", "new-output")
main(["--config", "analysis-config.json", "--output", "new-output"])
```

NumPy is required for the broad bootstrap and optional sensitivity figure. It
is already part of the project's `compute` optional dependency set. Matplotlib
is not required.

## Strict configuration

Unknown fields are rejected. Paths may be absolute or relative to the config
file.

```json
{
  "format": "remove-reset-analysis-config-v1",
  "production": true,
  "data": "data",
  "evaluations": [
    {"arm": "ordinary_lora", "seed": 43, "path": "outputs/eval-ordinary"},
    {
      "arm": "quota_free_guided_removal",
      "seed": 43,
      "path": "outputs/eval-quota-free-removal"
    }
  ],
  "training": [
    {"arm": "ordinary_lora", "seed": 43, "path": "outputs/train-ordinary-report"}
  ],
  "judgments": [
    {"seed": 43, "path": "outputs/imported-independent-judgments"}
  ],
  "sensitivity": {
    "path": "outputs/guided-sensitivity/arrays.npz",
    "metadata": {
      "format": "remove-reset-guided-sensitivity-v1",
      "sha256": "64-lowercase-hex-file-digest",
      "source_run": "reader-generated guided scoring run identity",
      "data_manifest_sha256": "64-lowercase-hex-data-manifest-digest",
      "metric": "mean_abs_sensitivity",
      "aggregation": "signed reference derivatives averaged before absolute value; equal groups and strata"
    }
  }
}
```

Canonical conditions are `base`, `ordinary_lora`, `random_removal`,
`random_reset`, `taylor_removal`, `taylor_reset`, `guided_removal`,
`guided_reset`, `quota_free_guided_removal`, and
`quota_free_guided_reset`.

`production: true` enforces 128 SQL queries across 20 databases, 64 graphs
with one direct and one two-step record each, 8,650 validation target tokens,
the complete 3,927-update training budget, 120 judgments per supplied arm,
12 prompts per domain, and 73 global source groups. Use `false` only for
small synthetic tests.

Each evaluation and training entry is a single `(arm, seed)` result. Contrasts
are formed only inside the same seed. Runs from different seeds are reported
separately and are never pooled as independent trials. `base` has no training
seed in its model artifact; its config seed assigns it to the paired analysis
stratum.

`data` names the release data root used for `masks\*.json`; its mask files are
checksummed into the generated layer-allocation table. A training path may
name either a sealed invocation report directly or its run root containing a
checksum-bound `latest_report.json`.

Judgment paths must be complete outputs from `judge-import`, containing
`judgments.json`. Only `kind: "primary"` records enter the 120-item
denominator. No automatic or substitute verdicts are produced.

The sensitivity input is optional, but the depth-derivative panel is marked
missing without it and the analysis bundle remains `partial`. A mask alone
cannot reconstruct that figure. The NPZ must contain
`mean_abs_sensitivity[23,2,34,10240]`, `loss_types` including `content`, and
23 `strata`, plus the explicit checksum and provenance above.

## Validation and statistics

Every input must have a sealed `remove-reset-artifacts-v1` manifest with
`status: complete`, an unchanged file set/checksums, and a complete
`result.json`. Failed, partial, paused, `budget_excluded`, context-excluded,
missing-row, duplicate-row, and stale cross-arm panels are rejected. The SQL
reference preflight must be passed, complete, and identity-matched.

Invalid model-generated SQL remains in the fixed denominator as an incorrect
answer. The accepted terminal prediction statuses are `parse_error`,
`execution_error`, fixed query `timeout`, and bounded result/worker-memory
`resource_error`, in addition to `ok`. Their raw execution status fields must
agree and every non-`ok` row must be marked incorrect. Evaluation-budget
exclusions, unknown statuses, worker exits, Windows job setup failures, failed
references, and other infrastructure failures are rejected rather than scored.

Primary scores are rebuilt from raw records rather than copied from summary
metrics:

- SQL uses each raw record's saved execution-agreement boolean; invalid model
  SQL remains an incorrect answer in the fixed denominator after valid reference
  preflight.
- Graph correctness is recomputed from stripped prediction/reference text.
- Validation NLL is recomputed as total raw NLL divided by raw target tokens.
- Pass/Fail/Uncertain counts are rebuilt from imported primary judgments.

SQL uses 10,000 `random.Random(20261020).choices` draws over sorted database
IDs. Whole databases repeat with multiplicity and accuracy remains
query-weighted. Graph analysis uses the same method over 64 sorted graph IDs;
both tasks are retained, with shared draws for combined, direct, and two-step
statistics.

Broad analysis uses NumPy `Generator(PCG64(20261020))`, drawing 73 global
source groups per attempt. Group multiplicities are shared across arms and
domains. Attempts with an empty domain are rejected until 10,000 draws are
accepted. The statistic is the equal-domain mean known-Pass fraction.
Uncertain remains in every denominator. Bounds `P/120` through `(P+U)/120`
are unresolved-verdict bounds, not confidence intervals.

All contrast rows name their direction as **condition minus comparator**.
Intervals are linearly interpolated 2.5th and 97.5th percentiles. They are
descriptive, unadjusted, and conditional on the observed masks, outputs,
judgments, and training seed.

## Outputs

The output is a fresh sealed bundle containing:

- `result.json`, `report.html`, `manifest.json`
- `source_provenance.json` with input paths, manifest hashes, config, software,
  and source checksums
- `bootstrap_protocol.json` and `bootstrap_values.json`
- CSV, JSON, and derived LaTeX in `tables\`
- dependency-free SVG plus companion CSV tables in `figures\`

Principal tables are `primary_outcomes`, `paired_sql_graph_contrasts`,
`broad_outcomes`, `paired_broad_contrasts`, `training_memory`,
`layer_allocations`, and `availability`. Missing arms, training reports,
judgments, or sensitivity inputs are explicit; unavailable memory is null,
never zero. New release resource fields (`peak_allocated_bytes`,
`peak_reserved_bytes`, `memory_scope`, and evaluation resources) are used when
present.

The standalone `result.json` and `report.html` embed the substantive
`primary_outcomes`, paired SQL/graph contrasts and intervals,
`training_memory`, broad outcomes, and broad contrasts. The CSV/JSON/LaTeX
sidecars remain the reusable machine-readable copies, but they are not
required to see the measured SQL, graph, NLL, resource, or uncertainty values.

`family_expansion.svg` is a fixed-design schematic and therefore needs no
historical numerical evidence. `quota_free_allocation.svg` comes only from the
released mask. In contrast, `depth_sensitivity.svg` is never inferred from a
mask or post-LoRA outcome: it requires the separate validated pre-intervention
derivative tensor and its calibration metadata.

An output is `complete` only when all ten conditions have evaluation and broad
judgment coverage for every configured seed, all nine trained arms have
training reports, and the sensitivity input is present. Valid subset analyses
are sealed as `partial`, not presented as a complete manuscript rebuild.
