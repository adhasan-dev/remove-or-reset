# Remove or Reset?

Companion code for [**Remove or Reset? A Case Study of Channel Selection Before
LoRA Fine-Tuning** (PDF)](paper/paper.pdf).

## What does the paper study?

The paper asks whether changing selected internal channels in **Gemma 3 4B**
before fine-tuning affects task accuracy, other capabilities, and resource use.
Channels are internal features in the model's feed-forward (MLP) layers. The
study compares two ways to change them:

- **Removal:** physically delete selected channels, making the model smaller.
- **Reset:** keep the model's size but reinitialize selected channels instead
  of keeping their pretrained weights.

Each modified model is then adapted to **text-to-SQL**: generating database
queries from natural-language questions. Adaptation uses **LoRA** (low-rank
adaptation), which trains small adapter weights while keeping the base weights
fixed. Evaluation measures SQL accuracy, reasoning and broader capabilities,
parameter counts, memory use, and training time.

There are **nine experimental conditions**, called *arms*: ordinary LoRA with
no channel changes, plus removal and reset under four selection methods:
random, Taylor (gradient-based importance), uniform guided (the same number of
channels selected in every layer), and quota-free guided (the same total
number selected, but distributed unevenly across layers).

## Results and conclusion

Removing 20% of the MLP channels with quota-free guided selection reduced the
native text-model parameter count by **13.78%** and peak allocated memory
during training by **10.5–10.8%** relative to ordinary LoRA. Across three
matched training seeds, SQL execution agreement averaged **72.40% for removal
versus 74.48% for ordinary LoRA**. All runs used the same 128 SQL queries and
120 cross-domain short-answer prompts.

| Training seed | SQL correct: ordinary LoRA | SQL correct: removal | Broad Pass: ordinary LoRA | Broad Pass: removal |
|---|---:|---:|---:|---:|
| 42 | 93/128 | 98/128 | 64/120 | 55/120 |
| 43 | 98/128 | 90/128 | 65/120 | 51/120 |
| 44 | 95/128 | 90/128 | 65/120 | 55/120 |

At seed 42, quota-free removal scored above random (88/128), Taylor (90/128),
and uniform guided removal (93/128) on SQL. Its advantage over uniform guided
removal was 3.91 percentage points, but the paired 95% interval included zero
(−2.59 to +10.29). Resetting the same quota-free channels instead of removing
them gave a different trade-off: lower SQL agreement (91 vs. 98 correct) but
more broad-screen passes (62 vs. 55), while retaining full model width.

**Conclusion:** Quota-free guided removal produced a smaller SQL specialist
with repeatable training-memory savings. Its initial SQL lead reversed in
both follow-up seeds, and it passed fewer broad-screen prompts at every seed.
The study does not establish lossless compression or a universally better
selector: only the ordinary-LoRA/removal pair was repeated, on one model and
one specialization task.

## What does this repository contain?

It contains four frozen channel masks (lists of selected channels), fixed
training and evaluation recipes, input-preparation scripts, training with
checkpoint/resume support, evaluation and blinded-judging tools, and local
analysis/report generation. You can retrain all nine arms or repeat the main
ordinary-LoRA versus quota-free-removal comparison across training seeds.

**The base model, tokenizer, fine-tuned weights/adapters, and historical run
outputs are not distributed here.** Readers obtain the specified model and
external data themselves and generate new results locally. The channel masks
are supplied as fixed inputs; this repository does not rerun their selection.

## How to use it

| Step | What it does |
|---|---|
| 1. [Install and run small checks](#environment-and-small-checks) | Verify the package and inputs, then exercise all nine arms on a tiny CPU model before committing to expensive training. |
| 2. [Obtain the model](#obtain-the-base-explicitly) and [prepare the data](data/README.md) | Acquire the exact pretrained model and external datasets, rebuild the prescribed inputs, and [extract the evaluation databases](#obtain-the-evaluation-databases-explicitly). |
| 3. [Check readiness and train](#fresh-preflight-preparation-and-training) | Verify inputs and model behavior, apply a supplied channel mask for an intervention arm, and train LoRA. Full-size runs need a CUDA/BF16 GPU and take hours per arm. |
| 4. [Evaluate and judge](#evaluation-and-blinded-judging) | Measure SQL and reasoning performance and generate broader evaluation answers for independent blinded human review. Human judgments are not automatic. |
| 5. [Analyze your results](#regenerate-tables-intervals-and-figures) | Produce tables, uncertainty intervals, supported figures, and standalone reports from your saved runs. These are new results, not a replay of the paper's outputs. |

Read `LICENSING.md`, `THIRD_PARTY_NOTICES.md`, and `data/README.md` before
redistribution. Original data are **CC BY 4.0**; original software is
**MIT-licensed** (see `LICENSE`).
Checksums establish input identity, not permission to redistribute upstream work.

## What is reproducible?

| Scope | Included |
|---|---|
| Retraining all nine arms | Four frozen masks, source and strict recipe; readers acquire and prepare SQL |
| Mask selection | Exact frozen random, Taylor, uniform-guided and quota-free-guided masks |
| Selector regeneration | **Not included or required.** Selection calibration remains provenance, not executable release code |
| Primary evaluation | Original reasoning inputs and SQL/validation selection recipes; reader obtains SQL and SQLite archive |
| Secondary evaluation | 120 gold-free inputs, separate references, common blinded-judge rubric; no automatic judge |

Guided benchmark texts, variants, reversible token plans/references and the
20 Spider SQLite databases are **not bundled**. No cleaned subset is substituted
for either guided mask's historical calibration.
See [methods and boundaries](docs/METHODS.md).

## Environment and small checks

Python 3.11. The CLI's `--help`, `list`, and data `check` need only the standard
library; computation uses the optional `compute` dependencies.

```powershell
py -3.11 -m venv .venv
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install --no-deps -e .
& $py -m remove_reset list --json
& $py -m remove_reset check --json
```

For the tiny CPU check, install the computation dependencies in your **local** environment.
For 4B training, install the appropriate CUDA Torch wheel **first**; full-size
runs require CUDA BF16, with no CPU/offload or quantized fallback. Other
hardware/software combinations are not guaranteed numerically identical.
No installer or model downloader runs automatically.

```powershell
& $py -m pip install -e ".[compute]"
& $py -m remove_reset tiny-check --output work\tiny
& $py -m remove_reset check --artifact work\tiny --json
```

`tiny-check` uses a real randomly initialized 3-layer FP32 CPU Gemma, PEFT,
safetensors, AdamW and greedy generation. It exercises all nine arm paths,
native save/reload, update-one pause/resume, a one-example final batch
(19 target tokens at the default seed),
and a resumed/uninterrupted comparison. **It does not validate historical 4B
outputs, resource feasibility, bitwise numerical parity, or benchmark accuracy.**

Portable shell equivalents after activating your local environment:

```sh
python -m pip install --no-deps -e .
python -m remove_reset list --json
python -m remove_reset check --json
python -m pip install -e ".[compute]"
python -m remove_reset tiny-check --output work/tiny
```

After installation, `remove-reset` can be used in place of
`python -m remove_reset` in the examples.

## Obtain the base explicitly

Base: [`google/gemma-3-4b-pt`](https://huggingface.co/google/gemma-3-4b-pt),
revision **`cc012e0a6d0787b4adcc0fa2c4da74402494554d`**.
Accept Gemma's applicable terms yourself. If authentication is required, provide
`HF_TOKEN` through your environment/secret manager; never put it in this repository.

The following is an **explicit reader acquisition command**, not part of setup
or any experiment command:

```sh
hf download google/gemma-3-4b-pt \
  --revision cc012e0a6d0787b4adcc0fa2c4da74402494554d \
  --cache-dir /your/base-cache \
  --include config.json generation_config.json "tokenizer*" \
    special_tokens_map.json added_tokens.json "*.safetensors" model.safetensors.index.json
```

In PowerShell use one line or backticks instead of `\` line continuations.
Every package model load is local-only, safe-tensor-only and text-only:
no remote code, pickle, quantization, CPU/offload fallback, or alternate base.
Use either the explicit snapshot directory or its HF cache root as `--base-dir`.

## Obtain the evaluation databases explicitly

**First follow [data/README.md](data/README.md)** to acquire and process the SQL
sources and build the local `work\data` runtime root. Downloaded records,
processed derivatives and token IDs are not included in the published data folder.

`data/recipes/evaluation.json` records the official archive URL/hash and the
20 exact database member hashes. Obtain that archive from the official provider
yourself, subject to its source terms; do not use a mirror or redistribute it.
Then extract only the selected databases into your ignored workspace:

```powershell
& $py -m remove_reset materialize-databases --data-root work\data --archive "<official-archive.zip>" `
  --output work\databases
```

This command verifies the **archive and every selected member**. It never
downloads anything or performs a general-purpose unzip. SQLite stays outside
the released `data/` payload.

## Fresh preflight, preparation and training

Real 4B runs require CUDA BF16 and substantial memory, disk and hours per arm.
The following training commands are **not** small setup tests.
No historical approval/result is required or impersonated: first produce a
fresh local technical preflight.

```powershell
$Base = "<your-base-cache>"
$Work = "work"
$Data = "$Work\data"
& $py -m remove_reset preflight --data-root $Data --base-dir $Base --workspace $Work `
  --database-root "$Work\databases" --config configs\study.json --output "$Work\preflight"
```

The gate checks all gold SQL queries before loading the base, exact train
token/order accounting, causal positions and repeat greedy token IDs.
It is **not** an accuracy threshold or a full-size training resource pilot.
Run `evaluate --arm base` for your fresh untouched-base evaluation.

| `--arm` | Frozen mask | Native shapes before LoRA |
|---|---|---|
| `ordinary_lora` | none | original |
| `random_removal`, `random_reset` | random | compact / original |
| `taylor_removal`, `taylor_reset` | Taylor | compact / original |
| `guided_removal`, `guided_reset` | uniform guided | compact / original |
| `quota_free_guided_removal`, `quota_free_guided_reset` | quota-free guided | nonuniform compact / original |

Prepare each of the eight intervention arms separately. Ordinary LoRA uses the
untouched base and must omit `--prepared`.

```powershell
& $py -m remove_reset prepare --data-root $Data --arm random_reset --base-dir $Base `
  --output "$Work\prepared\random_reset"

# Optional bounded pause: the FULL epoch/schedule is still configured.
& $py -m remove_reset train --data-root $Data --arm random_reset --base-dir $Base --workspace $Work `
  --prepared "$Work\prepared\random_reset" --preflight "$Work\preflight" `
  --output "$Work\train\random_reset" --pause-at-update 1

# Same run/config/data/native checkpoint; restore optimizer, scheduler, sampler and RNG.
& $py -m remove_reset train --data-root $Data --arm random_reset --base-dir $Base --workspace $Work `
  --prepared "$Work\prepared\random_reset" --preflight "$Work\preflight" `
  --output "$Work\train\random_reset" --resume
```

For ordinary LoRA:

```powershell
& $py -m remove_reset train --data-root $Data --arm ordinary_lora --base-dir $Base --workspace $Work `
  --preflight "$Work\preflight" --output "$Work\train\ordinary_lora"
```

Other arms use the same commands with the matching arm/preparation/output names.
The recipe rejects unknown fields and scientific-setting changes except an
explicit nonnegative 32-bit training `seed` in the configuration. The default
is the historical seed 42. For replication, copy `configs\study.json`, change
only `seed`, and pass the same configuration to `preflight` and `train` for
both arms. This seed controls LoRA initialization and example order; frozen
masks, reset seed 3042, and evaluation decoding settings remain unchanged.
Different seeds can change the final example's target-token count; the full
epoch and total target-token budget remain fixed. There is no
“best checkpoint”: evaluate the **completed epoch**. First Ctrl+C requests a
boundary pause; a second interrupt discards the partial update. Resume only from
the latest checksummed boundary. Never run two trainers on one output directory.

For one additional matched-seed pair, the offline runner below rebuilds inputs,
checks the seed on the CPU fixture, performs fresh preflight, and runs ordinary
LoRA followed by quota-free guided removal. Each arm first pauses after two
updates to exercise checkpoint resume, then completes the same full epoch.
It evaluates SQL/NLL/graphs and saves gold-free broad-screen responses; it
does not supply human judgments. This is a multi-hour GPU run, not a setup test.
Use a private copy of the five acquired raw files because SQL preparation
extracts database fixtures beside them.

```powershell
& $py scripts\run_seed_pair.py --base-dir $Base --raw-dir "<private-raw-directory>" `
  --workspace "<new-private-pair-directory>" --seed 43
```

`pair_status.json`, per-stage logs, and per-update training events record
progress and explicit failures. Output directories must be new; interrupted
training can be resumed with the individual commands above after inspecting
the recorded failure. The runner never changes masks or launches additional
seeds based on an observed score.

To reproduce the three-seed main comparison, run that command independently
with seeds **42, 43 and 44**, using new workspaces `work\seed-42`,
`work\seed-43` and `work\seed-44` and a private raw-input copy for each.
These are six full training runs, not six evaluations of one adapter.
The supplied analysis configuration consumes those exact output layouts:

```powershell
& $py scripts\analyze.py --config configs\main_comparison.analysis.json `
  --output "<new-private-three-seed-analysis>"
```

It reports per-seed raw outcomes and paired intervals without pooling repeated
queries as independent observations. It remains `partial` for full-manuscript
coverage because other arms, independent broad judgments and numerical
sensitivities are not supplied by that configuration.

## Evaluation and blinded judging

```powershell
& $py -m remove_reset evaluate --data-root $Data --arm base --base-dir $Base `
  --database-root "$Work\databases" --output "$Work\eval\base"
& $py -m remove_reset evaluate --data-root $Data --arm random_reset --base-dir $Base `
  --prepared "$Work\prepared\random_reset" --training-run "$Work\train\random_reset" `
  --database-root "$Work\databases" --output "$Work\eval\random_reset"
& $py -m remove_reset generate-heldout --data-root $Data --arm random_reset --base-dir $Base `
  --prepared "$Work\prepared\random_reset" --training-run "$Work\train\random_reset" `
  --output "$Work\heldout\random_reset"
```

For a pre-LoRA intervention evaluation, explicitly replace `--training-run ...`
with `--native-only`. It is never silently labeled a trained/recovered model.
Generate the secondary panel for the base and all nine completed arms, then:

```sh
python -m remove_reset judge-export --generations BASE_DIR ARM1_DIR ... ARM9_DIR --output work/blind
python -m remove_reset judge-import --export work/blind --verdicts VERDICT1.json ... VERDICT6.json --output work/judged
```

Send reviewers **only** the six `judge_batch_XX.json` files and the judgment
schema. Keep the private mapping away from them. Import requires exhaustive
Pass/Fail/Uncertain judgments with verbatim candidate evidence. No hosted API,
Copilot, strict-answer parser, or placeholder verdict is used.

**Independent re-evaluation, not deterministic score replay:** readers must
provide their own human judgments against the shipped references and rubric.
Historical model answers and human verdicts are not distributed. New judgments
can differ from the paper even when generated answers match; the scripts
reproduce the scoring procedure, not the historical human decisions. Keep
Uncertain verdicts in the denominator and report them explicitly. Generating
answers alone does not reproduce the published broad-screen scores.

## Regenerate tables, intervals, and figures

Use the offline analysis entry point with a JSON configuration listing your
sealed evaluation/training outputs and, when available, independently judged
broad-screen outputs:

```powershell
& $py scripts\analyze.py --config "<analysis-config.json>" --output "<new-analysis-directory>"
```

See [ANALYSIS.md](docs/ANALYSIS.md) for the exact schema, paired comparison
directions, 10,000-draw SQL/graph and domain-balanced broad-screen bootstraps,
and CSV/LaTeX/SVG outputs. Results remain separated by training seed. Missing
arms or judgments are reported, not invented or pooled as independent trials.
The channel-depth sensitivity figure additionally requires numerical
pre-intervention sensitivity data with provenance: it cannot be reconstructed
from masks or post-LoRA answers. Those diagnostics are not bundled, and this
command does not regenerate selectors.

## Inspect artifacts and selection scope

Successful analysis/preparation/evaluation outputs contain `result.json`,
standalone `report.html`, `manifest.json`, and numeric arrays when relevant.
Started runs leave `error.json` on failure and no success-shaped metric.
Argument/input/gate rejections before a run is created return an explicit JSON
error on stderr. Training reports live
under `TRAIN_RUN/invocations/`; `latest_report.json` points to the latest report.
`events.jsonl` is append-only; do not hold mutable checkpoint pointers open.
Outputs are immutable except explicit pointers and training continuation.

```powershell
& $py -m remove_reset run mask-audit --param "mask=data\masks\quota_free_guided.json" `
  --output "$Work\mask-audit"
```

Selector-regeneration code is intentionally excluded. The shipped masks are the
immutable training inputs; their manifests retain the historical source paths,
hashes, objectives and limitations without claiming the selectors were rerun.

The implementation in `src/remove_reset/` covers local model loading,
channel interventions, LoRA training and checkpointing, evaluation, and blinded
judging. `mask-audit` inspects the supplied channel choices; it does not select
new channels or establish their effect on model performance.
