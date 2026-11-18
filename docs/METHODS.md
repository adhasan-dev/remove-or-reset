# Fixed methods and reproduction boundaries

This document describes a recipe, not empirical outcomes. Source is adapted
into a standalone package; it is **not claimed byte-identical** to the old study
scripts or numerically identical across software/hardware. Data certification
does not certify the newly packaged code. All generated evidence stays local.

## Inputs and endpoint

- Official Gemma 3 4B **PT**, text-only, revision
  `cc012e0a6d0787b4adcc0fa2c4da74402494554d`.
- Frozen prepared SQL: train 62,817; selection 8,389; validation 7,553;
  Spider development 1,034. Gzip decompression preserves original bytes;
  both envelope and original hashes are checked.
- Train uncompressed SHA-256:
  `69ab28152f80a7cdf14271c3532fa484a20470001c7f2aadec17cd28f4b1b707`.
- One complete epoch: 3,927 optimizer updates, 7,628,998 nonpadding tokens,
  2,087,265 target tokens. Seed-42 Python shuffle over the entire input order:
  `0fdde173e56e111470582fb628a9fadbece4010748a67a5bf7d5d27d290edbec`.
- Final epoch checkpoint, never an evaluation-selected checkpoint.

Seed 42 is the historical default, not a requirement for new replications.
The configuration accepts a different explicit nonnegative 32-bit training
seed while keeping all other settings fixed. Each run records the resulting
whole-epoch order hash; both arms of a replication pair use the same seed.
Only seed 42 must match the historical order hash and 19-target-token tail.
Reset initialization and greedy evaluation retain their separate fixed seeds.

SQL content is not bundled. The offline scripts in `data/scripts/` rebuild the
historical SQL preparation from pinned reader-acquired sources and require exact
output hashes. They also assemble runtime evaluation/Taylor panels and tokenize
with the study serializer. See `data/README.md`. This does **not** claim
regeneration of stochastic discovery variants, model screening outcomes or
historical judgments. No helper model is substituted.

## Coordinate and dtype contract

For original layer `L` and channel `j`, the coordinate couples
`gate_proj.weight[j,:]`, `up_proj.weight[j,:]` and `down_proj.weight[:,j]`.
The measured activation is the actual gated product entering `down_proj`,
not either incoming projection alone.

Each uniform mask selects 2,048 of 10,240 channels in each of 34 layers:
69,632 channels total. Removal physically slices all three tensors and retains
the ascending original-ID complement, width 8,192 in every layer.

Quota-free guided selection preserves the uniform guided mask's same 1,100
protected coordinates, then globally selects the lowest 69,632 equal-stratum
mean raw content sensitivities without per-layer quotas. Exact ties use layer
then original channel ID. Frozen selected counts range from 0 to 7,905 by layer;
physical widths therefore range from 2,335 to 10,240. Relative to uniform guided,
both the allocation and ranking objective change.

Native construction records the original scalar universe **and** explicit
physical layer widths. The safe loader constructs those shapes on meta before
checking every key/shape/dtype and the shard index. No
`ignore_mismatched_sizes` fallback exists. Embedding ties and all assigned
parameters are checked. Nonpersistent rotary frequencies are reconstructed in
FP32, never blanket-cast to BF16.

Reset keeps full shapes. A separate **CPU** Torch Generator, seed 3042, traverses
layers ascending, gate then up. It draws selected incoming rows in FP32 from a
normal distribution with mean 0 and standard deviation 0.02,
casts them to BF16, and sets selected outgoing columns exactly to zero.
Unselected entries remain bitwise unchanged. Different layer allocations change
the RNG draw assignment; values on shared uniform/quota-free coordinates are not
claimed matched. Native parameters remain frozen during recovery.

Full-width arms have 3,880,263,168 native + 29,802,496 LoRA parameters.
Both same-total compact allocations have 3,345,489,408 native + 26,460,160 LoRA
parameters. Different shapes/adapter capacities are part of the treatment;
the comparison does not isolate selection alone.

Preparation checks visited no-op identity, unchanged retained/native-outside-mask
bytes, exact reset-vs-dense-zero logits, source compact-vs-dense-zero BF16
tolerances (max 1.0, mean 0.05), and exact save/reload on three technical prompts.
These controls do not establish benchmark preservation.

Training startup additionally checks exact zero-update LoRA function identity.
For reset, the selected products must supply nonzero down-LoRA-B gradient
differences in every layer versus selected-product zeroing, despite identical
initial forward losses. This uses one fixed training record with no optimizer
step; gradients are cleared and RNG restored before the epoch.

## Serialization, loss and optimization

For one row:

```text
prompt = [BOS] + encode(input, add_special_tokens=False)
target = encode("\n" + target_text, add_special_tokens=False) + [EOS]
input_ids = prompt + target
labels = [-100] * len(prompt) + target
```

Right padding also has label -100. No string merging at the prompt boundary,
packing, truncation, or silently dropped training record is allowed.
For labels at original positions `p+1`, the model is explicitly asked for
logits at previous positions `p`; no HF automatic shifted loss is used.
The batch requests the union of causal target positions and masks other
examples' positions.

Every microbatch returns a **sum** of FP32 target-token cross-entropies.
Backpropagation divides each sum by the target-token count of the **entire
effective batch**. This includes the one-example last update with 19 targets;
it is not divided by eight microbatches or a nominal 16 examples.

LoRA: rank16, alpha32, dropout0, bias none, default PEFT initialization;
q/k/v/o and gate/up/down in all 34 layers. Native BF16, trainable LoRA FP32.
Fused AdamW: LR0.0002, weight decay0.01, gradient norm clip1;
microbatch2 × accumulation8 = effective16; nonreentrant gradient checkpointing,
SDPA, context2048, deterministic settings, TF32 disabled.

Original schedule semantics: `warmup = ceil(3927*.03) = 118`;
LambdaLR initializes LR to `base_lr/118`. For step `s < warmup`, factor
`(s+1)/warmup`; afterward factor
`0.5*(1+cos(pi*clamp((s-warmup)/(3927-warmup),0,1)))`.
Optimizer step precedes scheduler step. Pausing never changes this horizon.

Checkpoints are due every 200 updates or 1,200 seconds, with a cooperative
1,800-second maximum-gap stop. Predictions trigger earlier boundary saves.
Partial updates cannot be saved. All trainable tensors, nonpersistent buffers,
named Adam moments, scheduler, whole permutation/cursor and Python/NumPy/Torch
RNG use safetensors plus tagged JSON. No `torch.load`, arbitrary pickle or
executable deserialization is used. Resume binds data, base/tokenizer files,
native intervention, recipe, source, software and execution hardware.
File fsync/atomic rename is not an arbitrary-power-loss guarantee.

## Frozen selector provenance

**Fixed-mask reproduction** trains exactly the shipped selected original IDs.
Selector-regeneration code is intentionally not shipped and is not required.
Legacy source-mask JSON is accepted only through an explicit known-format
transport conversion; its new release manifest binds its actual shipped bytes.
No legacy hard-bound run/archive/approval dependency is imported or weakened.

The random mask remains reproducible from NumPy `default_rng(1042)` / PCG64.
Taylor and both guided masks are frozen numerical inputs. The data and provenance
documents retain their historical calibration descriptions and source hashes,
including the quota-free mask's saved raw-content objective and protected set,
without implying that excluded selector inputs or code are part of this release.

## Evaluation and interpretation

Primary fixed denominators: SQL128 across20 databases; validation256 with8,650
target tokens; direct64 and two-step64 over the same64 graphs, seed20261013.
All gold SQL queries must pass the read-only sandbox before model loading.
Prediction is the complete stripped generated string, never extracted SQL.
SQLite uses readonly/immutable connections, an authorizer, no extensions,
2-second query/5-second parent limits, 50,000 complete rows, 8-MiB serialized
output, and a 512-MiB worker cap (Windows Job Object; Linux resource limit).
Failure/timeout never becomes a truncated successful query.

Compare exact typed rows with duplicate multiplicity and column order.
INTEGER/REAL use exact rational equality, no tolerance; TEXT/BLOB/NULL remain
distinct, including invalid UTF-8 TEXT. Row order matters only for top-level gold
ORDER BY. This is **custom single-database execution agreement**, not official
Spider scoring or semantic equivalence.

SQL generates256 and reasoning32 tokens, plain BOS, greedy, EOS `[1]`.
Secondary generates64 tokens for all120 English inputs, plain BOS/no chat,
greedy, EOS `[1,106]`, seed42. `do_sample=False` is both in the explicit
generation config and highest-priority kwargs; `use_model_defaults=False`
prevents inherited sampling. Effective greedy mode/EOS and actual token IDs
are checked. Raw special tokens and stop status remain available.

References/rubrics are separately versioned and never fed to generation.
Judging uses six blinded210-case batches:1,200 primary cases plus60 reliability
duplicates, excluded from denominators. Import does not certify judge correctness.
The exported effective rubric includes the panel's pre-outcome bare-selection
convention: a correct single named answer to "which" is not incomplete merely
because it omits "only". The archived common-rubric input bytes remain unchanged;
other essential requested parts and contradictions still matter.
Retain Uncertain judgments, raw evidence and source groups. The optional paired
bootstrap resamples whole databases/graphs/source groups, not channels/examples
as independent trials. The paper's nine-arm comparison uses training seed 42;
ordinary LoRA and quota-free guided removal additionally use seeds 43 and 44.
Limited seed coverage and broader guided calibration/compute still limit
causal/mechanistic and generalization claims. Historical run outputs are not
bundled; readers generate their own results.

