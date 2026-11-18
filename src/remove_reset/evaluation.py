"""Primary SQL execution/NLL/reasoning and secondary gold-free generation.

All generated token IDs and raw scores are retained in reader output. Evaluations
are new immutable invocations, not resumable training checkpoints: after a failed
evaluation, inspect error/partial JSONL and restart into a NEW output directory.
No reference text is ever passed to the generation function.
"""

from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from pathlib import Path

from .artifacts import (
    canonical_bytes,
    new_output,
    read_json,
    save_result,
    sha256,
    software,
    source_identity,
    text_hash,
    value_hash,
    write_json,
)
from .config import ARMS, Recipe
from .contracts import ToolResult
from .data import DOMAINS, encode_record, evaluation_panels, tokenizer_signature, validate_records
from .sql import canonical_rows, execute_sql, parse_query, results_equal


def preflight_references(panels, output, *, fixture=False):
    """Execute every gold read-only query BEFORE any model loading; fail closed."""
    cache, summaries = {}, []
    if not fixture and len(panels["sql"]) != 128:
        raise ValueError("Reference preflight requires all 128 frozen SQL records")
    for row in panels["sql"]:
        binding = panels["databases"][row["db_id"]]
        gold = execute_sql(binding["path"], row["target"])
        summary = {
            "id": row["id"],
            "db_id": row["db_id"],
            "status": gold["status"],
            "database_sha256": binding["sha256"],
            "reference_sql_sha256": value_hash(row["target"]),
            "row_count": gold.get("row_count"),
            "columns": gold.get("columns"),
            "error": gold.get("error"),
        }
        if gold["status"] == "ok":
            ordered = parse_query(row["target"])
            gold["ordered_reference"] = ordered
            summary.update(
                ordered_reference=ordered,
                reference_stratum="empty" if gold["row_count"] == 0 else "nonempty",
                canonical_result_sha256=value_hash(
                    {"columns": gold["columns"], "rows": canonical_rows(gold, ordered)}
                ),
            )
            cache[row["id"]] = gold
        summaries.append(summary)
    write_json(
        Path(output) / "reference_preflight.json",
        {
            "format": "remove-reset-reference-preflight-v1",
            "model_measurement": False,
            "status": "passed" if len(cache) == len(panels["sql"]) else "failed",
            "records": summaries,
            "records_sha256": value_hash(summaries),
        },
    )
    if len(cache) != len(panels["sql"]):
        raise RuntimeError(
            "One or more frozen reference queries failed; no model evaluation allowed"
        )
    return cache


def target_nll(adapter, record, *, chunk_size=16):
    """One native hidden-state pass, FP32 CE sums in target-vocabulary chunks.

    hidden[:, prompt_length-1:] predicts every target, including newline and EOS.
    No full-context vocabulary tensor is needed. This is the original evaluation
    normalization (total NLL / total target tokens), not a mean over examples.
    """
    import torch
    import torch.nn.functional as functional

    record.validate(vocab_size=adapter.info.vocab_size)
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("NLL chunk size must be a positive integer")
    ids = torch.tensor([record.input_ids[:-1]], dtype=torch.long, device=adapter.device)
    targets = torch.tensor(
        record.input_ids[record.prompt_length :], dtype=torch.long, device=adapter.device
    )
    with torch.inference_mode():
        hidden = adapter.base.model(
            input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, return_dict=True
        ).last_hidden_state[:, record.prompt_length - 1 :, :]
        if hidden.shape[1] != record.target_tokens:
            raise ValueError("NLL hidden-state causal alignment differs")
        nlls, total = [], 0.0
        for offset in range(0, record.target_tokens, chunk_size):
            logits = adapter.base.get_output_embeddings()(hidden[:, offset : offset + chunk_size])
            cap = adapter.base.config.final_logit_softcapping
            if cap is not None:
                logits = torch.tanh(logits / cap) * cap
            values = functional.cross_entropy(
                logits[0].float(), targets[offset : offset + chunk_size], reduction="none"
            )
            if not torch.isfinite(values).all():
                raise FloatingPointError("Nonfinite target NLL")
            nlls.extend(values.cpu().tolist())
            # Preserve the source's FP32 sum kernel per 16-token chunk. The
            # separately retained individual values can differ at rounding level.
            loss = functional.cross_entropy(
                logits[0].float(), targets[offset : offset + chunk_size], reduction="sum"
            )
            total += loss.item()
    return {
        "status": "ok",
        "target_tokens": record.target_tokens,
        "nll_sum": total,
        "mean_nll": total / record.target_tokens,
        "token_nll": nlls,
        "target_ids": record.input_ids[record.prompt_length :],
    }


def prompt_plans(data, tokenizer, rows, output, *, heldout=False):
    """Freeze exact plain-BOS IDs/checksums; optional shipped plans must agree."""
    plans = []
    for index, row in enumerate(rows):
        ids = [tokenizer.bos_token_id] + list(
            tokenizer.encode(row["input"], add_special_tokens=False)
        )
        limit = 512 if heldout else 2048
        if not 1 <= len(ids) <= limit:
            raise ValueError("Untruncated prompt exceeds the fixed limit")
        plan = {
            "inference_index": index,
            "prompt_id": row["prompt_id"] if heldout else row["id"],
            "input": row["input"],
            "input_utf8_sha256": text_hash(row["input"]),
            "prompt_ids": ids,
            "prompt_tokens": len(ids),
            "prompt_ids_sha256": value_hash(ids),
        }
        if heldout:
            plan["domain"] = row["domain"]
        plans.append(plan)
    if heldout and "heldout/token_plans.json" in data.manifest["files"]:
        supplied = read_json(data.path("heldout/token_plans.json"))
        supplied = supplied["plans"] if isinstance(supplied, dict) else supplied
        expected = {p["prompt_id"]: p["prompt_ids"] for p in supplied}
        if expected != {p["prompt_id"]: p["prompt_ids"] for p in plans}:
            raise ValueError("Pinned tokenizer IDs differ from shipped heldout token plans")
    doc = {
        "format": "remove-reset-prompt-plans-v1",
        "tokenizer_signature": tokenizer_signature(tokenizer),
        "data_manifest_sha256": data.fingerprint,
        "plans": plans,
        "plans_sha256": value_hash(plans),
    }
    write_json(Path(output) / "token_plans.json", doc)
    return plans


def load_evaluation_adapter(
    base_dir, data, arm, *, prepared=None, training_run=None, native_only=False, device="cuda:0"
):
    from .checkpoint import load_final_adapter, resolve_checkpoint
    from .model import attach_lora
    from .training import load_arm

    if arm not in (*ARMS, "base"):
        raise ValueError("Unknown evaluation arm")
    if training_run is not None and native_only:
        raise ValueError("Choose final trained adapter OR explicit native-only evaluation")
    if arm == "base":
        if training_run is not None or native_only:
            raise ValueError("Base evaluation has no adapter/native-only modifier")
    elif training_run is None:
        if not native_only or arm == "ordinary_lora":
            raise ValueError(
                "A recovered arm requires --training-run; intervention pre-LoRA uses --native-only"
            )
    adapter = load_arm(base_dir, data, arm, prepared=prepared, device=device)
    if training_run is not None:
        identity = read_json(Path(training_run) / "identity.json")
        recipe = Recipe.from_document(identity["recipe"])
        if (
            identity["arm"] != arm
            or identity["fixture"]
            or identity["recipe"] != recipe.document()
            or identity["base"] != adapter.provenance["base"]
            or identity["native_manifest_sha256"]
            != adapter.provenance.get("native_manifest_sha256")
            or identity["tokenization"]["identity"]["data_manifest"] != data.fingerprint
        ):
            raise ValueError("Final adapter base/native/arm/data/recipe binding differs")
        attach_lora(adapter, recipe)
        state = load_final_adapter(resolve_checkpoint(training_run), adapter, identity)
        if (
            state["updates"],
            state["examples"],
            state["nonpadding_tokens"],
            state["target_tokens"],
        ) != (3927, 62817, 7628998, 2087265):
            raise ValueError("Final adapter did not complete the entire frozen epoch")
        adapter.provenance["training"] = {
            "identity": identity,
            "final_updates": state["updates"],
            "checkpoint_manifest_sha256": sha256(
                resolve_checkpoint(training_run) / "manifest.json"
            ),
        }
    adapter.freeze()
    return adapter


def aggregate_metrics(sql, validation, reasoning):
    def sql_summary(rows):
        count = len(rows)
        correct = sum(bool(row.get("correct")) for row in rows)
        executable = sum(row.get("prediction_execution_status") == "ok" for row in rows)
        return {
            "total": count,
            "correct": correct,
            "execution_agreement": correct / count if count else None,
            "executable": executable,
            "executable_rate": executable / count if count else None,
            "strict_matches": sum(bool(row.get("strict_match")) for row in rows),
            "status_counts": dict(Counter(row["status"] for row in rows)),
        }

    primary = sql_summary(sql)
    primary["metric"] = "custom single-database execution agreement; not official Spider"
    primary["by_database"] = {
        db: sql_summary([r for r in sql if r["db_id"] == db])
        for db in sorted({r["db_id"] for r in sql})
    }
    primary["by_empty_reference"] = {
        group: sql_summary([r for r in sql if r["reference_stratum"] == group])
        for group in ("empty", "nonempty")
    }
    scored = [row for row in validation if row["status"] == "ok"]
    count = sum(row["target_tokens"] for row in scored)
    total = sum(row["nll_sum"] for row in scored)
    mean = total / count if count else None
    retention = {}
    graphs = defaultdict(dict)
    for row in reasoning:
        graphs[row["graph_id"]][row["task"]] = row
    for task in ("direct", "two_step"):
        rows = [r for r in reasoning if r["task"] == task]
        correct = sum(bool(r.get("correct")) for r in rows)
        retention[task] = {
            "total": len(rows),
            "correct": correct,
            "accuracy": correct / len(rows) if rows else None,
            "status_counts": dict(Counter(r["status"] for r in rows)),
        }
    conditional = [
        g["two_step"]
        for g in graphs.values()
        if g["direct"].get("correct") and g["two_step"]["status"] != "budget_excluded"
    ]
    correct = sum(bool(r.get("correct")) for r in conditional)
    retention["two_step_given_direct_success"] = {
        "denominator": len(conditional),
        "correct": correct,
        "accuracy": correct / len(conditional) if conditional else None,
        "unobserved_two_step_among_direct_successes": sum(
            bool(g["direct"].get("correct")) and g["two_step"]["status"] == "budget_excluded"
            for g in graphs.values()
        ),
    }
    return {
        "sql": primary,
        "validation_nll": {
            "total": len(validation),
            "scored_examples": len(scored),
            "scored_target_tokens": count,
            "nll_sum": total if count else None,
            "token_weighted_nll": mean,
            "mean_example_nll": sum(r["mean_nll"] for r in scored) / len(scored)
            if scored
            else None,
            "perplexity": math.exp(mean)
            if mean is not None and mean <= math.log(float.fromhex("0x1.fffffffffffffp+1023"))
            else None,
            "exclusion_status_counts": dict(
                Counter(r["status"] for r in validation if r["status"] != "ok")
            ),
        },
        "reasoning": retention,
    }


def evaluate(
    base_dir,
    data,
    output,
    arm,
    *,
    prepared=None,
    training_run=None,
    native_only=False,
    database_root=None,
    device="cuda:0",
    budget_seconds=None,
):
    if budget_seconds is not None and (
        type(budget_seconds) not in (int, float)
        or not math.isfinite(budget_seconds)
        or budget_seconds <= 0
    ):
        raise ValueError("Evaluation budget must be positive and finite")
    output = new_output(output)
    try:
        panels = evaluation_panels(data, database_root)
        references = preflight_references(panels, output)
        adapter = load_evaluation_adapter(
            base_dir,
            data,
            arm,
            prepared=prepared,
            training_run=training_run,
            native_only=native_only,
            device=device,
        )
        all_rows = panels["sql"] + panels["reasoning"]
        plans = {p["prompt_id"]: p for p in prompt_plans(data, adapter.tokenizer, all_rows, output)}
        validation_tokens = [encode_record(adapter.tokenizer, row) for row in panels["validation"]]
        if sum(r.target_tokens for r in validation_tokens) != 8650:
            raise ValueError("Frozen validation panel must contain exactly 8650 target tokens")
        # Fixed source warmups, excluded from every scientific denominator.
        warmups = [
            adapter.generate(
                "Relation: A -> B\nQuestion: Follow A once.\nAnswer:",
                max_new_tokens=8,
                eos_token_ids=[1],
                skip_special_tokens=True,
            ),
            adapter.generate(
                "Schema:\nCREATE TABLE demo (x INTEGER);\n\nQuestion: Count rows.\nSQL:",
                max_new_tokens=8,
                eos_token_ids=[1],
                skip_special_tokens=True,
            ),
        ]
        import torch

        if adapter.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(adapter.device)
        started = time.monotonic()
        deadline = None if budget_seconds is None else started + budget_seconds
        records = {"sql": [], "reasoning": [], "validation": []}
        for panel in ("sql", "reasoning", "validation"):
            with (output / f"{panel}_records.jsonl").open(
                "x", encoding="utf-8", newline="\n"
            ) as stream:
                for index, row in enumerate(panels[panel]):
                    measured = {"id": row["id"], "reference": row["target"]}
                    if panel == "sql":
                        gold = references[row["id"]]
                        measured.update(
                            db_id=row["db_id"],
                            correct=False,
                            strict_match=False,
                            reference_stratum="empty" if gold["row_count"] == 0 else "nonempty",
                            ordered_reference=gold["ordered_reference"],
                        )
                    elif panel == "reasoning":
                        measured.update(graph_id=row["graph_id"], task=row["task"], correct=False)
                    if deadline is not None and time.monotonic() >= deadline:
                        measured["status"] = "budget_excluded"
                    elif panel == "validation":
                        record = validation_tokens[index]
                        measured.update(
                            target_nll(adapter, record)
                            if record.eligible
                            else {"status": "context_length", "target_tokens": record.target_tokens}
                        )
                    else:
                        allowance = 256 if panel == "sql" else 32
                        plan = plans[row["id"]]
                        if plan["prompt_tokens"] + allowance > 2048:
                            measured["status"] = "context_length"
                        else:
                            generated = adapter.generate(
                                row["input"],
                                max_new_tokens=allowance,
                                eos_token_ids=[1],
                                skip_special_tokens=True,
                                expected_prompt_ids=plan["prompt_ids"],
                            )
                            measured.update(generated)
                            prediction = generated["raw_answer"].strip()
                            measured.update(prediction=prediction, status="ok")
                            if deadline is not None and time.monotonic() >= deadline:
                                measured["status"] = "budget_excluded"
                            elif panel == "reasoning":
                                measured["correct"] = prediction == row["target"].strip()
                            else:
                                executed = execute_sql(
                                    panels["databases"][row["db_id"]]["path"],
                                    prediction,
                                    deadline=deadline,
                                )
                                measured.update(
                                    prediction_execution=executed,
                                    prediction_execution_status=executed["status"],
                                    status=executed["status"],
                                    strict_match=prediction == row["target"].strip(),
                                    correct=results_equal(
                                        executed, gold, gold["ordered_reference"]
                                    ),
                                )
                    records[panel].append(measured)
                    stream.write(canonical_bytes(measured).decode("utf-8") + "\n")
                    stream.flush()
        metrics = aggregate_metrics(**records)
        metrics["resources"] = {
            "evaluation_seconds_after_warmup": time.monotonic() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(adapter.device)
            if adapter.device.startswith("cuda") else None,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(adapter.device)
            if adapter.device.startswith("cuda") else None,
            "scope": "SQL, graph and NLL evaluation after warmup; excludes loading",
        }
        partial = any(r["status"] == "budget_excluded" for rows in records.values() for r in rows)
        return save_result(
            output,
            ToolResult(
                title="Frozen primary evaluation",
                summary="Full-query single-database execution agreement, target NLL and graph answers.",
                metrics=metrics,
                tables={
                    "provenance": [
                        {
                            "arm": arm,
                            "native_only": native_only,
                            "model": adapter.provenance,
                            "data_manifest_sha256": data.fingerprint,
                            "software": software(),
                            "source_identity": source_identity(),
                            "warmups": warmups,
                            "budget_seconds": budget_seconds,
                        }
                    ]
                },
                notes=[
                    "All SQL (128) and reasoning (64/task) records remain in fixed denominators.",
                    "NLL reports scored-token coverage explicitly; exclusions are never successes.",
                    "SQL equality on one database is not semantic equivalence or official Spider scoring.",
                    "Gold references are preflighted but never passed to generation.",
                    "Greedy decoding is explicitly enforced; source SQL EOS remains [1], not [1,106].",
                    "A one-seed comparison does not establish a general mechanistic causal explanation.",
                ],
            ),
            status="partial" if partial else "complete",
        )
    except BaseException as exc:
        if not (output / "result.json").exists():
            save_result(
                output,
                ToolResult(
                    title="Evaluation failed",
                    summary="Inspect the error and partial records; no success inferred.",
                ),
                status="failed",
                error=exc,
            )
        raise


def heldout_inputs(data):
    rows = data.rows("heldout/prompts.jsonl")
    validate_records(rows, id_key="prompt_id", require_target=False)
    forbidden = {
        "target",
        "reference",
        "reference_answer",
        "accepted_alternatives",
        "accepted_equivalents",
        "pass_criteria",
        "fail_criteria",
    }
    if (
        len(rows) != 120
        or Counter(row["domain"] for row in rows) != Counter({domain: 12 for domain in DOMAINS})
        or any(forbidden & set(row) for row in rows)
        or any(row.get("language", "en") != "en" for row in rows)
    ):
        raise ValueError("Heldout prompts must be 120 gold-free English inputs, 12/domain")
    return rows


def generate_heldout(
    base_dir,
    data,
    output,
    arm,
    *,
    prepared=None,
    training_run=None,
    native_only=False,
    device="cuda:0",
):
    output = new_output(output)
    try:
        rows = heldout_inputs(data)  # Deliberately never opens references.jsonl.
        adapter = load_evaluation_adapter(
            base_dir,
            data,
            arm,
            prepared=prepared,
            training_run=training_run,
            native_only=native_only,
            device=device,
        )
        plans = prompt_plans(data, adapter.tokenizer, rows, output, heldout=True)
        responses = []
        with (output / "responses.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
            for plan in plans:
                value = adapter.generate(
                    plan["input"],
                    max_new_tokens=64,
                    eos_token_ids=[1, 106],
                    expected_prompt_ids=plan["prompt_ids"],
                    skip_special_tokens=False,
                )
                response = {
                    "prompt_id": plan["prompt_id"],
                    "domain": plan["domain"],
                    "input": plan["input"],
                    "arm": arm,
                    "status": "ok",
                    **value,
                }
                responses.append(response)
                stream.write(canonical_bytes(response).decode("utf-8") + "\n")
                stream.flush()
        write_json(
            output / "generation.json",
            {
                "format": "remove-reset-heldout-generation-v1",
                "arm": arm,
                "native_only": native_only,
                "data_manifest_sha256": data.fingerprint,
                "model": adapter.provenance,
                "software": software(),
                "source_identity": source_identity(),
                "prompt_plans_sha256": value_hash(plans),
                "seed": 42,
                "max_new_tokens": 64,
                "eos_token_ids": [1, 106],
                "no_chat_template": True,
                "reference_conditioned": False,
            },
        )
        return save_result(
            output,
            ToolResult(
                title="Gold-free secondary generation",
                summary="120 raw plain-BOS greedy continuations; no correctness judgments computed.",
                metrics={
                    "arm": arm,
                    "responses": len(responses),
                    "continuation_tokens": sum(r["continuation_tokens"] for r in responses),
                },
                notes=[
                    "No reference, rubric, verdict or score was supplied to the model.",
                    "Raw special tokens and token-limit stops remain visible for blinded review.",
                    "Use judge-export/import separately; no network judge is invoked.",
                ],
            ),
        )
    except BaseException as exc:
        if not (output / "result.json").exists():
            save_result(
                output,
                ToolResult(
                    title="Generation failed", summary="No judgments or missing responses invented."
                ),
                status="failed",
                error=exc,
            )
        raise


def paired_cluster_difference(
    left, right, *, cluster_key, value_key="correct", replicates=2000, seed=20261013
):
    """Paired whole-cluster bootstrap, not independent channel/mask-seed trials.

    Inputs must share IDs/groups. Resample whole databases/graphs/source groups,
    retaining every member and raw paired values. One training seed still cannot
    estimate between-training-seed uncertainty.
    """
    import numpy as np

    if type(replicates) is not int or replicates < 1:
        raise ValueError("Bootstrap replicate count must be positive")
    a, b = {r["id"]: r for r in left}, {r["id"]: r for r in right}
    if len(a) != len(left) or len(b) != len(right) or not a or set(a) != set(b):
        raise ValueError("Paired rows must have identical unique IDs")
    groups = defaultdict(list)
    for key in sorted(a):
        if a[key][cluster_key] != b[key][cluster_key]:
            raise ValueError("Paired cluster identities differ")
        difference = float(b[key][value_key]) - float(a[key][value_key])
        if not math.isfinite(difference):
            raise ValueError("Nonfinite paired raw value")
        groups[a[key][cluster_key]].append(difference)
    clusters = sorted(groups)
    rng = np.random.default_rng(seed)
    values = np.asarray(
        [
            np.mean(
                [
                    value
                    for index in rng.integers(len(clusters), size=len(clusters))
                    for value in groups[clusters[index]]
                ]
            )
            for _ in range(replicates)
        ]
    )
    return {
        "difference_right_minus_left": float(
            np.mean([v for group in groups.values() for v in group])
        ),
        "interval_95": np.quantile(values, [0.025, 0.975]).tolist(),
        "clusters": len(clusters),
        "replicates": replicates,
        "seed": seed,
        "raw_paired_differences": dict(groups),
    }
