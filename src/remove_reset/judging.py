"""Outcome-blind export and schema-checked import; NO network or automatic judge.

The private assignment is local reader metadata. Never give it to the judge.
The common rubric and versioned references are joined only AFTER model inference.
Pass/Fail/Uncertain (P/F/U) are human/LLM judgments, not exact-string parser scores.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from .artifacts import (
    new_output,
    read_json,
    save_result,
    sha256,
    value_hash,
    verify_bundle,
    write_json,
)
from .contracts import ToolResult
from .data import DOMAINS
from .evaluation import heldout_inputs

JUDGE_ARMS = (
    "base",
    "ordinary_lora",
    "random_removal",
    "random_reset",
    "taylor_removal",
    "taylor_reset",
    "guided_removal",
    "guided_reset",
    "quota_free_guided_reset",
    "quota_free_guided_removal",
)
FIELDS = (
    "case_id",
    "verdict",
    "rationale",
    "answer_evidence",
    "reference_criterion",
    "uncertainty_reason",
)
REFERENCE_FIELDS = (
    "reference_answer",
    "accepted_alternatives",
    "pass_criteria",
    "fail_criteria",
    "uncertain_criteria",
    "reference_derivation_or_source",
)


def normalized_references(data):
    rows = data.rows("heldout/references.jsonl")
    result = {}
    for row in rows:
        mapped = dict(row)
        mapped["prompt_id"] = row.get("prompt_id", row.get("id"))
        for normalized, original in (
            ("accepted_alternatives", "accepted_equivalents"),
            ("uncertain_criteria", "uncertain_conditions"),
            ("reference_derivation_or_source", "reference_derivation"),
        ):
            if normalized not in mapped and original in row:
                mapped[normalized] = row[original]
        if (
            mapped["prompt_id"] in result
            or any(
                key not in mapped or mapped[key] is None or mapped[key] == ""
                for key in REFERENCE_FIELDS
            )
            or not isinstance(mapped["accepted_alternatives"], list)
        ):
            raise ValueError("Invalid/duplicate versioned reference")
        result[mapped["prompt_id"]] = mapped
    expected = {r["prompt_id"] for r in heldout_inputs(data)}
    if len(result) != 120 or set(result) != expected:
        raise ValueError("Reference/input ID coverage differs")
    return result


def rank(namespace, seed, panel_sha256, **metadata):
    return value_hash(
        {"namespace": namespace, "seed": seed, "panel_sha256": panel_sha256, **metadata}
    )


def make_assignment(plans, *, panel_sha256, seed=20261028):
    """Source's 6 balanced batches + 10 common reliability sources, metadata ONLY.

    Every primary prompt appears for all ten model arms. Six batches each hold
    200 primary cases plus 10 duplicated reliability cases. The 60 reliability
    judgments are EXCLUDED from primary denominators. Source-group dependence
    and judgment disagreement must not be interpreted as model replications.
    """
    if (
        len(plans) != 120
        or len({p["prompt_id"] for p in plans}) != 120
        or Counter(p["domain"] for p in plans) != Counter({d: 12 for d in DOMAINS})
    ):
        raise ValueError("Assignment requires the complete balanced 120-input metadata")
    cases = {i: [] for i in range(1, 7)}
    for domain in DOMAINS:
        selected = sorted(
            (p for p in plans if p["domain"] == domain),
            key=lambda p: (
                rank(
                    "heldout-primary-batch-v1",
                    seed,
                    panel_sha256,
                    domain=domain,
                    prompt_id=p["prompt_id"],
                    input_sha256=p["input_utf8_sha256"],
                ),
                p["prompt_id"],
            ),
        )
        for index, prompt in enumerate(selected):
            batch = index % 6 + 1
            ordered = sorted(
                JUDGE_ARMS,
                key=lambda arm: rank(
                    "heldout-per-prompt-model-alias-v1",
                    seed,
                    panel_sha256,
                    prompt_id=prompt["prompt_id"],
                    arm=arm,
                ),
            )
            for alias_index, arm in enumerate(ordered):
                key = rank(
                    "heldout-opaque-primary-case-v1",
                    seed,
                    panel_sha256,
                    batch=batch,
                    prompt_id=prompt["prompt_id"],
                    arm=arm,
                )
                alias = rank(
                    "heldout-primary-alias-v1",
                    seed,
                    panel_sha256,
                    prompt_id=prompt["prompt_id"],
                    arm=arm,
                )
                cases[batch].append(
                    {
                        "case_id": "c-" + key[:32],
                        "candidate_alias": "Candidate " + alias[:8].upper(),
                        "prompt_model_order": alias_index,
                        "batch": batch,
                        "kind": "primary",
                        "domain": domain,
                        "prompt_id": prompt["prompt_id"],
                        "arm": arm,
                    }
                )
    for domain in DOMAINS:
        candidates = [
            (p["prompt_id"], arm) for p in plans if p["domain"] == domain for arm in JUDGE_ARMS
        ]
        prompt_id, arm = min(
            candidates,
            key=lambda pair: rank(
                "heldout-reliability-source-before-outcomes-v1",
                seed,
                panel_sha256,
                domain=domain,
                prompt_id=pair[0],
                arm=pair[1],
            ),
        )
        for batch in range(1, 7):
            key = rank(
                "heldout-opaque-reliability-case-v1",
                seed,
                panel_sha256,
                batch=batch,
                domain=domain,
                prompt_id=prompt_id,
                arm=arm,
            )
            alias = rank(
                "heldout-reliability-alias-v1",
                seed,
                panel_sha256,
                batch=batch,
                domain=domain,
                prompt_id=prompt_id,
                arm=arm,
            )
            cases[batch].append(
                {
                    "case_id": "c-" + key[:32],
                    "candidate_alias": "Candidate " + alias[:8].upper(),
                    "batch": batch,
                    "kind": "reliability",
                    "domain": domain,
                    "prompt_id": prompt_id,
                    "arm": arm,
                }
            )
    batches = []
    for batch in range(1, 7):
        ordered = sorted(
            cases[batch],
            key=lambda case: rank(
                "heldout-blinded-case-order-v1",
                seed,
                panel_sha256,
                batch=batch,
                case_id=case["case_id"],
            ),
        )
        for index, case in enumerate(ordered):
            case["case_index"] = index
        batches.append({"batch": batch, "cases": ordered})
    assignment = {
        "format": "remove-reset-private-judge-assignment-v1",
        "seed": seed,
        "panel_sha256": panel_sha256,
        "batches": batches,
    }
    ids = [c["case_id"] for b in batches for c in b["cases"]]
    primary = [
        (c["prompt_id"], c["arm"]) for b in batches for c in b["cases"] if c["kind"] == "primary"
    ]
    if (
        len(ids) != len(set(ids))
        or len(ids) != 1260
        or len(set(primary)) != 1200
        or any(len(b["cases"]) != 210 for b in batches)
    ):
        raise ValueError("Blinded assignment coverage differs")
    return assignment


def judgment_schema():
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["judgments"],
        "properties": {
            "judgments": {
                "type": "array",
                "minItems": 210,
                "maxItems": 210,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": list(FIELDS),
                    "properties": {
                        "case_id": {"type": "string"},
                        "verdict": {"enum": ["Pass", "Fail", "Uncertain"]},
                        **{k: {"type": "string"} for k in FIELDS[2:-1]},
                        "uncertainty_reason": {"type": ["string", "null"]},
                    },
                },
            }
        },
    }


def export_judge(data, generation_dirs, output):
    from .artifacts import read_jsonl

    output = new_output(output)
    try:
        references = normalized_references(data)
        inputs = heldout_inputs(data)
        rows_by_id = {r["prompt_id"]: r for r in inputs}
        generations, common_plans = {}, None
        bindings = {}
        for directory in generation_dirs:
            verify_bundle(directory)
            generation = read_json(Path(directory) / "generation.json")
            arm = generation["arm"]
            if (
                arm in generations
                or arm not in JUDGE_ARMS
                or generation["native_only"]
                or generation["data_manifest_sha256"] != data.fingerprint
            ):
                raise ValueError("Judge requires one matching final generation per arm, plus base")
            plans = read_json(Path(directory) / "token_plans.json")["plans"]
            if common_plans is not None and plans != common_plans:
                raise ValueError("Model generations used different exact prompt plans")
            common_plans = plans
            rows = read_jsonl(Path(directory) / "responses.jsonl")
            by_id = {r["prompt_id"]: r for r in rows}
            if len(rows) != 120 or len(by_id) != 120 or set(by_id) != set(references):
                raise ValueError("Incomplete/duplicate generation response coverage")
            for key, row in by_id.items():
                if (
                    row["arm"] != arm
                    or row["input"] != rows_by_id[key]["input"]
                    or row["status"] != "ok"
                    or row["effective_do_sample"] is not False
                    or row["effective_generation_mode"] != "greedy_search"
                ):
                    raise ValueError("Generation does not satisfy the approved raw-answer contract")
            generations[arm] = by_id
            bindings[arm] = sha256(Path(directory) / "manifest.json")
        if set(generations) != set(JUDGE_ARMS):
            raise ValueError("Blinded export requires all nine trained arms and the untouched base")
        assignment = make_assignment(common_plans, panel_sha256=value_hash(common_plans))
        # This pre-outcome panel convention was separate from the archived rubric.
        rubric = (
            data.path("heldout/judge_rubric.txt").read_text("utf-8").rstrip()
            + "\n\nFROZEN PANEL INTERPRETATION\n"
            "A single named item in response to 'which' is normally a complete "
            "selection, not uncertain merely because it lacks 'only'. "
            "This does not waive other essential requested parts or contradictions.\n"
        )
        write_json(output / "private_mapping.json", assignment)
        write_json(output / "judgment_schema.json", judgment_schema())
        private_cases = []
        for batch in assignment["batches"]:
            cases = []
            for case in batch["cases"]:
                response = generations[case["arm"]][case["prompt_id"]]
                reference = references[case["prompt_id"]]
                cases.append(
                    {
                        "case_id": case["case_id"],
                        "candidate_alias": case["candidate_alias"],
                        "prompt": rows_by_id[case["prompt_id"]]["input"],
                        "reference": {key: reference[key] for key in REFERENCE_FIELDS},
                        "candidate": response["raw_answer"],
                        "stop_status": response["stop_status"],
                        "continuation_tokens": response["continuation_tokens"],
                    }
                )
                private_cases.append(
                    {
                        **case,
                        "raw_answer": response["raw_answer"],
                        "source_group_id": reference.get("source_group_id"),
                        "raw_answer_sha256": response["raw_answer_utf8_sha256"],
                    }
                )
            write_json(
                output / f"judge_batch_{batch['batch']:02d}.json",
                {
                    "format": "remove-reset-blinded-judge-batch-v1",
                    "batch": batch["batch"],
                    "rubric": rubric,
                    "cases": cases,
                },
            )
        write_json(output / "private_cases.json", private_cases)
        write_json(output / "generation_bindings.json", bindings)
        return save_result(
            output,
            ToolResult(
                title="Blinded judge export",
                summary="Metadata-only deterministic assignments; no judge invoked or verdict invented.",
                metrics={"primary_cases": 1200, "reliability_cases": 60, "batches": 6},
                notes=[
                    "Send only judge_batch_XX.json and judgment_schema.json to an independent reviewer.",
                    "Keep private_mapping/private_cases/generation_bindings away from reviewers.",
                    "Prompt and answer contents are untrusted data, never instructions to execute.",
                    "References were not model inputs. Duplicated reliability cases do not increase n.",
                ],
            ),
        )
    except BaseException as exc:
        if not (output / "result.json").exists():
            save_result(
                output,
                ToolResult(title="Judge export failed", summary="No judgments created."),
                status="failed",
                error=exc,
            )
        raise


def validate_verdicts(doc, cases):
    """Strict P/F/U schema, exhaustive IDs and verbatim answer evidence."""
    if not isinstance(doc, dict) or set(doc) != {"judgments"}:
        raise ValueError("Verdict document must contain only judgments")
    rows = doc["judgments"]
    if not isinstance(rows, list) or len(rows) != len(cases):
        raise ValueError("Every supplied case requires exactly one judgment")
    by_id = {case["case_id"]: case for case in cases}
    seen = set()
    for row in rows:
        if (
            not isinstance(row, dict)
            or set(row) != set(FIELDS)
            or row["case_id"] not in by_id
            or row["case_id"] in seen
            or row["verdict"] not in ("Pass", "Fail", "Uncertain")
        ):
            raise ValueError("Unknown/duplicate case, fields or P/F/U verdict")
        seen.add(row["case_id"])
        for key in ("rationale", "reference_criterion"):
            if not isinstance(row[key], str) or not row[key].strip():
                raise ValueError(f"Judgment needs nonempty {key}")
        evidence = row["answer_evidence"]
        answer = by_id[row["case_id"]]["raw_answer"]
        if not isinstance(evidence, str) or evidence not in answer or (answer and not evidence):
            raise ValueError("Answer evidence must be a nonempty verbatim candidate span")
        reason = row["uncertainty_reason"]
        if row["verdict"] == "Uncertain":
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError("Uncertain verdict needs a specific uncertainty reason")
        elif reason is not None:
            raise ValueError("Resolved judgments require null uncertainty_reason")
    return rows


def import_judge(export_dir, verdict_files, output):
    output = new_output(output)
    try:
        verify_bundle(export_dir)
        cases = read_json(Path(export_dir) / "private_cases.json")
        verdicts = []
        seen_batches = set()
        for path in verdict_files:
            doc = read_json(path)
            supplied = {row["case_id"] for row in doc.get("judgments", [])}
            matching = [case for case in cases if case["case_id"] in supplied]
            batches = {case["batch"] for case in matching}
            if len(batches) != 1 or len(matching) != 210 or batches & seen_batches:
                raise ValueError(
                    "Each verdict file must match one distinct complete 210-case batch"
                )
            seen_batches |= batches
            verdicts.extend(validate_verdicts(doc, matching))
        if seen_batches != set(range(1, 7)):
            raise ValueError("All six blinded judgment batches are required")
        by_id = {row["case_id"]: row for row in verdicts}
        joined = [{**case, **by_id[case["case_id"]]} for case in cases]
        primary = [row for row in joined if row["kind"] == "primary"]
        summaries = []
        for arm in JUDGE_ARMS:
            rows = [r for r in primary if r["arm"] == arm]
            counts = dict(Counter(r["verdict"] for r in rows))
            summaries.append(
                {
                    "arm": arm,
                    "denominator": 120,
                    "counts": counts,
                    "pass_fraction_fixed_denominator": counts.get("Pass", 0) / 120,
                }
            )
        write_json(output / "judgments.json", joined)
        return save_result(
            output,
            ToolResult(
                title="Imported independent P/F/U judgments",
                summary="Exhaustive schema/evidence checks; uncertainty and raw judgments preserved.",
                metrics={"primary_denominator_per_arm": 120, "reliability_excluded": 60},
                tables={"primary_counts": summaries},
                notes=[
                    "Import validation checks structure/evidence provenance, not correctness of the judge.",
                    "Uncertain stays explicit and does not become a fabricated pass or discarded denominator.",
                    "Use source-group paired analysis, report judge disagreement, and retain human review.",
                    "A semantic judge is not an official benchmark scorer or a mechanistic proof.",
                ],
            ),
        )
    except BaseException as exc:
        if not (output / "result.json").exists():
            save_result(
                output,
                ToolResult(title="Judge import failed", summary="No partial score accepted."),
                status="failed",
                error=exc,
            )
        raise
