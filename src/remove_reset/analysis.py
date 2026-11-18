"""Offline publication analysis from reader-generated, sealed artifacts only."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from .artifacts import (
    new_output,
    read_json,
    read_jsonl,
    save_result,
    sha256,
    software,
    source_identity,
    value_hash,
    verify_bundle,
    write_json,
)
from .contracts import ToolResult

ARMS = (
    "base",
    "ordinary_lora",
    "random_removal",
    "random_reset",
    "taylor_removal",
    "taylor_reset",
    "guided_removal",
    "guided_reset",
    "quota_free_guided_removal",
    "quota_free_guided_reset",
)
TRAINED_ARMS = ARMS[1:]
DOMAINS = (
    "arithmetic",
    "logic",
    "reading_comprehension",
    "commonsense",
    "science",
    "factual_knowledge",
    "code_comprehension",
    "algorithms",
    "data_reasoning",
    "language_understanding",
)
VERDICTS = ("Pass", "Fail", "Uncertain")
BOOTSTRAP_SEED = 20261020
BOOTSTRAP_DRAWS = 10000


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _index(rows, field="id"):
    result = {row[field]: row for row in rows}
    _require(len(result) == len(rows), f"Duplicate {field}")
    return result


def _percentile(values, probability):
    """NumPy-compatible linear percentile for a finite, nonempty sequence."""
    values = sorted(float(value) for value in values)
    _require(values and all(math.isfinite(value) for value in values), "Invalid percentile values")
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def _interval(values):
    return [_percentile(values, 0.025), _percentile(values, 0.975)]


def cluster_bootstrap(
    conditions,
    *,
    cluster_key,
    id_key="id",
    value_key="correct",
    replicates=BOOTSTRAP_DRAWS,
    seed=BOOTSTRAP_SEED,
):
    """Exact shared Python-random whole-cluster bootstrap.

    Rates are row weighted after cluster multiplicity is applied. The returned
    draws are shared across every condition, so contrasts are paired.
    """
    _require(type(replicates) is int and replicates > 0, "Replicates must be positive")
    _require(type(seed) is int and seed >= 0, "Seed must be a nonnegative integer")
    _require(isinstance(conditions, dict) and conditions, "Conditions are required")
    indexed = {name: _index(rows, id_key) for name, rows in conditions.items()}
    first_name = next(iter(indexed))
    ids = set(indexed[first_name])
    _require(ids, "Bootstrap rows are empty")
    for name, rows in indexed.items():
        _require(set(rows) == ids, f"Paired IDs differ for {name}")
    groups = defaultdict(list)
    for identity in sorted(ids):
        group = indexed[first_name][identity][cluster_key]
        for name, rows in indexed.items():
            _require(rows[identity][cluster_key] == group, f"Cluster identity differs for {name}")
            value = rows[identity][value_key]
            _require(type(value) in (bool, int, float), "Bootstrap value must be numeric")
            _require(math.isfinite(float(value)), "Bootstrap value must be finite")
        groups[group].append(identity)
    cluster_ids = sorted(groups)
    rng = random.Random(seed)
    draws = [rng.choices(cluster_ids, k=len(cluster_ids)) for _ in range(replicates)]
    rates = {name: [] for name in conditions}
    for draw in draws:
        sampled_ids = [identity for group in draw for identity in groups[group]]
        for name, rows in indexed.items():
            rates[name].append(
                math.fsum(float(rows[identity][value_key]) for identity in sampled_ids)
                / len(sampled_ids)
            )
    observed = {
        name: math.fsum(float(row[value_key]) for row in rows.values()) / len(rows)
        for name, rows in indexed.items()
    }
    return {
        "groups": cluster_ids,
        "draws": draws,
        "rates": rates,
        "observed": observed,
        "replicates": replicates,
        "seed": seed,
        "rng": "random.Random(seed).choices(sorted_cluster_ids, k=cluster_count)",
    }


def named_contrast(sampled, condition, comparator):
    """Return condition minus comparator; never infer direction from argument position."""
    _require(condition in sampled["rates"], f"Unknown condition: {condition}")
    _require(comparator in sampled["rates"], f"Unknown comparator: {comparator}")
    differences = [
        left - right
        for left, right in zip(
            sampled["rates"][condition], sampled["rates"][comparator], strict=True
        )
    ]
    return {
        "condition": condition,
        "comparator": comparator,
        "direction": f"{condition} minus {comparator}",
        "difference": sampled["observed"][condition] - sampled["observed"][comparator],
        "interval_95": _interval(differences),
    }


def broad_bootstrap(
    rows,
    *,
    arms,
    domains=DOMAINS,
    replicates=BOOTSTRAP_DRAWS,
    seed=BOOTSTRAP_SEED,
):
    """PCG64 global-source-group bootstrap with empty-domain rejection."""
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Publication analysis requires NumPy (install the compute extra)") from exc

    _require(type(replicates) is int and replicates > 0, "Replicates must be positive")
    prompt_metadata = {}
    seen = set()
    for row in rows:
        identity = (row["arm"], row["prompt_id"])
        _require(identity not in seen, "Duplicate arm/prompt judgment")
        seen.add(identity)
        metadata = (row["domain"], row["source_group_id"])
        old = prompt_metadata.setdefault(row["prompt_id"], metadata)
        _require(old == metadata, "Prompt domain/source group differs across arms")
        _require(row["verdict"] in VERDICTS, "Unknown judgment verdict")
    _require(set(row["arm"] for row in rows) == set(arms), "Judgment arm coverage differs")
    prompts = sorted(prompt_metadata)
    for arm in arms:
        _require(
            {row["prompt_id"] for row in rows if row["arm"] == arm} == set(prompts),
            f"Prompt coverage differs for {arm}",
        )
    groups = sorted({metadata[1] for metadata in prompt_metadata.values()})
    group_index = {group: i for i, group in enumerate(groups)}
    domain_index = {domain: i for i, domain in enumerate(domains)}
    arm_index = {arm: i for i, arm in enumerate(arms)}
    group_domain = np.zeros((len(groups), len(domains)), dtype=np.int32)
    group_pass = np.zeros((len(groups), len(arms), len(domains)), dtype=np.int32)
    for prompt_id, (domain, group) in prompt_metadata.items():
        _require(domain in domain_index, f"Unknown domain for {prompt_id}: {domain}")
        group_domain[group_index[group], domain_index[domain]] += 1
    for row in rows:
        if row["verdict"] == "Pass":
            group_pass[
                group_index[row["source_group_id"]],
                arm_index[row["arm"]],
                domain_index[row["domain"]],
            ] += 1
    _require(not (group_domain.sum(axis=0) == 0).any(), "A broad domain has no prompts")
    rng = np.random.Generator(np.random.PCG64(seed))
    weights = []
    rejected_masks = []
    attempts = 0
    while len(weights) < replicates:
        attempts += 1
        sampled = rng.integers(0, len(groups), size=len(groups), dtype=np.int64)
        multiplicities = np.bincount(sampled, minlength=len(groups))
        empty = multiplicities @ group_domain == 0
        if empty.any():
            rejected_masks.append(empty.astype(np.uint8))
            continue
        weights.append(multiplicities)
    weights = np.asarray(weights, dtype=np.int32)
    denominators = weights @ group_domain
    numerator = np.einsum(
        "rg,gad->rad", weights.astype(float), group_pass.astype(float)
    )
    rates = (numerator / denominators[:, None, :]).mean(axis=2)
    observed = {}
    for arm in arms:
        arm_rows = [row for row in rows if row["arm"] == arm]
        observed[arm] = sum(row["verdict"] == "Pass" for row in arm_rows) / len(arm_rows)
    return {
        "groups": groups,
        "rates": {arm: rates[:, i].tolist() for i, arm in enumerate(arms)},
        "observed": observed,
        "weights": weights,
        "rejected_domain_masks": np.asarray(rejected_masks, dtype=np.uint8).reshape(
            -1, len(domains)
        ),
        "empty_domain_redraws": attempts - replicates,
        "attempts": attempts,
        "replicates": replicates,
        "seed": seed,
        "rng": "numpy.random.Generator(numpy.random.PCG64(seed))",
    }


def unresolved_bounds(rows):
    counts = Counter(row["verdict"] for row in rows)
    denominator = len(rows)
    _require(denominator > 0, "No judgments")
    return {
        "Pass": counts["Pass"],
        "Fail": counts["Fail"],
        "Uncertain": counts["Uncertain"],
        "denominator": denominator,
        "known_pass_fraction": counts["Pass"] / denominator,
        "unresolved_lower": counts["Pass"] / denominator,
        "unresolved_upper": (counts["Pass"] + counts["Uncertain"]) / denominator,
        "bounds_are_not_confidence_intervals": True,
    }


def _resolve(config_dir, value):
    _require(isinstance(value, str) and value, "Artifact path must be nonempty text")
    path = Path(value)
    return (config_dir / path).resolve() if not path.is_absolute() else path.resolve()


def _verify_input(directory):
    manifest = verify_bundle(directory)
    result = read_json(Path(directory) / "result.json")
    _require(result.get("status") == "complete", f"Result is not complete: {directory}")
    return result, {
        "path": str(Path(directory).resolve()),
        "manifest_sha256": sha256(Path(directory) / "manifest.json"),
        "manifest_status": manifest["status"],
    }


def _training_seed(provenance):
    try:
        return provenance["model"]["training"]["identity"]["recipe"]["seed"]
    except (KeyError, TypeError):
        return None


def _validate_sql_outcome(row):
    """Keep terminal model-query failures; reject exclusions and infrastructure failures."""
    status = row.get("status")
    terminal = {"ok", "parse_error", "execution_error", "timeout", "resource_error"}
    _require(status in terminal, f"SQL row has excluded/unsupported status: {status!r}")
    execution = row.get("prediction_execution")
    _require(isinstance(execution, dict), "SQL row lacks raw prediction execution")
    _require(
        execution.get("status") == status
        and row.get("prediction_execution_status") == status,
        "SQL execution status fields disagree",
    )
    _require(type(row.get("correct")) is bool, "SQL raw correctness is not boolean")
    if status != "ok":
        _require(row["correct"] is False, "Non-executable SQL cannot be marked correct")
    if status == "resource_error":
        _require(
            execution.get("limit")
            in {"complete_rows", "serialized_bytes", "worker_memory_bytes"},
            "SQL worker/infrastructure resource failure is not a scored model outcome",
        )
        _require("worker_exit_code" not in execution, "SQL worker failure is not a model outcome")
    if status == "timeout":
        _require(
            execution.get("limit") in {"query_seconds", "parent_seconds"},
            "SQL timeout lacks the fixed query-limit provenance",
        )


def _load_evaluation(entry, config_dir, production):
    _require(isinstance(entry, dict) and set(entry) == {"arm", "seed", "path"}, "Bad evaluation entry")
    arm, seed = entry["arm"], entry["seed"]
    _require(arm in ARMS, f"Unknown evaluation arm: {arm}")
    _require(type(seed) is int and 0 <= seed < 2**32, "Evaluation seed must be uint32")
    path = _resolve(config_dir, entry["path"])
    result, binding = _verify_input(path)
    tables = result.get("tables", {})
    provenance_rows = tables.get("provenance", [])
    _require(len(provenance_rows) == 1, "Evaluation needs one provenance row")
    provenance = provenance_rows[0]
    _require(provenance.get("arm") == arm, "Evaluation arm differs from config")
    actual_seed = _training_seed(provenance)
    if arm != "base":
        _require(actual_seed == seed, "Evaluation training seed differs from config")
    records = {}
    for panel in ("sql", "reasoning", "validation"):
        rows = read_jsonl(path / f"{panel}_records.jsonl")
        _require(rows, f"{panel} records are empty")
        if panel != "sql":
            _require(
                all(row.get("status") == "ok" for row in rows),
                f"{panel} has excluded/failed rows",
            )
        records[panel] = rows
    sql = records["sql"]
    for row in sql:
        _validate_sql_outcome(row)
    preflight = read_json(path / "reference_preflight.json")
    preflight_rows = preflight.get("records", [])
    _require(
        preflight.get("format") == "remove-reset-reference-preflight-v1"
        and preflight.get("status") == "passed"
        and len(preflight_rows) == len(sql)
        and all(row.get("status") == "ok" for row in preflight_rows),
        "SQL reference/data preflight is incomplete or failed",
    )
    preflight_by_id = _index(preflight_rows)
    _require(
        set(preflight_by_id) == {row["id"] for row in sql}
        and all(
            preflight_by_id[row["id"]]["db_id"] == row["db_id"]
            for row in sql
        ),
        "SQL reference preflight identities differ from scored rows",
    )
    reasoning = records["reasoning"]
    for row in reasoning:
        _require(row["task"] in ("direct", "two_step"), "Unknown graph task")
        row["inferred_correct"] = row["prediction"].strip() == row["reference"].strip()
    validation = records["validation"]
    for row in validation:
        _require(
            type(row.get("target_tokens")) is int
            and row["target_tokens"] > 0
            and type(row.get("nll_sum")) in (int, float)
            and math.isfinite(row["nll_sum"]),
            "Invalid raw validation score",
        )
    if production:
        _require(len(sql) == 128 and len({row["db_id"] for row in sql}) == 20, "SQL panel is not 128/20")
        task_counts = Counter(row["task"] for row in reasoning)
        _require(task_counts == Counter({"direct": 64, "two_step": 64}), "Graph panel is not 64/task")
        _require(len({row["graph_id"] for row in reasoning}) == 64, "Graph panel is not 64 graphs")
        _require(
            len(validation) > 0 and sum(row["target_tokens"] for row in validation) == 8650,
            "Validation target-token denominator differs",
        )
    plans = read_json(path / "token_plans.json")
    binding.update(
        kind="evaluation",
        arm=arm,
        seed=seed,
        data_manifest_sha256=provenance.get("data_manifest_sha256"),
        token_plans_sha256=value_hash(plans),
    )
    return {"arm": arm, "seed": seed, "path": path, "records": records, "provenance": provenance}, binding


def _panel_identity(evaluation):
    return {
        "sql": {
            row["id"]: (row["db_id"], row["reference"], row.get("reference_stratum"))
            for row in evaluation["records"]["sql"]
        },
        "reasoning": {
            row["id"]: (row["graph_id"], row["task"], row["reference"])
            for row in evaluation["records"]["reasoning"]
        },
        "validation": {
            row["id"]: (row["reference"], row["target_tokens"])
            for row in evaluation["records"]["validation"]
        },
    }


def _evaluation_outcome(evaluation):
    sql = evaluation["records"]["sql"]
    reasoning = evaluation["records"]["reasoning"]
    validation = evaluation["records"]["validation"]
    target_tokens = sum(row["target_tokens"] for row in validation)
    direct = [row for row in reasoning if row["task"] == "direct"]
    two_step = [row for row in reasoning if row["task"] == "two_step"]
    resources = read_json(evaluation["path"] / "result.json").get("metrics", {}).get(
        "resources", {}
    )
    return {
        "seed": evaluation["seed"],
        "arm": evaluation["arm"],
        "sql_correct": sum(row["correct"] for row in sql),
        "sql_denominator": len(sql),
        "sql_fraction": sum(row["correct"] for row in sql) / len(sql),
        "validation_examples": len(validation),
        "validation_target_tokens": target_tokens,
        "validation_token_weighted_nll": math.fsum(row["nll_sum"] for row in validation)
        / target_tokens,
        "graph_direct_correct": sum(row["inferred_correct"] for row in direct),
        "graph_direct_denominator": len(direct),
        "graph_two_step_correct": sum(row["inferred_correct"] for row in two_step),
        "graph_two_step_denominator": len(two_step),
        "evaluation_seconds_after_warmup": resources.get("evaluation_seconds_after_warmup"),
        "evaluation_peak_allocated_bytes": resources.get("peak_allocated_bytes"),
        "evaluation_peak_reserved_bytes": resources.get("peak_reserved_bytes"),
        "evaluation_memory_scope": resources.get("scope"),
        "evaluation_memory_available": resources.get("peak_allocated_bytes") is not None,
    }


def _load_training(entry, config_dir, production):
    _require(isinstance(entry, dict) and set(entry) == {"arm", "seed", "path"}, "Bad training entry")
    arm, seed = entry["arm"], entry["seed"]
    _require(arm in TRAINED_ARMS and type(seed) is int and 0 <= seed < 2**32, "Bad training arm/seed")
    path = _resolve(config_dir, entry["path"])
    if not (path / "result.json").is_file():
        latest = read_json(path / "latest_report.json")
        _require(
            isinstance(latest, dict)
            and latest.get("status") == "complete"
            and isinstance(latest.get("path"), str),
            "Training latest_report does not identify a complete report",
        )
        run_root = path.resolve(strict=True)
        path = (run_root / latest["path"]).resolve(strict=True)
        _require(path.is_relative_to(run_root), "Training report path escapes its run")
        _require(
            sha256(path / "manifest.json") == latest.get("manifest_sha256"),
            "Training latest_report manifest checksum differs",
        )
    result, binding = _verify_input(path)
    bindings = result.get("tables", {}).get("bindings", [])
    _require(len(bindings) == 1, "Training report needs one identity binding")
    identity = bindings[0]
    _require(identity.get("arm") == arm, "Training arm differs from config")
    _require(identity.get("recipe", {}).get("seed") == seed, "Training seed differs from config")
    metrics = result["metrics"]
    _require(metrics.get("status") == "complete", "Training report is not a complete epoch")
    if production:
        _require(
            (metrics.get("updates"), metrics.get("examples"), metrics.get("nonpadding_tokens"), metrics.get("target_tokens"))
            == (3927, 62817, 7628998, 2087265),
            "Training budget differs from production",
        )
    counts = metrics.get("parameter_counts", {})
    binding.update(kind="training", arm=arm, seed=seed)
    return {
        "seed": seed,
        "arm": arm,
        "updates": metrics.get("updates"),
        "examples": metrics.get("examples"),
        "nonpadding_tokens": metrics.get("nonpadding_tokens"),
        "target_tokens": metrics.get("target_tokens"),
        "train_token_weighted_nll": metrics.get("target_weighted_train_nll"),
        "training_seconds": metrics.get("training_seconds"),
        "native_parameters": counts.get("native_parameters"),
        "adapter_parameters": counts.get("adapter_parameters"),
        "trainable_parameters": counts.get("trainable_parameters"),
        "training_peak_allocated_bytes": metrics.get("peak_allocated_bytes"),
        "training_peak_reserved_bytes": metrics.get("peak_reserved_bytes"),
        "memory_scope": metrics.get("memory_scope"),
        "memory_available": metrics.get("peak_allocated_bytes") is not None,
        "memory_note": (
            None
            if metrics.get("peak_allocated_bytes") is not None
            else "Not recorded by the release training report; unavailable is not zero."
        ),
    }, binding


def _load_judgments(entry, config_dir, production):
    _require(isinstance(entry, dict) and set(entry) == {"seed", "path"}, "Bad judgment entry")
    seed = entry["seed"]
    _require(type(seed) is int and 0 <= seed < 2**32, "Judgment seed must be uint32")
    path = _resolve(config_dir, entry["path"])
    result, binding = _verify_input(path)
    joined = read_json(path / "judgments.json")
    primary = [row for row in joined if row.get("kind") == "primary"]
    _require(primary, "No imported primary judgments")
    rows = []
    for row in primary:
        _require(row.get("arm") in ARMS and row.get("verdict") in VERDICTS, "Invalid judgment row")
        _require(
            all(isinstance(row.get(key), str) and row[key] for key in ("prompt_id", "domain", "source_group_id")),
            "Judgment lacks prompt/domain/source-group provenance",
        )
        rows.append(
            {
                "seed": seed,
                "arm": row["arm"],
                "prompt_id": row["prompt_id"],
                "domain": row["domain"],
                "source_group_id": row["source_group_id"],
                "verdict": row["verdict"],
            }
        )
    if production:
        by_arm = Counter(row["arm"] for row in rows)
        _require(all(by_arm[arm] == 120 for arm in by_arm), "Broad denominator is not 120/arm")
        _require(len({row["source_group_id"] for row in rows}) == 73, "Broad panel is not 73 source groups")
        for arm in by_arm:
            _require(
                Counter(row["domain"] for row in rows if row["arm"] == arm)
                == Counter({domain: 12 for domain in DOMAINS}),
                f"Broad domains are not 12 each for {arm}",
            )
    binding.update(kind="judgments", seed=seed, arms=sorted({row["arm"] for row in rows}))
    return rows, binding


def _csv_cell(value):
    if value is None:
        return ""
    if isinstance(value, (list, dict, tuple)):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def _write_csv(path, rows):
    rows = list(rows)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_cell(row.get(field)) for field in fields})


def _latex(value):
    if value is None:
        return ""
    if isinstance(value, float):
        value = f"{value:.6g}"
    text = str(value)
    mapping = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(mapping.get(character, character) for character in text)


def _write_table(output, name, rows):
    rows = list(rows)
    _write_csv(output / "tables" / f"{name}.csv", rows)
    write_json(output / "tables" / f"{name}.json", rows)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    tex = "\\begin{tabular}{" + "l" * max(1, len(fields)) + "}\n\\hline\n"
    if fields:
        tex += " & ".join(_latex(field) for field in fields) + " \\\\\n\\hline\n"
        tex += "".join(
            " & ".join(_latex(row.get(field)) for field in fields) + " \\\\\n"
            for row in rows
        )
    tex += "\\hline\n\\end{tabular}\n"
    path = output / "tables" / f"{name}.tex"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tex, encoding="utf-8", newline="\n")


def _svg_bars(title, subtitle, rows, value_key, denominator_key, output):
    width = 1100
    height = max(300, 110 + 42 * len(rows))
    left, right, top = 250, 1040, 80
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">',
        f"<title>{html.escape(title)}</title><desc>{html.escape(subtitle)}</desc>",
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="Arial, sans-serif" fill="#192d3e">',
        f'<text x="24" y="32" font-size="22" font-weight="bold">{html.escape(title)}</text>',
        f'<text x="24" y="55" font-size="13">{html.escape(subtitle)}</text>',
    ]
    for tick in range(0, 101, 20):
        x = left + (right - left) * tick / 100
        parts.append(f'<line x1="{x}" y1="{top}" x2="{x}" y2="{height-40}" stroke="#dde3e8"/>')
        parts.append(f'<text x="{x}" y="{height-18}" text-anchor="middle" font-size="12">{tick}%</text>')
    for index, row in enumerate(rows):
        y = top + index * 42
        denominator = row[denominator_key]
        value = row[value_key]
        fraction = value / denominator
        label = f"{row['arm']} (seed {row['seed']})"
        parts.append(f'<text x="{left-12}" y="{y+18}" text-anchor="end" font-size="12">{html.escape(label)}</text>')
        parts.append(
            f'<rect x="{left}" y="{y+4}" width="{(right-left)*fraction:.2f}" height="21" fill="#287f76"/>'
        )
        parts.append(f'<text x="{right+8}" y="{y+19}" font-size="12">{value}/{denominator}</text>')
    parts.append("</g></svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(parts), encoding="utf-8", newline="\n")


def _allocation_tables(data_root, production):
    rows = []
    for selector in ("random", "taylor", "guided", "quota_free_guided"):
        path = data_root / "masks" / f"{selector}.json"
        if not path.is_file():
            continue
        mask = read_json(path)
        layers = mask.get("layers")
        _require(isinstance(layers, dict), f"Invalid mask: {path}")
        if production:
            _require(set(layers) == {str(i) for i in range(34)}, f"Mask layers differ: {selector}")
        total = 0
        for layer in sorted((int(key) for key in layers)):
            channels = layers[str(layer)]
            _require(
                len(channels) == len(set(channels))
                and all(type(channel) is int and 0 <= channel < 10240 for channel in channels),
                f"Invalid mask coordinates: {selector}/{layer}",
            )
            total += len(channels)
            rows.append(
                {
                    "selector": selector,
                    "layer": layer,
                    "selected": len(channels),
                    "retained": 10240 - len(channels),
                    "mask_path": str(path),
                    "mask_sha256": sha256(path),
                }
            )
        if production:
            _require(total == 69632, f"Mask budget differs: {selector}")
    return rows


def _allocation_svg(rows, output):
    selected = [row for row in rows if row["selector"] == "quota_free_guided"]
    if not selected:
        return False
    width, height = 1200, 560
    left, top, plot_width, plot_height = 80, 80, 1080, 380
    maximum = max(1, max(row["selected"] for row in selected))
    step = plot_width / len(selected)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        "<title>Quota-free guided selected channels</title>",
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="Arial, sans-serif" fill="#192d3e">',
        '<text x="24" y="32" font-size="22" font-weight="bold">Quota-free guided selected channels</text>',
        '<text x="24" y="54" font-size="13">Exact frozen-mask counts; same coordinates for removal and reset.</text>',
    ]
    for row in selected:
        x = left + (row["layer"] + 0.15) * step
        bar_height = plot_height * row["selected"] / maximum
        y = top + plot_height - bar_height
        parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{step*0.7:.2f}" height="{bar_height:.2f}" fill="#d78137"/>')
        parts.append(f'<text x="{x+step*0.35:.2f}" y="{y-5:.2f}" text-anchor="middle" font-size="9">{row["selected"]}</text>')
        parts.append(f'<text x="{x+step*0.35:.2f}" y="{top+plot_height+20}" text-anchor="middle" font-size="10">{row["layer"]}</text>')
    parts.append(f'<text x="{left+plot_width/2}" y="520" text-anchor="middle" font-size="14">Layer (zero-based)</text>')
    parts.append("</g></svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(parts), encoding="utf-8", newline="\n")
    return True


def _family_expansion_svg(output):
    """Render the fixed discovery-family expansion design, not measured evidence."""
    width, height = 1100, 430
    boxes = (
        (40, 120, 205, 90, "Original problem", "one frozen family source"),
        (325, 70, 205, 90, "English variants", "original + two paraphrases"),
        (325, 245, 205, 90, "Family identity", "kept together in splitting"),
        (610, 70, 205, 90, "Translations", "two languages per English form"),
        (895, 120, 165, 90, "Up to 9 prompts", "after validation/filtering"),
    )
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">',
        "<title>Discovery family expansion design</title>",
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="Arial, sans-serif" fill="#192d3e">',
        '<text x="24" y="34" font-size="22" font-weight="bold">Discovery family expansion design</text>',
        '<text x="24" y="58" font-size="13">Fixed protocol schematic; not a historical result or selector regeneration.</text>',
        '<line x1="245" y1="165" x2="325" y2="115" stroke="#667788" stroke-width="2"/>',
        '<line x1="245" y1="165" x2="325" y2="290" stroke="#667788" stroke-width="2"/>',
        '<line x1="530" y1="115" x2="610" y2="115" stroke="#667788" stroke-width="2"/>',
        '<line x1="815" y1="115" x2="895" y2="165" stroke="#667788" stroke-width="2"/>',
    ]
    for x, y, box_width, box_height, title, subtitle in boxes:
        parts.append(
            f'<rect x="{x}" y="{y}" width="{box_width}" height="{box_height}" '
            'rx="8" fill="#eef4f7" stroke="#527083"/>'
        )
        parts.append(
            f'<text x="{x + box_width / 2}" y="{y + 35}" text-anchor="middle" '
            f'font-size="16" font-weight="bold">{html.escape(title)}</text>'
        )
        parts.append(
            f'<text x="{x + box_width / 2}" y="{y + 61}" text-anchor="middle" '
            f'font-size="12">{html.escape(subtitle)}</text>'
        )
    parts.append(
        '<text x="40" y="390" font-size="13">Rejected, duplicate, ineligible, or unsupported variants remain absent; related variants are not independent samples.</text>'
    )
    parts.append("</g></svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(parts), encoding="utf-8", newline="\n")


def _sensitivity(config, config_dir, output):
    if config is None:
        return None
    _require(
        isinstance(config, dict) and set(config) == {"path", "metadata"},
        "Sensitivity config needs exactly path and metadata",
    )
    metadata = config["metadata"]
    required = {"format", "sha256", "source_run", "data_manifest_sha256", "metric", "aggregation"}
    _require(isinstance(metadata, dict) and set(metadata) == required, "Sensitivity metadata fields differ")
    path = _resolve(config_dir, config["path"])
    _require(path.is_file() and sha256(path) == metadata["sha256"], "Sensitivity checksum differs")
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Sensitivity analysis requires NumPy") from exc
    with np.load(path, allow_pickle=False) as arrays:
        required_arrays = {"mean_abs_sensitivity", "loss_types", "strata"}
        _require(required_arrays <= set(arrays.files), "Sensitivity NPZ arrays differ")
        raw = arrays["mean_abs_sensitivity"]
        loss_types = arrays["loss_types"].tolist()
        strata = arrays["strata"].tolist()
        _require(
            raw.shape == (23, 2, 34, 10240)
            and len(strata) == 23
            and "content" in loss_types
            and np.isfinite(raw).all()
            and (raw >= 0).all(),
            "Sensitivity tensor does not match the documented raw input",
        )
        scores = raw[:, loss_types.index("content")].mean(axis=0)
    summaries = []
    for layer, values in enumerate(scores):
        summaries.append(
            {
                "layer": layer,
                "channels": len(values),
                "mean": float(values.mean()),
                "p10": float(np.quantile(values, 0.1, method="linear")),
                "median": float(np.quantile(values, 0.5, method="linear")),
                "p90": float(np.quantile(values, 0.9, method="linear")),
            }
        )
    _write_table(output, "depth_sensitivity_summary", summaries)
    width, height = 1100, 570
    left, top, plot_width, plot_height = 90, 90, 950, 390
    maximum = max(row["p90"] for row in summaries) or 1
    def points(key):
        return " ".join(
            f"{left + row['layer'] / 33 * plot_width:.2f},"
            f"{top + (1 - row[key] / maximum) * plot_height:.2f}"
            for row in summaries
        )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">'
        "<title>Local content-loss sensitivity across depth</title>"
        '<rect width="100%" height="100%" fill="white"/><g font-family="Arial, sans-serif" fill="#192d3e">'
        '<text x="24" y="32" font-size="22" font-weight="bold">Local content-loss sensitivity across depth</text>'
        '<text x="24" y="54" font-size="13">Raw equal-stratum content derivatives; descriptive channel percentiles, not confidence intervals.</text>'
        f'<polyline points="{points("mean")}" fill="none" stroke="#9d3e67" stroke-width="3"/>'
        f'<polyline points="{points("median")}" fill="none" stroke="#173e64" stroke-width="3"/>'
        f'<text x="{left+plot_width/2}" y="540" text-anchor="middle">Layer (zero-based)</text>'
        "</g></svg>"
    )
    (output / "figures").mkdir(parents=True, exist_ok=True)
    (output / "figures" / "depth_sensitivity.svg").write_text(svg, encoding="utf-8", newline="\n")
    return {**metadata, "path": str(path), "layers": 34, "channels": int(scores.size)}


def _comparison_pairs(available):
    pairs = []
    if "base" in available:
        pairs.extend((arm, "base", "finetuned_minus_base") for arm in available if arm != "base")
    if "ordinary_lora" in available:
        pairs.extend(
            (arm, "ordinary_lora", "intervened_minus_ordinary")
            for arm in available
            if arm not in ("base", "ordinary_lora")
        )
    for selector in ("random", "taylor", "guided", "quota_free_guided"):
        removal, reset = f"{selector}_removal", f"{selector}_reset"
        if removal in available and reset in available:
            pairs.append((reset, removal, "matched_reset_minus_removal"))
    for operation in ("removal", "reset"):
        guided = f"guided_{operation}"
        for comparator in (f"random_{operation}", f"taylor_{operation}"):
            if guided in available and comparator in available:
                pairs.append((guided, comparator, "guided_minus_control"))
        quota = f"quota_free_guided_{operation}"
        if quota in available and guided in available:
            pairs.append((quota, guided, "quota_free_minus_uniform_guided"))
    return pairs


def analyze(config_path, output):
    config_path = Path(config_path).resolve(strict=True)
    config = read_json(config_path)
    allowed = {
        "format",
        "production",
        "data",
        "evaluations",
        "training",
        "judgments",
        "sensitivity",
    }
    _require(isinstance(config, dict) and set(config) <= allowed, "Unknown analysis config fields")
    _require(
        config.get("format") == "remove-reset-analysis-config-v1"
        and type(config.get("production")) is bool
        and isinstance(config.get("data"), str)
        and isinstance(config.get("evaluations"), list)
        and config["evaluations"],
        "Invalid analysis config header",
    )
    for optional in ("training", "judgments"):
        _require(isinstance(config.get(optional, []), list), f"{optional} must be a list")
    output = new_output(output)
    try:
        config_dir = config_path.parent
        production = config["production"]
        provenance = {
            "format": "remove-reset-analysis-provenance-v1",
            "config_path": str(config_path),
            "config_sha256": sha256(config_path),
            "config": config,
            "software": software(),
            "source_identity": source_identity(),
            "inputs": [],
        }
        evaluations = []
        seen = set()
        for entry in config["evaluations"]:
            evaluation, binding = _load_evaluation(entry, config_dir, production)
            identity = (evaluation["seed"], evaluation["arm"])
            _require(identity not in seen, "Duplicate seed/arm evaluation")
            seen.add(identity)
            evaluations.append(evaluation)
            provenance["inputs"].append(binding)
        by_seed = defaultdict(list)
        for evaluation in evaluations:
            by_seed[evaluation["seed"]].append(evaluation)
        for seed, seed_evaluations in by_seed.items():
            reference = _panel_identity(seed_evaluations[0])
            data_hash = seed_evaluations[0]["provenance"].get("data_manifest_sha256")
            for evaluation in seed_evaluations[1:]:
                _require(_panel_identity(evaluation) == reference, f"Stale/mismatched panels at seed {seed}")
                _require(
                    evaluation["provenance"].get("data_manifest_sha256") == data_hash,
                    f"Data manifest differs at seed {seed}",
                )

        primary = [_evaluation_outcome(evaluation) for evaluation in evaluations]
        primary.sort(key=lambda row: (row["seed"], ARMS.index(row["arm"])))
        _write_table(output, "primary_outcomes", primary)

        contrast_rows = []
        bootstrap_arrays = {}
        for seed, seed_evaluations in sorted(by_seed.items()):
            available = {evaluation["arm"] for evaluation in seed_evaluations}
            sql_conditions = {
                evaluation["arm"]: [
                    {**row, "raw_value": row["correct"]}
                    for row in evaluation["records"]["sql"]
                ]
                for evaluation in seed_evaluations
            }
            sql_sampled = cluster_bootstrap(
                sql_conditions, cluster_key="db_id", value_key="raw_value"
            )
            graph_conditions = {
                evaluation["arm"]: [
                    {**row, "raw_value": row["inferred_correct"]}
                    for row in evaluation["records"]["reasoning"]
                ]
                for evaluation in seed_evaluations
            }
            graph_sampled = cluster_bootstrap(
                graph_conditions, cluster_key="graph_id", value_key="raw_value"
            )
            graph_by_task = {
                task: cluster_bootstrap(
                    {
                        arm: [row for row in rows if row["task"] == task]
                        for arm, rows in graph_conditions.items()
                    },
                    cluster_key="graph_id",
                    value_key="raw_value",
                )
                for task in ("direct", "two_step")
            }
            bootstrap_arrays[f"sql_seed_{seed}_rates"] = sql_sampled["rates"]
            bootstrap_arrays[f"graph_seed_{seed}_rates"] = graph_sampled["rates"]
            bootstrap_arrays[f"sql_seed_{seed}_groups"] = sql_sampled["groups"]
            bootstrap_arrays[f"sql_seed_{seed}_draw_indices"] = [
                [sql_sampled["groups"].index(group) for group in draw]
                for draw in sql_sampled["draws"]
            ]
            bootstrap_arrays[f"graph_seed_{seed}_groups"] = graph_sampled["groups"]
            bootstrap_arrays[f"graph_seed_{seed}_draw_indices"] = [
                [graph_sampled["groups"].index(group) for group in draw]
                for draw in graph_sampled["draws"]
            ]
            for task, sampled in graph_by_task.items():
                bootstrap_arrays[f"graph_{task}_seed_{seed}_rates"] = sampled["rates"]
            for condition, comparator, family in _comparison_pairs(available):
                metric_samples = [
                    ("sql", sql_sampled),
                    ("graph_all_tasks", graph_sampled),
                    ("graph_direct", graph_by_task["direct"]),
                    ("graph_two_step", graph_by_task["two_step"]),
                ]
                for metric, sampled in metric_samples:
                    contrast = named_contrast(sampled, condition, comparator)
                    contrast_rows.append(
                        {
                            "seed": seed,
                            "family": family,
                            "metric": metric,
                            "condition": condition,
                            "comparator": comparator,
                            "direction": contrast["direction"],
                            "difference_pp": 100 * contrast["difference"],
                            "interval_95_low_pp": 100 * contrast["interval_95"][0],
                            "interval_95_high_pp": 100 * contrast["interval_95"][1],
                            "clusters": len(sampled["groups"]),
                            "replicates": sampled["replicates"],
                            "bootstrap_seed": sampled["seed"],
                        }
                    )
        _write_table(output, "paired_sql_graph_contrasts", contrast_rows)

        training_rows = []
        for entry in config.get("training", []):
            row, binding = _load_training(entry, config_dir, production)
            training_rows.append(row)
            provenance["inputs"].append(binding)
        training_rows.sort(key=lambda row: (row["seed"], ARMS.index(row["arm"])))
        _write_table(output, "training_memory", training_rows)

        judgment_rows = []
        broad_outcomes = []
        broad_contrasts = []
        broad_protocol = []
        for entry in config.get("judgments", []):
            rows, binding = _load_judgments(entry, config_dir, production)
            provenance["inputs"].append(binding)
            judgment_rows.extend(rows)
            seed = entry["seed"]
            supplied_eval_arms = {evaluation["arm"] for evaluation in by_seed.get(seed, [])}
            arms = tuple(arm for arm in ARMS if arm in supplied_eval_arms and any(r["arm"] == arm for r in rows))
            _require(arms, f"No judgment/evaluation arm intersection at seed {seed}")
            selected = [row for row in rows if row["arm"] in arms]
            for arm in arms:
                outcome = unresolved_bounds([row for row in selected if row["arm"] == arm])
                broad_outcomes.append({"seed": seed, "arm": arm, **outcome})
            sampled = broad_bootstrap(selected, arms=arms)
            broad_protocol.append(
                {
                    "seed": seed,
                    "groups": len(sampled["groups"]),
                    "accepted_draws": sampled["replicates"],
                    "attempts": sampled["attempts"],
                    "empty_domain_redraws": sampled["empty_domain_redraws"],
                    "rng": sampled["rng"],
                }
            )
            bootstrap_arrays[f"broad_seed_{seed}_rates"] = sampled["rates"]
            bootstrap_arrays[f"broad_seed_{seed}_groups"] = sampled["groups"]
            bootstrap_arrays[f"broad_seed_{seed}_weights"] = sampled["weights"].tolist()
            bootstrap_arrays[f"broad_seed_{seed}_rejected_domain_masks"] = sampled[
                "rejected_domain_masks"
            ].tolist()
            for condition, comparator, family in _comparison_pairs(set(arms)):
                contrast = named_contrast(sampled, condition, comparator)
                broad_contrasts.append(
                    {
                        "seed": seed,
                        "family": family,
                        "condition": condition,
                        "comparator": comparator,
                        "direction": contrast["direction"],
                        "difference_pp": 100 * contrast["difference"],
                        "interval_95_low_pp": 100 * contrast["interval_95"][0],
                        "interval_95_high_pp": 100 * contrast["interval_95"][1],
                        "source_groups": len(sampled["groups"]),
                        "replicates": sampled["replicates"],
                        "bootstrap_seed": sampled["seed"],
                    }
                )
        _write_table(output, "broad_outcomes", broad_outcomes)
        _write_table(output, "paired_broad_contrasts", broad_contrasts)
        _write_table(output, "broad_bootstrap_protocol", broad_protocol)

        data_root = _resolve(config_dir, config["data"])
        _require(data_root.is_dir(), "Analysis data directory does not exist")
        allocations = _allocation_tables(data_root, production)
        _write_table(output, "layer_allocations", allocations)
        allocation_figure = _allocation_svg(
            allocations, output / "figures" / "quota_free_allocation.svg"
        )
        _family_expansion_svg(output / "figures" / "family_expansion.svg")
        sensitivity = _sensitivity(config.get("sensitivity"), config_dir, output)

        for seed, rows in sorted(
            (seed, [row for row in primary if row["seed"] == seed]) for seed in by_seed
        ):
            _svg_bars(
                f"SQL outcomes, seed {seed}",
                "Execution agreement inferred from complete raw SQL records.",
                rows,
                "sql_correct",
                "sql_denominator",
                output / "figures" / f"sql_outcomes_seed_{seed}.svg",
            )
            broad = [row for row in broad_outcomes if row["seed"] == seed]
            if broad:
                _svg_bars(
                    f"Broad outcomes, seed {seed}",
                    "Known Pass over all 120 judgments; Uncertain remains in the denominator.",
                    broad,
                    "Pass",
                    "denominator",
                    output / "figures" / f"broad_outcomes_seed_{seed}.svg",
                )

        availability = []
        evaluation_keys = {(row["seed"], row["arm"]) for row in primary}
        training_keys = {(row["seed"], row["arm"]) for row in training_rows}
        judgment_keys = {(row["seed"], row["arm"]) for row in broad_outcomes}
        for seed in sorted(by_seed):
            for arm in ARMS:
                availability.append(
                    {
                        "kind": "condition",
                        "seed": seed,
                        "arm": arm,
                        "evaluation": (seed, arm) in evaluation_keys,
                        "training": arm == "base" or (seed, arm) in training_keys,
                        "judgments": (seed, arm) in judgment_keys,
                    }
                )
        availability.append(
            {
                "kind": "figure",
                "seed": None,
                "arm": "depth_sensitivity",
                "available": sensitivity is not None,
                "missing_requirement": (
                    None
                    if sensitivity is not None
                    else "validated pre-intervention sensitivity NPZ plus calibration provenance"
                ),
                "evaluation": None,
                "training": None,
                "judgments": None,
            }
        )
        availability.append(
            {
                "kind": "figure",
                "seed": None,
                "arm": "quota_free_allocation",
                "available": allocation_figure,
                "missing_requirement": None if allocation_figure else "quota-free guided mask",
                "evaluation": None,
                "training": None,
                "judgments": None,
            }
        )
        _write_table(output, "availability", availability)

        write_json(
            output / "bootstrap_protocol.json",
            {
                "format": "remove-reset-bootstrap-protocol-v1",
                "seed": BOOTSTRAP_SEED,
                "replicates": BOOTSTRAP_DRAWS,
                "sql": "random.Random.choices over sorted database IDs; query-weighted; shared draws",
                "graph": "random.Random.choices over sorted graph IDs; both tasks retained; shared draws",
                "broad": "NumPy PCG64 global source groups; equal-domain known-Pass; empty-domain attempts rejected",
                "percentiles": "linear interpolation at 2.5 and 97.5 percent",
                "comparison_direction": "explicit named condition minus comparator",
                "uncertain": "retained in denominator; P/N..(P+U)/N are unresolved bounds, not intervals",
                "broad_runs": broad_protocol,
            },
        )
        write_json(output / "source_provenance.json", provenance)
        write_json(output / "bootstrap_values.json", bootstrap_arrays)

        complete = all(
            row["evaluation"] and row["training"] and row["judgments"]
            for row in availability
            if row["seed"] is not None
        ) and sensitivity is not None
        status = "complete" if complete else "partial"
        missing = [
            {
                "seed": row["seed"],
                "arm_or_panel": row["arm"],
                "missing": (
                    [row["missing_requirement"]]
                    if row["kind"] == "figure"
                    else [
                        key
                        for key in ("evaluation", "training", "judgments")
                        if row[key] is False
                    ]
                ),
            }
            for row in availability
            if (
                row["kind"] == "figure"
                and row.get("available") is False
                or row["kind"] == "condition"
                and any(row[key] is False for key in ("evaluation", "training", "judgments"))
            )
        ]
        return save_result(
            output,
            ToolResult(
                title="Publication analysis",
                summary=(
                    "Regenerated tables and figures from sealed reader outputs."
                    if status == "complete"
                    else "Regenerated the supplied subset; unavailable manuscript panels are explicit."
                ),
                metrics={
                    "status": status,
                    "evaluation_runs": len(evaluations),
                    "training_reports": len(training_rows),
                    "judgment_rows": len(judgment_rows),
                    "analysis_seeds": sorted(by_seed),
                    "sensitivity_figure_available": sensitivity is not None,
                },
                tables={
                    "primary_outcomes": primary,
                    "paired_sql_graph_contrasts": contrast_rows,
                    "training_memory": training_rows,
                    "broad_outcomes": broad_outcomes,
                    "paired_broad_contrasts": broad_contrasts,
                    "missing_panels": missing,
                    "sensitivity_provenance": [] if sensitivity is None else [sensitivity],
                },
                figures={
                    "directory": "figures",
                    "svg_only": True,
                    "historical_results_bundled": False,
                },
                notes=[
                    "All comparisons are paired within the same training seed; trials are never pooled.",
                    "Intervals condition on observed outputs and do not estimate retraining uncertainty.",
                    "No judge is invoked: only sealed imported independent judgments are analyzed.",
                    "The depth figure is unavailable without the declared raw numerical sensitivity NPZ.",
                ],
            ),
            status=status,
        )
    except BaseException as exc:
        if not (output / "result.json").exists():
            save_result(
                output,
                ToolResult(
                    title="Publication analysis failed",
                    summary="No incomplete or stale input was accepted as evidence.",
                ),
                status="failed",
                error=exc,
            )
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    analyze(args.config, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
