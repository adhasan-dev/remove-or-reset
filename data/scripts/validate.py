"""Verify released originals, acquired source bytes, or reader-built runtime data."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from remove_reset.artifacts import read_json, sha256, value_hash  # noqa: E402
from remove_reset.data import SPLITS, DataRoot, validate_records  # noqa: E402


def sources(raw):
    acquisition = read_json(ROOT / "data" / "acquisition.json")
    bindings = acquisition["source_pins"]["prepared_source_files"] + [
        {
            "id": "spider-official-archive",
            **acquisition["official_spider_databases"]["official_archive"],
        }
    ]
    checked = {}
    for item in bindings:
        suffix = ".zip" if item["id"] == "spider-official-archive" else ".parquet"
        path = Path(raw) / (item["id"] + suffix)
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise ValueError(f"Missing/changed acquisition: {path.name}")
        if path.stat().st_size != item.get("size", item.get("bytes")):
            raise ValueError(f"Acquisition byte count differs: {path.name}")
        checked[item["id"]] = item["sha256"]
    return checked


def prepared(directory):

    expected = read_json(ROOT / "data" / "external_manifest.json")["files"]
    rows = {}
    checked = {}
    for split in SPLITS:
        name = f"sql/{split}.jsonl.gz"
        path = Path(directory) / f"{split}.jsonl"
        binding = expected[name]
        if sha256(path) != binding["uncompressed_sha256"]:
            raise ValueError(f"Prepared bytes differ from the study: {split}")
        rows[split] = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
        validate_records(rows[split])
        if len(rows[split]) != SPLITS[split]:
            raise ValueError(f"Prepared row count differs: {split}")
        checked[split] = len(rows[split])
    for i, a in enumerate(SPLITS):
        for b in list(SPLITS)[i + 1 :]:
            if {r["schema_id"] for r in rows[a]} & {r["schema_id"] for r in rows[b]}:
                raise ValueError(f"Schema leakage: {a}, {b}")
    return checked


def runtime(directory):
    from remove_reset.config import REASONING_SHA256, SQL_ID_SHA256, VALIDATION_ID_SHA256

    data = DataRoot(directory)
    checked = data.check()
    panels = read_json(data.path("sql/evaluation_panels.json"))
    if value_hash(sorted(r["id"] for r in panels["sql"])) != SQL_ID_SHA256:
        raise ValueError("SQL evaluation IDs differ")
    if value_hash(sorted(r["id"] for r in panels["validation"])) != VALIDATION_ID_SHA256:
        raise ValueError("Validation IDs differ")
    if value_hash(panels["reasoning"]) != REASONING_SHA256:
        raise ValueError("Reasoning records differ")
    for key in ("sql", "validation", "reasoning"):
        validate_records(panels[key])
    return {
        "files": len(checked),
        "panels": {k: len(panels[k]) for k in ("sql", "validation", "reasoning")},
    }


def originals():
    data = DataRoot(ROOT / "data")
    checked = data.check()
    unlisted = (
        {
            p.relative_to(data.root).as_posix()
            for p in data.root.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts
        }
        - set(checked)
        - {"manifest.json", "README.md", "LICENSE.md", "DISCOVERY.md"}
    )
    unlisted = {
        name
        for name in unlisted
        if not (name.startswith("scripts/") and name.endswith((".py", "/requirements.txt")))
    }
    if unlisted:
        raise ValueError(f"Undeclared data payloads: {sorted(unlisted)}")
    reasoning = data.rows("original/reasoning.jsonl")
    validate_records(reasoning)
    if len(reasoning) != 128:
        raise ValueError("Expected 128 reasoning probes")
    for row in reasoning:
        target = row["start"]
        for _ in range(row["steps"]):
            target = row["edges"][target]
        if target != row["target"]:
            raise ValueError(f"Incorrect graph reference: {row['id']}")
    probes = read_json(data.path("original/taylor_auxiliary.json"))
    for rows in probes["panels"].values():
        validate_records(rows)
        if any("entry" in r or r["family"] == "sql" for r in rows):
            raise ValueError("Downloaded SQL leaked into original probes")
    from remove_reset.evaluation import heldout_inputs

    heldout_inputs(data)
    from remove_reset.interventions import normalize_mask, validate_mask

    for name in ("random", "taylor", "guided", "quota_free_guided"):
        validate_mask(normalize_mask(read_json(data.path(f"masks/{name}.json"))), study=True)
    return {
        "files": len(checked),
        "reasoning": len(reasoning),
        "taylor_auxiliary": sum(map(len, probes["panels"].values())),
        "heldout": 120,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--raw", type=Path)
    group.add_argument("--prepared", type=Path)
    group.add_argument("--runtime", type=Path)
    args = parser.parse_args()
    try:
        if args.raw:
            result = sources(args.raw)
        elif args.prepared:
            result = prepared(args.prepared)
        elif args.runtime:
            result = runtime(args.runtime)
        else:
            result = originals()
        print(json.dumps({"status": "verified", "checks": result}, indent=2))
    except (ValueError, OSError, KeyError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
