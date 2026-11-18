"""Replay deterministic discovery candidate filtering on lawfully acquired files.

No model screening, paraphrasing, translation, or LLM generation is implemented.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import discovery_preparation as prep  # noqa: E402
from validate import ROOT

from remove_reset.artifacts import (
    local_path,
    read_json,
    sha256,
    value_hash,
    write_json,
    write_jsonl,
)


def build(raw, output):
    lock = read_json(ROOT / "data" / "recipes" / "discovery_sources.json")
    for source in lock["sources"]:
        for binding in source["files"]:
            path = local_path(raw, binding["path"])
            if sha256(path) != binding["sha256"]:
                raise ValueError(f"Acquired discovery file differs: {binding['path']}")
    if output.resolve().is_relative_to((ROOT / "data").resolve()):
        raise ValueError("Downloaded derivatives must stay outside the published data folder")
    output.mkdir(parents=True, exist_ok=False)
    buckets, seen, rejected = defaultdict(list), set(), []
    for source in lock["sources"]:
        for row in prep.source_rows(raw, source):
            content = value_hash(prep.normalized(row["original_input"]))
            if content in seen:
                rejected.append(
                    {
                        "family_id": row["family_id"],
                        "reason": "duplicate_input",
                        "content_sha256": content,
                    }
                )
                continue
            seen.add(content)
            if len(row["input"]) > 6500:
                rejected.append(
                    {
                        "family_id": row["family_id"],
                        "reason": "character_length",
                        "input_characters": len(row["input"]),
                    }
                )
                continue
            buckets[(row["source"], row["source_task"])].append(row)
    candidates = []
    for key, rows in sorted(buckets.items()):
        ordered = sorted(rows, key=lambda r: value_hash([prep.FORMAT, prep.SEED, r["family_id"]]))
        cap = ordered[0]["quota_per_task"] * (1 if key[0] in ("gsm8k", "mbpp") else 3)
        candidates.extend(ordered[:cap])
    candidates.sort(
        key=lambda r: (r["source"], r["source_task"], value_hash([prep.SEED, r["family_id"]]))
    )
    prep.validate_families(candidates)
    write_jsonl(output / "candidates.jsonl", candidates)
    write_jsonl(output / "preparation_exclusions.jsonl", rejected)
    if sha256(output / "candidates.jsonl") != lock["expected_candidates_sha256"]:
        raise ValueError("Discovery candidates differ from the historical frozen pool")
    report = {
        "status": "verified",
        "candidates": len(candidates),
        "coverage": prep.coverage(candidates),
        "exclusions": len(rejected),
    }
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(build(args.raw, args.output))


if __name__ == "__main__":
    main()
