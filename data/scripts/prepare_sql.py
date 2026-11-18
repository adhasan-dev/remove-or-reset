"""Rebuild frozen SQL splits from verified, reader-acquired sources; no downloads."""

import argparse
import collections
import json
import logging
import sys
from pathlib import Path

import sql_preparation as prep
from validate import prepared, sources

ROOT = Path(__file__).resolve().parents[2]
logging.getLogger("sqlglot").setLevel(logging.ERROR)


def build(raw, output):
    sources(raw)
    if output.resolve().is_relative_to((ROOT / "data").resolve()):
        raise ValueError("Downloaded derivatives must stay outside the published data folder")
    output.mkdir(parents=True, exist_ok=False)
    prep.RAW = raw.resolve()
    prep.ROOT = raw.resolve().parent
    prep.safe_extract_spider(raw / "spider-official-archive.zip")
    counts = collections.Counter()
    rejected = []
    spider, compatibility = prep.prepare_spider(rejected, counts)
    gretel = prep.prepare_gretel(rejected, counts)
    dev = [r for r in spider if r["metadata"]["original_split"] == "validation"]
    protected = {r["schema_id"] for r in dev}
    pools = {name: [] for name in ("train", "selection", "validation")}
    seen = set()
    for row in [r for r in spider if r["metadata"]["original_split"] == "train"] + gretel:
        identity = prep.stable_hash(
            prep.canonical_json(
                [
                    row["question"].strip().casefold(),
                    row["schema_id"],
                    " ".join(row["sql"].split()).casefold(),
                ]
            )
        )
        if identity in seen:
            counts["exact_duplicates_removed"] += 1
            continue
        seen.add(identity)
        if row["schema_id"] in protected:
            counts["protected_dev_schema_exclusions"] += 1
            continue
        split = prep.assignment(row["schema_id"])
        row["metadata"]["prepared_split"] = split
        pools[split].append(row)
    for name, rows in {**pools, "spider_dev": dev}.items():
        prep.write_jsonl(output / f"{name}.jsonl", rows)
    result = prepared(output)
    prep.write_json(
        output / "preparation_report.json",
        {
            "status": "verified",
            "rows": result,
            "counts": dict(counts),
            "compatibility": compatibility,
            "rejected_before_deduplication": len(rejected),
            "source_sha256": sources(raw),
            "script_sha256": prep.sha256_file(Path(__file__)),
            "helper_sha256": prep.sha256_file(Path(prep.__file__)),
            "python": sys.version,
            "sqlite": prep.sqlite3.sqlite_version,
            "sqlglot": prep.sqlglot.__version__,
            "pyarrow": prep.pq.__version__
            if hasattr(prep.pq, "__version__")
            else __import__("pyarrow").__version__,
        },
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.raw, args.output)
    print(json.dumps({"status": "verified", "splits": result}, indent=2))


if __name__ == "__main__":
    main()
