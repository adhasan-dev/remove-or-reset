"""Replay the 43 final inspection holds on separately supplied LLM variants.

This is filtering only. It never calls an LLM or generates a variant.
"""

import argparse
from pathlib import Path

from validate import ROOT

from remove_reset.artifacts import read_json, read_jsonl, sha256, write_jsonl


def build(source, output):
    recipe = read_json(ROOT / "data" / "recipes" / "variant_filter.json")
    if sha256(source) != recipe["input_sha256"]:
        raise ValueError("Expected the exact historical pre-inspection variants")
    if output.resolve().is_relative_to((ROOT / "data").resolve()):
        raise ValueError("Downloaded derivatives must stay outside the published data folder")
    rows = read_jsonl(source)
    if len(rows) != recipe["input_count"] or len({r["variant_id"] for r in rows}) != len(rows):
        raise ValueError("Pre-inspection variants have incorrect count or duplicate IDs")
    held = set(recipe["held_variant_ids"])
    if not held <= {r["variant_id"] for r in rows}:
        raise ValueError("Missing inspection-held variant IDs")
    retained = [r for r in rows if r["variant_id"] not in held]
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / "variants.jsonl", retained)
    if (
        len(retained) != recipe["output_count"]
        or sha256(output / "variants.jsonl") != recipe["output_sha256"]
    ):
        raise ValueError("Filtered variants differ from the frozen study")
    return {"status": "verified", "retained": len(retained), "held": len(held)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(build(args.input, args.output))


if __name__ == "__main__":
    main()
