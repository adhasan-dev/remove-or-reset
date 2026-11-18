"""Combine original inputs and independently prepared SQL in a NEW local data root."""

import argparse
import gzip
import json
import shutil
from pathlib import Path

from validate import ROOT, prepared, runtime

from remove_reset.artifacts import (  # noqa: E402
    read_json,
    read_jsonl,
    sha256,
    value_hash,
    write_json,
)
from remove_reset.data import DataRoot, encode_record  # noqa: E402


def zipped(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with (
        path.open("xb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream,
    ):
        stream.write(payload)


def build(directory, base_dir, output):
    prepared(directory)
    if output.resolve().is_relative_to((ROOT / "data").resolve()):
        raise ValueError("Runtime data must stay outside the published data folder")
    originals = DataRoot(ROOT / "data")
    originals.check()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(base_dir.resolve(strict=True)), local_files_only=True, trust_remote_code=False
    )
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(json.dumps(originals.manifest))
    manifest["distribution_policy"] = (
        "READER_LOCAL_ONLY: contains independently acquired source derivatives; "
        "not the public original-only data folder and not cleared for redistribution."
    )
    manifest["publication_status"] = {
        "reader_local_only": True,
        "third_party_payloads_included": True,
        "publication_ready": False,
    }
    for name in manifest["files"]:
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(originals.path(name), target)
    expected = read_json(ROOT / "data" / "external_manifest.json")["files"]
    for split in ("train", "selection", "validation", "spider_dev"):
        name = f"sql/{split}.jsonl.gz"
        zipped(output / name, (directory / f"{split}.jsonl").read_bytes())

    recipe = read_json(originals.path("recipes/evaluation.json"))
    dev = {r["id"]: r for r in read_jsonl(directory / "spider_dev.jsonl")}
    validation = {r["id"]: r for r in read_jsonl(directory / "validation.jsonl")}
    panels = {k: recipe[k] for k in ("format", "seed")}
    panels.update(
        sql=[dev[i] for i in recipe["sql_ids"]],
        validation=[validation[i] for i in recipe["validation_ids"]],
        reasoning=originals.rows("original/reasoning.jsonl"),
        databases={
            name: {key: item[key] for key in ("path", "sha256", "archive_member")}
            for name, item in recipe["databases"].items()
        },
        database_archive=recipe["database_archive"],
    )
    # Match the historical release's insertion order and pretty-printed JSON bytes.
    (output / "sql" / "evaluation_panels.json").write_text(
        json.dumps(panels, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    recipe = read_json(originals.path("recipes/taylor.json"))
    auxiliary = read_json(originals.path("original/taylor_auxiliary.json"))
    rows = read_jsonl(directory / "selection.jsonl")
    by_id = {r["id"]: (i, r) for i, r in enumerate(rows)}
    entries = {}
    calibration = {"metadata": recipe["metadata"], "panels": {}}
    for part, selected in recipe["panels"].items():
        sql = []
        for wrapper in selected:
            index, row = by_id[wrapper["id"]]
            encoded = encode_record(tokenizer, row).validate()
            entry = {
                "id": row["id"],
                "prepared_split": "selection",
                "source": row["source"],
                "source_index": index,
                "prompt_ids": encoded.input_ids[: encoded.prompt_length],
                "target_ids": encoded.input_ids[encoded.prompt_length :],
                "prompt_length": encoded.prompt_length,
                "target_tokens": encoded.target_tokens,
                "total_length": len(encoded.input_ids),
            }
            sql.append({**wrapper, "entry": entry})
        entries[part] = [r["entry"] for r in sql]
        calibration["panels"][part] = sql + auxiliary["panels"][part]
    if value_hash(entries) != recipe["expected_sql_entries_sha256"]:
        raise ValueError("Taylor SQL tokens differ: check the pinned local tokenizer")
    zipped(
        output / "selection" / "taylor_calibration.json.gz",
        (json.dumps(calibration, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(),
    )
    for name, binding in expected.items():
        path = output / name
        actual = sha256(path)
        if actual != binding["sha256"]:
            raise ValueError(f"Rebuilt payload differs from the frozen study: {name}")
        manifest["files"][name] = binding
    write_json(output / "manifest.json", manifest)
    return runtime(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.prepared, args.tokenizer_dir, args.output), indent=2))


if __name__ == "__main__":
    main()
