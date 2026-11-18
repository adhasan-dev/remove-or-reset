"""Check separately supplied final discovery artifacts against recorded study hashes."""

import argparse
import json
from pathlib import Path

from validate import ROOT

from remove_reset.artifacts import read_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", type=Path)
    parser.add_argument("--variants", type=Path)
    args = parser.parse_args()
    if not args.families and not args.variants:
        parser.error("Supply --families and/or --variants; this script does not generate them")
    bindings = read_json(ROOT / "data" / "acquisition.json")["guided_selector_inputs"]
    expected = {b["id"]: b for b in bindings["expected_artifacts"]}
    checked = {}
    for name in ("families", "variants"):
        path = getattr(args, name)
        if path is not None:
            if sha256(path) != expected[name]["sha256"]:
                raise ValueError(f"{name} do not match the frozen study; not an exact reproduction")
            checked[name] = expected[name]["sha256"]
    print(json.dumps({"status": "verified", "files": checked}, indent=2))


if __name__ == "__main__":
    main()
