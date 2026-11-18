"""Build and validate the study's training token cache using a local tokenizer."""

import argparse
import json
from pathlib import Path

from validate import ROOT  # noqa: F401

from remove_reset.data import DataRoot, training_tokens  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    if args.workspace.resolve().is_relative_to((ROOT / "data").resolve()):
        raise ValueError("Token caches must stay outside the published data folder")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer_dir.resolve(strict=True)),
        local_files_only=True,
        trust_remote_code=False,
    )
    _, summary = training_tokens(DataRoot(args.data_root), tokenizer, args.workspace)
    print(json.dumps({"status": "verified", **summary}, indent=2))


if __name__ == "__main__":
    main()
