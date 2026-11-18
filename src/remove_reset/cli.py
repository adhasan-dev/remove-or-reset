"""Argparse transport. Scientific implementations live in pure Python modules."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .artifacts import (
    new_output,
    parse_json,
    redact_error,
    save_result,
    software,
    source_identity,
    verify_bundle,
    write_json,
)
from .config import ARMS, Recipe


def _data_args(parser):
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Directory containing the released data/manifest.json",
    )


def _model_args(parser):
    _data_args(parser)
    parser.add_argument(
        "--base-dir",
        type=Path,
        required=True,
        help="Explicit local pinned snapshot or HF cache root; never downloads",
    )
    parser.add_argument("--device", default="cuda:0", help="Study requires CUDA BF16, no fallback")


def _evaluation_args(parser):
    _model_args(parser)
    parser.add_argument("--arm", choices=("base", *ARMS), required=True)
    parser.add_argument("--prepared", type=Path, help="Reader-generated intervention preparation")
    parser.add_argument("--training-run", type=Path, help="Completed full-epoch LoRA run")
    parser.add_argument(
        "--native-only",
        action="store_true",
        help="Explicit pre-LoRA intervention evaluation, not a recovered arm",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="NEW empty/nonexistent directory"
    )


def parser():
    root = argparse.ArgumentParser(
        prog="remove-reset",
        description="Local reproduction source: no downloads, remote inference, UI, or bundled weights.",
    )
    root.add_argument("--version", action="version", version="remove-reset 0.1.0")
    sub = root.add_subparsers(dest="command", required=True)
    listing = sub.add_parser("list", help="List registered analysis ToolSpecs and all nine arms")
    listing.add_argument("--json", action="store_true")
    check = sub.add_parser(
        "check", help="CPU/stdlib data checksums or reader-artifact verification"
    )
    _data_args(check)
    check.add_argument("--artifact", type=Path, help="Verify artifacts instead of data")
    check.add_argument("--json", action="store_true")
    run = sub.add_parser("run", help="Run a registered pure-Python analysis ToolSpec")
    run.add_argument("tool")
    run.add_argument("--param", action="append", default=[], metavar="NAME=VALUE")
    run.add_argument("--output", type=Path, required=True)
    materialize = sub.add_parser(
        "materialize-databases",
        help="Extract only declared DBs from a manually obtained pinned archive",
    )
    _data_args(materialize)
    materialize.add_argument("--archive", type=Path, required=True)
    materialize.add_argument("--output", type=Path, required=True)
    prepare = sub.add_parser(
        "prepare", help="Apply a frozen mask to a local base; safe native save/reload"
    )
    _model_args(prepare)
    prepare.add_argument(
        "--arm", choices=tuple(a for a in ARMS if a != "ordinary_lora"), required=True
    )
    prepare.add_argument("--output", type=Path, required=True)
    preflight = sub.add_parser(
        "preflight", help="Produce a fresh technical gate; no old approval/result needed"
    )
    _model_args(preflight)
    preflight.add_argument("--workspace", type=Path, required=True)
    preflight.add_argument("--database-root", type=Path)
    preflight.add_argument("--config", type=Path)
    preflight.add_argument("--output", type=Path, required=True)
    train = sub.add_parser(
        "train", help="Fixed full epoch, with safe optimizer-boundary pause/resume"
    )
    _model_args(train)
    train.add_argument("--arm", choices=ARMS, required=True)
    train.add_argument("--prepared", type=Path)
    train.add_argument("--preflight", type=Path, required=True)
    train.add_argument("--workspace", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--config", type=Path)
    train.add_argument("--resume", action="store_true")
    train.add_argument("--pause-at-update", type=int)
    evaluation = sub.add_parser("evaluate", help="Frozen primary SQL/NLL/reasoning evaluation")
    _evaluation_args(evaluation)
    evaluation.add_argument("--database-root", type=Path)
    evaluation.add_argument("--budget-seconds", type=float)
    heldout = sub.add_parser(
        "generate-heldout", help="120 gold-free explicit-greedy secondary generations"
    )
    _evaluation_args(heldout)
    export = sub.add_parser(
        "judge-export", help="Export blinded cases, no judge/network invocation"
    )
    _data_args(export)
    export.add_argument(
        "--generations",
        type=Path,
        nargs="+",
        required=True,
        help="Ten complete generation directories (base plus nine arms)",
    )
    export.add_argument("--output", type=Path, required=True)
    importing = sub.add_parser(
        "judge-import", help="Strictly import six independent P/F/U verdict files"
    )
    importing.add_argument("--export", type=Path, required=True)
    importing.add_argument("--verdicts", type=Path, nargs="+", required=True)
    importing.add_argument("--output", type=Path, required=True)
    tiny = sub.add_parser("tiny-check", help="Real bounded CPU fixture; NOT the 4B reproduction")
    tiny.add_argument("--output", type=Path, required=True)
    tiny.add_argument("--seed", type=int, default=42, help="Fixture LoRA and sampler seed")
    return root


def _params(values):
    result = {}
    for item in values:
        if "=" not in item:
            raise ValueError("Each --param must be NAME=VALUE")
        key, value = item.split("=", 1)
        if key in result:
            raise ValueError(f"Duplicate parameter: {key}")
        try:
            value = parse_json(value)
        except (json.JSONDecodeError, ValueError):
            # Plain path/text is allowed, but JSON nonfinite constants are not.
            if value in ("NaN", "Infinity", "-Infinity"):
                raise ValueError("Nonfinite parameter")
        result[key] = value
    return result


def dispatch(args):
    if args.command == "list":
        from .interventions import get_tools

        return {
            "arms": list(ARMS),
            "evaluation_control": "base",
            "tools": [tool.describe() for tool in get_tools()],
        }
    if args.command == "check":
        if args.artifact is not None:
            return verify_bundle(args.artifact, complete=False)
        from .data import DataRoot

        data = DataRoot(args.data_root)
        return {
            "status": "verified",
            "data_manifest_sha256": data.fingerprint,
            "files": data.check(),
            "notes": [
                "Checksums are not license grants. Read LICENSING.md before publication.",
                "No GPU, model, or SQLite database payload is loaded by this check.",
            ],
        }
    if args.command == "run":
        from .contracts import ToolContext, ToolResult
        from .interventions import get_tool

        output = new_output(args.output)
        try:
            spec = get_tool(args.tool)
            parameters = spec.validate(_params(args.param))
            write_json(
                output / "config.json",
                {
                    "tool": spec.id,
                    "params": parameters,
                    "seed": 42,
                    "source_identity": source_identity(),
                    "software": software(),
                },
            )
            result = spec.run(ToolContext(model=None, seed=42, run_dir=output), parameters)
            return save_result(output, result)
        except BaseException as exc:
            save_result(
                output,
                ToolResult(title="Analysis failed", summary="No metric inferred."),
                status="failed",
                error=exc,
            )
            raise
    if args.command == "tiny-check":
        from .fixture import tiny_check

        return tiny_check(args.output, seed=args.seed)
    if args.command == "judge-import":
        from .judging import import_judge

        return import_judge(args.export, args.verdicts, args.output)
    from .data import DataRoot

    data = DataRoot(args.data_root)
    if args.command == "materialize-databases":
        from .contracts import ToolResult
        from .data import materialize_databases

        value = materialize_databases(data, args.archive, args.output)
        return save_result(
            args.output,
            ToolResult(
                title="Pinned upstream databases materialized",
                summary="Reader-supplied local archive, only declared hash-matched members.",
                metrics=value,
                notes=["No archive/network acquisition or redistribution permission inferred."],
            ),
        )
    if args.command == "judge-export":
        from .judging import export_judge

        return export_judge(data, args.generations, args.output)
    if args.command == "prepare":
        from .training import prepare_native

        return prepare_native(args.base_dir, data, args.arm, args.output, device=args.device)
    if args.command == "preflight":
        from .training import preflight

        return preflight(
            args.base_dir,
            data,
            args.workspace,
            args.output,
            database_root=args.database_root,
            device=args.device,
            recipe=Recipe.read(args.config),
        )
    if args.command == "train":
        from .training import train

        return train(
            args.base_dir,
            data,
            args.workspace,
            args.output,
            args.arm,
            args.preflight,
            prepared=args.prepared,
            device=args.device,
            recipe=Recipe.read(args.config),
            resume=args.resume,
            pause_at_update=args.pause_at_update,
        )
    if args.command in ("evaluate", "generate-heldout"):
        from .evaluation import evaluate, generate_heldout

        kwargs = dict(
            prepared=args.prepared,
            training_run=args.training_run,
            native_only=args.native_only,
            device=args.device,
        )
        if args.command == "evaluate":
            kwargs.update(database_root=args.database_root, budget_seconds=args.budget_seconds)
            return evaluate(args.base_dir, data, args.output, args.arm, **kwargs)
        return generate_heldout(args.base_dir, data, args.output, args.arm, **kwargs)
    raise ValueError("Unsupported command")


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        result = dispatch(args)
        if args.command == "list" and not args.json:
            print("Arms: " + ", ".join(result["arms"]))
            for tool in result["tools"]:
                print(f"{tool['id']}: {tool['description']}")
        else:
            # Runtime artifacts hold full provenance/raw arrays. Keep console output bounded.
            console = result
            if args.command not in ("check", "list"):
                console = {
                    "status": result.get("status", "complete"),
                    "output": str(args.output),
                    "metrics": result.get("metrics", result),
                }
            print(
                json.dumps(console, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            )
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        print(
            json.dumps(
                {"status": "failed", "type": type(exc).__name__, "error": redact_error(exc)},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2
