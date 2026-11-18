"""Run one fixed-mask replication pair from a source-only checkout, offline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from remove_reset.artifacts import (  # noqa: E402
    atomic_json,
    now,
    read_json,
    sha256,
    source_identity,
    verify_bundle,
    write_json,
)
from remove_reset.config import Recipe  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True, help="NEW private directory")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    recipe = Recipe.from_document({**Recipe().document(), "seed": args.seed})
    base = args.base_dir.resolve(strict=True)
    raw = args.raw_dir.resolve(strict=True)
    work = args.workspace.resolve()
    if work.is_relative_to(ROOT / "data"):
        raise ValueError("Private outputs must stay outside the published data directory")
    work.mkdir(parents=True, exist_ok=False)
    logs = work / "logs"
    logs.mkdir()
    config = work / "recipe.json"
    write_json(config, recipe.document())
    write_json(work / "run_plan.json", {
        "format": "remove-reset-seed-pair-v1",
        "created_utc": now(),
        "source_identity": source_identity(),
        "runner_sha256": sha256(__file__),
        "base_dir": str(base),
        "raw_dir": str(raw),
        "recipe": recipe.document(),
        "arms_in_order": ["ordinary_lora", "quota_free_guided_removal"],
        "selection_policy": "frozen masks; no reranking or outcome-dependent launch",
        "endpoint": "full epoch, not best checkpoint; independent broad judgments required",
    })
    environment = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
        "PYTHONUNBUFFERED": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
    }
    status = {
        "status": "running", "pid": os.getpid(), "seed": args.seed,
        "started_utc": now(), "stages": [],
    }

    def stage(name, command, *, bundle=None, training=None, paused=False):
        command = list(map(str, command))
        record = {"name": name, "command": command, "started_utc": now(), "status": "running"}
        status["stages"].append(record)
        status["current_stage"] = name
        atomic_json(work / "pair_status.json", status)
        print(f"{now()} START {name}", flush=True)
        started = time.monotonic()
        with (logs / f"{name}.log").open("xb") as stream:
            process = subprocess.Popen(
                [sys.executable, *map(str, command)], cwd=ROOT, env=environment,
                stdout=stream, stderr=subprocess.STDOUT,
            )
            status["child_pid"] = process.pid
            atomic_json(work / "pair_status.json", status)
            code = process.wait()
        record.update(returncode=code, seconds=time.monotonic() - started, ended_utc=now())
        if code:
            record["status"] = "failed"
            raise RuntimeError(f"{name} exited {code}; inspect {logs / (name + '.log')}")
        if bundle:
            verify_bundle(bundle)
            if read_json(bundle / "result.json")["status"] != "complete":
                raise ValueError(f"{name} did not produce complete output")
        if training:
            pointer = read_json(training / "latest_report.json")
            directory = training / pointer["path"]
            verify_bundle(directory, complete=not paused)
            result = read_json(directory / "result.json")
            if result["status"] != ("paused" if paused else "complete"):
                raise ValueError(f"{name} did not reach its declared training endpoint")
            metrics = result["metrics"]
            if paused:
                if metrics["updates"] != 2 or metrics["complete"]:
                    raise ValueError("Pilot did not pause after exactly two updates")
            elif (metrics["updates"], metrics["examples"], metrics["target_tokens"]) != (
                3927, 62817, 2087265
            ):
                raise ValueError("Full-epoch accounting differs")
            if read_json(training / "recipe.json") != recipe.document():
                raise ValueError("Training seed or scientific recipe differs")
        record["status"] = "complete"
        atomic_json(work / "pair_status.json", status)
        print(f"{now()} COMPLETE {name} ({record['seconds']:.1f}s)", flush=True)

    try:
        stage("original-inputs", [ROOT / "data" / "scripts" / "validate.py"])
        prepared_sql = work / "prepared-sql"
        stage("prepare-sql", [
            ROOT / "data" / "scripts" / "prepare_sql.py",
            "--raw", raw, "--output", prepared_sql,
        ])
        data = work / "data"
        stage("runtime-data", [
            ROOT / "data" / "scripts" / "build_runtime.py",
            "--prepared", prepared_sql, "--tokenizer-dir", base, "--output", data,
        ])
        stage("tiny-seed-check", [
            "-m", "remove_reset", "tiny-check", "--seed", args.seed,
            "--output", work / "tiny-check",
        ], bundle=work / "tiny-check")
        databases = work / "databases"
        stage("databases", [
            "-m", "remove_reset", "materialize-databases", "--data-root", data,
            "--archive", raw / "spider-official-archive.zip", "--output", databases,
        ], bundle=databases)
        model = [
            "--data-root", data, "--base-dir", base, "--device", args.device,
        ]
        gate = work / "preflight"
        stage("preflight", [
            "-m", "remove_reset", "preflight", *model, "--workspace", work,
            "--database-root", databases, "--config", config, "--output", gate,
        ], bundle=gate)
        for arm in ("ordinary_lora", "quota_free_guided_removal"):
            intervention = []
            if arm != "ordinary_lora":
                native = work / "prepared" / arm
                stage("prepare-" + arm, [
                    "-m", "remove_reset", "prepare", *model, "--arm", arm,
                    "--output", native,
                ], bundle=native)
                intervention = ["--prepared", native]
            training = work / "training" / arm
            command = [
                "-m", "remove_reset", "train", *model, "--arm", arm, *intervention,
                "--workspace", work, "--config", config, "--preflight", gate,
                "--output", training,
            ]
            stage("pilot-" + arm, [*command, "--pause-at-update", "2"],
                  training=training, paused=True)
            stage("train-" + arm, [*command, "--resume"], training=training)
            evaluation = work / "evaluation" / arm
            stage("evaluate-" + arm, [
                "-m", "remove_reset", "evaluate", *model, "--arm", arm, *intervention,
                "--training-run", training, "--database-root", databases,
                "--output", evaluation,
            ], bundle=evaluation)
            heldout = work / "heldout" / arm
            stage("heldout-" + arm, [
                "-m", "remove_reset", "generate-heldout", *model, "--arm", arm,
                *intervention, "--training-run", training, "--output", heldout,
            ], bundle=heldout)
        analysis_config = work / "analysis-config.json"
        arms = ("ordinary_lora", "quota_free_guided_removal")
        write_json(analysis_config, {
            "format": "remove-reset-analysis-config-v1",
            "production": True,
            "data": "data",
            "evaluations": [
                {"arm": arm, "seed": args.seed, "path": f"evaluation/{arm}"} for arm in arms
            ],
            "training": [
                {"arm": arm, "seed": args.seed, "path": f"training/{arm}"} for arm in arms
            ],
        })
        analysis_output = work / "analysis"
        stage("analyze-pair", [
            ROOT / "scripts" / "analyze.py", "--config", analysis_config,
            "--output", analysis_output,
        ])
        verify_bundle(analysis_output, complete=False)
        analysis = read_json(analysis_output / "result.json")
        if analysis["status"] not in ("complete", "partial"):
            raise ValueError("Pair analysis failed")
        status.update(
            status="complete", ended_utc=now(),
            scope="Two trained arms, primary evaluations and gold-free broad responses; "
                  "broad-screen human scoring not performed.",
        )
        atomic_json(work / "pair_status.json", status)
        print(json.dumps(status, indent=2), flush=True)
        return 0
    except BaseException as exc:
        status.update(status="failed", ended_utc=now(), error=str(exc))
        atomic_json(work / "pair_status.json", status)
        print(json.dumps(status, indent=2), flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
